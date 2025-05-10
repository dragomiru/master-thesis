import os
import json
from typing import Dict, Optional

from langchain_core.documents import Document as LangchainDocument
from neo4j import GraphDatabase, Driver, exceptions as neo4j_exceptions
from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node as LangchainBaseNode,
    Relationship as LangchainBaseRelationship
)


# --- Driver Management ---
def get_neo4j_driver(uri: str, user: str, password: str) -> Optional[Driver]:
    """
    Creates and returns a Neo4j driver instance.
    Tests the connection upon creation.
    """
    print(f"[NEO4J_HANDLER] Attempting to connect to Neo4j: URI='{uri}', User='{user}'")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("[INFO] Neo4j driver created and connectivity verified successfully.")
        return driver
    except neo4j_exceptions.AuthError as e:
        print(f"[ERROR] Neo4j Authentication failed for URI '{uri}'. Please check credentials.")
        print(f"   Neo4j driver error details: {e}")
    except neo4j_exceptions.ServiceUnavailable as e:
        print(f"[ERROR] Neo4j Service Unavailable at URI '{uri}'. Is the database running and accessible?")
        print(f"   Neo4j driver error details: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while connecting to Neo4j at URI '{uri}': {e}")
        print(f"   Error type: {type(e)}")
    return None

def close_neo4j_driver(driver: Optional[Driver]):
    """Closes the Neo4j driver if it's open."""
    if driver:
        try:
            driver.close()
            print("[INFO] Neo4j driver closed.")
        except Exception as e:
            print(f"[ERROR] Error closing Neo4j driver: {e}")

# --- Database Operations ---
def clear_neo4j_database(driver: Driver, database_name: str):
    """Deletes all nodes and relationships in the specified Neo4j database."""
    if not driver:
        print("[ERROR] Neo4j driver not available for clearing database.")
        return
    try:
        with driver.session(database=database_name) as session:
            print(f"[INFO] Attempting to clear database: '{database_name}'")
            session.run("MATCH (n) DETACH DELETE n")
            print(f"[INFO] Neo4j database '{database_name}' cleared successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to clear Neo4j database '{database_name}': {e}")

# --- Mapping Logic (from LLM JSON to Langchain GraphDocument) ---
def _map_llm_node_to_langchain_node(node_data: Dict) -> LangchainBaseNode:
    node_id = node_data.get("id", f"missing_id_{hash(json.dumps(node_data))}")
    node_type = node_data.get("type", "Unknown").capitalize()
    
    properties = node_data.copy()
    if 'id' not in properties and node_data.get("id"): # Ensure original 'id' is also a property
        properties['id_original'] = node_data.get("id") # Store original id if we generated a fallback for node_id

    return LangchainBaseNode(id=node_id, type=node_type, properties=properties)

def _map_llm_rel_to_langchain_rel(rel_data: Dict, nodes_map: Dict[str, LangchainBaseNode]) -> Optional[LangchainBaseRelationship]:
    source_id = rel_data.get("source")
    target_id = rel_data.get("target")
    rel_type = rel_data.get("type", "RELATED_TO").upper().replace(" ", "_")

    if not source_id or not target_id:
        print(f"[WARNING] Relationship missing source ('{source_id}') or target ('{target_id}') ID: {rel_data}")
        return None
        
    source_node = nodes_map.get(source_id)
    target_node = nodes_map.get(target_id)

    if not source_node:
        print(f"[WARNING] Source node with ID '{source_id}' not found in nodes_map for relationship: {rel_data}")
        return None
    if not target_node:
        print(f"[WARNING] Target node with ID '{target_id}' not found in nodes_map for relationship: {rel_data}")
        return None
    
    properties = {k: v for k, v in rel_data.items() if k not in ["source", "target", "type"]}

    return LangchainBaseRelationship(
        source=source_node,
        target=target_node,
        type=rel_type,
        properties=properties
    )

def map_json_to_graph_document(llm_json_output: Dict, source_doc_text: str = "") -> Optional[GraphDocument]:
    if not isinstance(llm_json_output, dict) or "nodes" not in llm_json_output or "rels" not in llm_json_output:
        print(f"[ERROR] Invalid JSON structure for graph document mapping. Expected dict with 'nodes' and 'rels'. Got: {type(llm_json_output)}")
        return None

    langchain_nodes_list = []
    nodes_id_to_object_map: Dict[str, LangchainBaseNode] = {}
    for node_data in llm_json_output.get("nodes", []):
        if not isinstance(node_data, dict) or "id" not in node_data or "type" not in node_data:
            print(f"[WARNING] Skipping invalid node data (must be dict with 'id' and 'type'): {node_data}")
            continue
        lnode = _map_llm_node_to_langchain_node(node_data)
        langchain_nodes_list.append(lnode)
        nodes_id_to_object_map[lnode.id] = lnode # Assumes lnode.id is the same as node_data.get("id")
    
    if not langchain_nodes_list:
        print("[WARNING] No valid nodes mapped from JSON to create graph document.")

    langchain_relationships_list = []
    for rel_data in llm_json_output.get("rels", []):
        if not isinstance(rel_data, dict):
            print(f"[WARNING] Skipping invalid relationship data (must be dict): {rel_data}")
            continue
        lrel = _map_llm_rel_to_langchain_rel(rel_data, nodes_id_to_object_map)
        if lrel:
            langchain_relationships_list.append(lrel)

    return GraphDocument(
        nodes=langchain_nodes_list,
        relationships=langchain_relationships_list,
        source=LangchainDocument(page_content=source_doc_text)
    )

# --- Storing GraphDocument in Neo4j ---
def store_graph_document_in_neo4j(driver: Driver, graph_document: GraphDocument, database_name: str):
    if not driver:
        print("[ERROR] Neo4j driver not available for storing graph document.")
        return
    if not graph_document:
        print("[WARNING] Invalid or empty graph document provided. Nothing to store in Neo4j.")
        return
    if not graph_document.nodes:
        print("[INFO] Graph document has no nodes. Nothing to store in Neo4j.")

    try:
        with driver.session(database=database_name) as session:
            print(f"[INFO] Storing graph document in Neo4j database: '{database_name}'")
            nodes_written = 0
            for node in graph_document.nodes:
                cypher_props = {k: v for k, v in node.properties.items() if v is not None}
                # The LangchainBaseNode 'id' is its unique identifier for linking within GraphDocument.
                # We use node.properties['id'] (which should match node.id if mapped correctly) for MERGE.
                node_merge_id = node.properties.get('id', node.id)
                if node_merge_id is None:
                    print(f"[WARNING] Node missing 'id' in properties, cannot merge reliably: {node}")
                    continue


                session.run(f"""
                    MERGE (n:{node.type} {{id: $id_param}})
                    ON CREATE SET n = $props
                    ON MATCH SET n += $props 
                """, id_param=node_merge_id, props=cypher_props)
                nodes_written += 1
            print(f"[INFO] Stored/Merged {nodes_written} nodes in Neo4j.")

            # Store relationships
            rels_written = 0
            for rel in graph_document.relationships:
                source_merge_id = rel.source.properties.get('id', rel.source.id)
                target_merge_id = rel.target.properties.get('id', rel.target.id)
                
                if source_merge_id is None or target_merge_id is None:
                    print(f"[WARNING] Relationship source/target node missing 'id' in properties, cannot merge: {rel}")
                    continue

                cypher_rel_props = {k:v for k,v in rel.properties.items() if v is not None}
                
                session.run(f"""
                    MATCH (s:{rel.source.type} {{id: $source_id}})
                    MATCH (t:{rel.target.type} {{id: $target_id}})
                    MERGE (s)-[r:{rel.type}]->(t)
                    ON CREATE SET r = $props
                    ON MATCH SET r += $props
                """, source_id=source_merge_id, target_id=target_merge_id, props=cypher_rel_props)
                rels_written +=1
            print(f"[INFO] Stored/Merged {rels_written} relationships in Neo4j.")

    except Exception as e:
        print(f"[ERROR] Failed to store graph document in Neo4j: {e}")

# --- Orchestration Function ---
def process_and_store_graph(
    driver: Driver,
    llm_json_output: Dict,
    source_text: str,
    database_name: str
):
    if not driver:
        print("[ERROR] Neo4j driver not provided for processing and storing graph.")
        return

    print("[INFO] Converting LLM JSON to graph document...")
    graph_document = map_json_to_graph_document(llm_json_output, source_text)

    if graph_document:
        print("[INFO] Graph document created. Storing in Neo4j...")
        store_graph_document_in_neo4j(driver, graph_document, database_name)
        print("[INFO] Graph data processing and attempt to store in Neo4j completed.")
    else:
        print("[WARNING] Graph document creation failed. Nothing stored in Neo4j.")


# if __name__ == '__main__':
#     print("--- Testing storage/neo4j_handler.py ---")
    
#     # Load from a .env file if present (for NEO4J_TEST_PASSWORD)
#     from dotenv import load_dotenv
#     load_dotenv() # Looks for .env in the current directory or parent directories

#     NEO4J_URI_TEST = os.getenv("NEO4J_URI", "bolt://localhost:7687") # Use config default if test not set
#     NEO4J_USER_TEST = os.getenv("NEO4J_USER", "neo4j")
#     NEO4J_PASSWORD_TEST = os.getenv("NEO4J_PASSWORD", "password") # Ensure this is in your .env or set directly
#     NEO4J_DATABASE_TEST = os.getenv("NEO4J_DATABASE_TEST", "neo4jtest") # Use a dedicated test database

#     print(f"Attempting to connect to Neo4j for testing: URI='{NEO4J_URI_TEST}', User='{NEO4J_USER_TEST}', DB='{NEO4J_DATABASE_TEST}'")
    
#     test_driver = get_neo4j_driver(NEO4J_URI_TEST, NEO4J_USER_TEST, NEO4J_PASSWORD_TEST)

#     if test_driver:
#         print("\n[STEP 1] Clearing test database (if it exists)...")
#         clear_neo4j_database(test_driver, NEO4J_DATABASE_TEST)

#         print("\n[STEP 2] Testing graph mapping and storage...")
#         sample_llm_output = {
#             "nodes": [
#                 {"id": "Accident_XYZ", "type": "UniqueAccident", "description": "Major collision event", "year": 2025},
#                 {"id": "2025-05-09", "type": "Date", "day_of_week": "Friday"},
#                 {"id": "Human Error Factor", "type": "ContributingFactor", "details": "Operator fatigue"}
#             ],
#             "rels": [
#                 {"source": "Accident_XYZ", "target": "2025-05-09", "type": "HAS_DATE", "certainty": "high"},
#                 {"source": "Accident_XYZ", "target": "Human Error Factor", "type": "CAUSED_BY_FACTOR"}
#             ]
#         }
#         sample_source_text = "A major collision (Accident_XYZ) occurred on 2025-05-09 due to Human Error Factor."

#         process_and_store_graph(test_driver, sample_llm_output, sample_source_text, NEO4J_DATABASE_TEST)
        
#         print("\n[INFO] Test data processing complete. Check your Neo4j browser for the '{}' database.".format(NEO4J_DATABASE_TEST))
#         print("   Sample Cypher to check nodes: MATCH (n) RETURN n LIMIT 5")
#         print("   Sample Cypher to check relationships: MATCH ()-[r]->() RETURN r LIMIT 5")
        
#         close_neo4j_driver(test_driver)
#     else:
#         print("[FAIL] Could not connect to Neo4j. Aborting Neo4j handler tests.")
#         print("   Ensure Neo4j is running and credentials are correct in your environment or .env file:")
#         print(f"   NEO4J_URI='{NEO4J_URI_TEST}', NEO4J_USER='{NEO4J_USER_TEST}', NEO4J_PASSWORD='<your_password>', NEO4J_DATABASE_TEST='{NEO4J_DATABASE_TEST}'")
    
#     print("-------------------------------------")