import streamlit as st
from neo4j import exceptions as neo4j_exceptions
import graphviz

import sys
import pathlib
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from storage import neo4j_handler

# --- Page Configuration ---
st.set_page_config(
    page_title="Neo4j Graph Viewer (Static)",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Neo4j Static Graph Viewer")
st.markdown("Visualize the knowledge graph from Neo4j as a static image using Graphviz.")
st.markdown("---")

# --- Visualization Helper Function (using Graphviz) ---
# In pages/02_Graph_Explorer.py or visualization/graph_visualizer.py

import graphviz
import json # For debug printing if needed

def convert_neo4j_records_to_dot_source(records_as_dicts, graph_name="Neo4jGraph") -> str:
    """
    Converts Neo4j records (containing nodes and/or relationships, as structured by record.data())
    into a DOT language string suitable for Graphviz.
    """
    print(f"[DOT_CONVERTER] Processing {len(records_as_dicts)} records (each representing one RETURNED item).")
    dot = graphviz.Digraph(graph_name, comment='Knowledge Graph', engine='dot')
    dot.attr(rankdir='LR', splines='true', overlap='false', concentrate='true', K='0.6') # Added K for spring constant
    dot.attr('node', shape='ellipse', style='filled', color='skyblue2', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9', color='gray50', arrowsize='0.7')

    unique_nodes_data_by_element_id = {}  # Key: neo4j_element_id (str), Value: node_data_dict
    node_map_neo_to_dot = {}            # Key: neo4j_element_id (str), Value: dot_id (str, e.g., "N0")
    dot_node_counter = 0

    def get_or_create_dot_id(neo4j_element_id_str: str) -> str:
        nonlocal dot_node_counter
        if neo4j_element_id_str not in node_map_neo_to_dot:
            node_map_neo_to_dot[neo4j_element_id_str] = f"N{dot_node_counter}"
            dot_node_counter += 1
        return node_map_neo_to_dot[neo4j_element_id_str]

    # --- Pass 1: Collect all unique node data and map their element_ids to DOT IDs ---
    # Each `record_dict` in `records_as_dicts` is like {"n": node_details, "r": rel_details, "m": node_details}
    for i, record_dict in enumerate(records_as_dicts):
        # print(f"[DOT_CONVERTER_P1] Record {i}: {record_dict}")
        for alias_key, entity_data_from_record in record_dict.items(): # alias_key is 'n', 'r', 'm', etc.
            # entity_data_from_record is the actual node/rel dict or list (for paths)
            
            items_to_inspect = []
            if isinstance(entity_data_from_record, list): # If a path or list of entities was returned under one alias
                items_to_inspect.extend(entity_data_from_record)
            elif isinstance(entity_data_from_record, dict): # If a single entity was returned under an alias
                items_to_inspect.append(entity_data_from_record)
            
            for entity_data in items_to_inspect:
                if not isinstance(entity_data, dict) or 'elementId' not in entity_data : # Neo4j Bloom JSON uses "elementId"
                    # print(f"[DOT_CONVERTER_P1] Skipping item, not a valid entity dict or no elementId: {str(entity_data)[:100]}")
                    continue

                # Check if it's a node based on the presence of 'labels'
                if 'labels' in entity_data: # This is a NODE
                    neo4j_elem_id = str(entity_data.get('elementId')) # Use "elementId"
                    if neo4j_elem_id not in unique_nodes_data_by_element_id:
                        # print(f"[DOT_CONVERTER_P1] Discovered Node (EID: {neo4j_elem_id}) via alias '{alias_key}'")
                        unique_nodes_data_by_element_id[neo4j_elem_id] = entity_data
                    get_or_create_dot_id(neo4j_elem_id) # Ensure DOT ID is mapped

    # --- Pass 2: Add collected unique nodes to the DOT graph ---
    print(f"[DOT_CONVERTER_P2] Adding {len(unique_nodes_data_by_element_id)} unique nodes to DOT graph.")
    if not unique_nodes_data_by_element_id:
        print("[DOT_CONVERTER_P2] No unique nodes found to add to the graph.")

    for neo4j_elem_id, node_data in unique_nodes_data_by_element_id.items():
        dot_id = node_map_neo_to_dot.get(neo4j_elem_id) # Should exist from Pass 1
        if not dot_id: # Should not happen if logic is correct
            print(f"[ERROR_DOT_CONVERTER] DOT ID not found for Neo4j Element ID: {neo4j_elem_id}")
            continue
            
        node_properties = node_data.get('properties', {})
        node_labels_list = list(node_data.get('labels', []))
        
        # Determine display label for the node in Graphviz
        display_label_prop = node_properties.get("id", node_properties.get("name")) # Prefer 'id' or 'name' from properties
        node_display_text = str(display_label_prop if display_label_prop is not None else (node_labels_list[0] if node_labels_list else dot_id))
        
        full_node_label = f"{node_display_text}\n:{':'.join(node_labels_list)}" if node_labels_list else node_display_text
        
        # print(f"[DOT_CONVERTER_P2] Adding NODE to DOT: DOT_ID='{dot_id}', Label='{full_node_label}' (Neo4j EID: {neo4j_elem_id})")
        dot.node(dot_id, label=full_node_label)

    # --- Pass 3: Collect and add relationships to the DOT graph ---
    edges_added_count = 0
    print(f"[DOT_CONVERTER_P3] Processing relationships...")
    for i, record_dict in enumerate(records_as_dicts):
        for _alias, entity_data_from_record in record_dict.items():
            items_to_inspect = []
            if isinstance(entity_data_from_record, list): items_to_inspect.extend(entity_data_from_record)
            elif isinstance(entity_data_from_record, dict): items_to_inspect.append(entity_data_from_record)

            for entity_data in items_to_inspect:
                if not isinstance(entity_data, dict) or 'elementId' not in entity_data:
                    continue

                # Check if it's a relationship
                if 'type' in entity_data and 'startNodeElementId' in entity_data and 'endNodeElementId' in entity_data: # Neo4j Bloom JSON uses these for rels
                    start_neo4j_eid = str(entity_data.get('startNodeElementId'))
                    end_neo4j_eid = str(entity_data.get('endNodeElementId'))
                    rel_type = entity_data.get('type', 'RELATED_TO')
                    
                    if start_neo4j_eid in node_map_neo_to_dot and end_neo4j_eid in node_map_neo_to_dot:
                        dot_start_id = node_map_neo_to_dot[start_neo4j_eid]
                        dot_end_id = node_map_neo_to_dot[end_neo4j_eid]
                        # print(f"[DOT_CONVERTER_P3] Adding EDGE to DOT: From='{dot_start_id}' (Neo4j StartEID: {start_neo4j_eid}), To='{dot_end_id}' (Neo4j EndEID: {end_neo4j_eid}), Label='{rel_type}'")
                        dot.edge(dot_start_id, dot_end_id, label=rel_type)
                        edges_added_count += 1
                    else:
                        print(f"[DOT_CONVERTER_P3] Skipping EDGE: Start EID '{start_neo4j_eid}' or End EID '{end_neo4j_eid}' not mapped to DOT ID. Rel data: {entity_data}")
    
    final_dot_source = dot.source
    # Use a different way to count nodes in dot object if body_json is not available in your graphviz version
    num_nodes_in_dot = len(node_map_neo_to_dot) # This reflects unique nodes prepared for DOT
    
    print(f"[DOT_CONVERTER] DOT source generation complete. Unique nodes processed for DOT: {num_nodes_in_dot}, Edges added to DOT: {edges_added_count}")
    if not num_nodes_in_dot and not edges_added_count: # Check if any actual elements were prepared for DOT
        print("[DOT_CONVERTER] Warning: No nodes or edges were effectively added to the DOT graph structure.")
    # For debugging the generated DOT:
    # if num_nodes_in_dot > 0 or edges_added_count > 0:
    #     print(f"Generated DOT source (first 1000 chars):\n{final_dot_source[:1000]}")
    return final_dot_source


# --- Main Page Logic ---
if 'neo4j_driver' not in st.session_state or st.session_state.neo4j_driver is None:
    st.session_state.neo4j_driver = neo4j_handler.get_neo4j_driver(
        config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD
    )

if st.session_state.neo4j_driver:
    st.sidebar.success("Neo4j: Connected")
    
    default_query = "MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 15"
    custom_query = st.text_area(
        "Enter Cypher Query to Visualize:",
        default_query,
        height=100,
        key="static_graph_query_input_graphviz",
        help="Query should ideally return nodes and relationships (e.g., `MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 15`)."
    )

    if st.button("Generate Static Graph", type="primary", key="generate_static_graph_button"):
        if not custom_query.strip():
            st.warning("Please enter a Cypher query.")
        else:
            with st.spinner("Fetching data and generating graph image..."):
                try:
                    with st.session_state.neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                        result = session.run(custom_query)
                        records_as_dicts = [record.data() for record in result]

                    if records_as_dicts:
                        st.write(f"Query returned {len(records_as_dicts)} records/paths.") # Info for user
                        dot_source = convert_neo4j_records_to_dot_source(records_as_dicts)
                        if dot_source and dot_source.strip() != "digraph Neo4jGraph {}": # Check if DOT source is not empty
                            st.success("Graph DOT source generated. Rendering...")
                            st.graphviz_chart(dot_source)
                            with st.expander("View Generated DOT Source"):
                                st.code(dot_source, language="dot")
                        elif dot_source.strip() == "digraph Neo4jGraph {}":
                             st.info("Query results led to an empty graph (no nodes or edges added to DOT).")
                        else:
                            st.warning("Could not generate valid DOT source from query results.")
                    else:
                        st.info("Your query returned no data to visualize.")
                except neo4j_exceptions.CypherSyntaxError as cse:
                    st.error(f"Cypher Syntax Error: {cse}")
                except neo4j_exceptions.ServiceUnavailable:
                    st.error("Neo4j service is unavailable. Please ensure the database is running.")
                    st.session_state.neo4j_driver = None 
                    st.rerun()
                except neo4j_exceptions.AuthError:
                    st.error("Neo4j authentication error. Please check credentials.")
                    st.session_state.neo4j_driver = None
                    st.rerun()
                except FileNotFoundError: # Specifically for graphviz executable not found
                    st.error(
                        "Graphviz executable not found. "
                        "Please ensure Graphviz is installed and its 'bin' directory is in your system's PATH. "
                        "Download from: https://graphviz.org/download/"
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.exception(e)
else:
    st.error("Neo4j connection failed or not configured. Cannot visualize graph.")
    if st.button("Attempt to Reconnect to Neo4j"):
        if 'neo4j_driver' in st.session_state:
            del st.session_state['neo4j_driver']
        st.rerun()