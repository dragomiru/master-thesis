from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

# --- Models for Validating LLM's JSON Output ---

class NodeModel(BaseModel):
    """
    Represents a single node in the knowledge graph as extracted by the LLM.
    The 'id' is the unique identifier/name of the entity.
    The 'type' is the category of the entity (e.g., UniqueAccident, Date).
    """
    id: str = Field(..., description="The unique identifier or name of the node/entity.")
    type: str = Field(..., description="The type or category of the node/entity.")
    
class RelationshipModel(BaseModel):
    """
    Represents a relationship between two nodes in the knowledge graph as extracted by the LLM.
    'source' and 'target' refer to the 'id' of the NodeModels.
    """
    source: str = Field(..., description="The ID of the source node for the relationship.")
    target: str = Field(..., description="The ID of the target node for the relationship.")
    type: str = Field(..., description="The type or label of the relationship (e.g., 'occurred_in', 'has_date').")


class KnowledgeGraphModel(BaseModel):
    """
    Represents the overall structure of the knowledge graph extracted by the LLM.
    It contains a list of nodes and a list of relationships.
    """
    nodes: List[NodeModel] = Field(..., description="A list of all nodes in the graph.")
    rels: List[RelationshipModel] = Field(..., alias="rels", description="A list of all relationships in the graph.") # 'rels' is used in your notebook


# --- Example Usage and Validation Function ---
def validate_llm_output(json_data: Dict[str, Any]) -> Optional[KnowledgeGraphModel]:
    """
    Validates the raw JSON data (as a Python dictionary) from the LLM against the KnowledgeGraphModel.

    Args:
        json_data (Dict[str, Any]): The parsed JSON data from the LLM.

    Returns:
        Optional[KnowledgeGraphModel]: A validated KnowledgeGraphModel instance if successful, None otherwise.
    """
    try:
        validated_graph = KnowledgeGraphModel.model_validate(json_data)
        print("[INFO] LLM output JSON is valid according to KnowledgeGraphModel schema.")
        return validated_graph
    except ValidationError as e:
        print(f"[ERROR] LLM output JSON validation failed:\n{e}")
        return None
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred during LLM output validation: {ex}")
        return None

if __name__ == '__main__':
    print("--- Testing validation/schemas.py ---")

    # Example 1: Valid JSON data
    valid_json_data = {
        "nodes": [
            {"id": "Accident123", "type": "UniqueAccident"},
            {"id": "2025-05-09", "type": "Date"},
            {"id": "Main Street Crossing", "type": "TrackSection"}
        ],
        "rels": [
            {"source": "Accident123", "target": "2025-05-09", "type": "HAS_DATE"},
            {"source": "Accident123", "target": "Main Street Crossing", "type": "OCCURRED_AT_SECTION"}
        ]
    }
    print("\nTesting with valid JSON data:")
    graph_model_instance = validate_llm_output(valid_json_data)
    if graph_model_instance:
        print(f"Validated Graph Model: {graph_model_instance.model_dump_json(indent=2)}")
        assert len(graph_model_instance.nodes) == 3
        assert len(graph_model_instance.rels) == 2

    # Example 2: Invalid JSON data (missing 'type' in a node)
    invalid_json_data_node = {
        "nodes": [
            {"id": "Accident456"}, # Missing 'type'
            {"id": "2025-05-10", "type": "Date"}
        ],
        "rels": [
            {"source": "Accident456", "target": "2025-05-10", "type": "HAS_DATE"}
        ]
    }
    print("\nTesting with invalid JSON data (node missing 'type'):")
    validate_llm_output(invalid_json_data_node)

    # Example 3: Invalid JSON data (relationship missing 'source')
    invalid_json_data_rel = {
        "nodes": [
            {"id": "Accident789", "type": "UniqueAccident"},
            {"id": "2025-05-11", "type": "Date"}
        ],
        "rels": [
            {"target": "2025-05-11", "type": "HAS_DATE"}
        ]
    }
    print("\nTesting with invalid JSON data (relationship missing 'source'):")
    validate_llm_output(invalid_json_data_rel)

    # Example 4: Invalid top-level structure (missing 'nodes')
    invalid_json_data_top_level = {
        "rels": [
            {"source": "AccidentABC", "target": "DateXYZ", "type": "HAS_DATE"}
        ]
    }
    print("\nTesting with invalid JSON data (missing 'nodes' field):")
    validate_llm_output(invalid_json_data_top_level)

    print("--------------------------------------")