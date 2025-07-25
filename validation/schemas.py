# --- General modules ---
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
