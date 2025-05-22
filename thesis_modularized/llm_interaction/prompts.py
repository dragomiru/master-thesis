from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from typing import List

# JSON schema example for few-shot learning
SCHEMA_EXAMPLE = """{
    "nodes": [
        {"id": "Dublin-Cork Accident", "type": "UniqueAccident"},
        {"id": "Level Crossing Accident involving a train", "type": "AccidentType"},
        {"id": "105 MP-108 MP", "type": "TrackSection"},
        {"id": "23/12/2021", "type": "Date"},
        {"id": "16:32", "type": "Time"},
        {"id": "Ireland", "type": "Country"},
        {"id": "Complexity", "type": "ContributingFactor"},
        {"id": "Environment", "type": "ContributingFactor"},
        {"id": "Fatigue", "type": "ContributingFactor"},
        {"id": "Leadership and commitment", "type": "SystemicFactor"},
        {"id": "Safety objectives and planning", "type": "SystemicFactor"},
        {"id": "Information and communication", "type": "SystemicFactor"},
        {"id": "European Rail Agency", "type": "RegulatoryBody"},
    ],
    "rels": [
        {"source": "Dublin-Cork Accident", "target": "Ireland", "type": "occurred_in"},
        {"source": "Dublin-Cork Accident", "target": "Collision", "type": "is_type"},
        {"source": "Dublin-Cork Accident", "target": "105 MP-108 MP", "type": "is_track_section"},
        {"source": "Dublin-Cork Accident", "target": "European Rail Agency", "type": "investigated_by"},
        {"source": "Dublin-Cork Accident", "target": "23/12/2021", "type": "has_date"},
        {"source": "Dublin-Cork Accident", "target": "16:32", "type": "has_time"},
        {"source": "Dublin-Cork Accident", "target": "Complexity", "type": "contributing_factor"},
        {"source": "Dublin-Cork Accident", "target": "Environment", "type": "contributing_factor"},
        {"source": "Dublin-Cork Accident", "target": "Fatigue", "type": "contributing_factor"},
        {"source": "Dublin-Cork Accident", "target": "Leadership and commitment", "type": "systemic_factor"},
        {"source": "Dublin-Cork Accident", "target": "Safety objectives and planning", "type": "systemic_factor"},
        {"source": "Dublin-Cork Accident", "target": "Information and communication", "type": "systemic_factor"}
    ]
}"""

ENTITY_TYPES_OF_INTEREST = [
    "UniqueAccident", "AccidentType", "TrackSection", "Date", "Time",
    "Country", "RegulatoryBody", "ContributingFactor", "SystemicFactor"
]

def create_extraction_prompt(entities_of_interest: List[str] = ENTITY_TYPES_OF_INTEREST, 
                             schema_example: str = SCHEMA_EXAMPLE) -> ChatPromptTemplate:
    """
    Creates the ChatPromptTemplate for the initial knowledge extraction task.
    """
    # Input variables are the ones that will be passed when chain.invoke is called, while
    # partial variables are the ones that are set in the prompt template and will be filled in at runtime
    runtime_input_variables = ["relevant_report_text"] 

    extraction_chat_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are an expert in analyzing railway accident reports. Follow the JSON schema provided."
),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(
                """
                Analyze the following railway accident report context and extract structured knowledge in JSON format.

                Return a JSON object with:
                - `nodes`: A list of entities, specifically {entities_of_interest_str_list}.
                - `rels`: A list of relationships linking entities.

                Guidelines:
                - Look at the JSON schema example response and follow it closely. 
                - Ensure that the `source` and `target` nodes in `rels` are the SAME entities from the `nodes` list, and NOT different ones. 
                - Make sure to map ALL nodes with other important entities, e.g., (node UniqueAccident has_date node Date, node UniqueAccident occurred_at node Country).
                - Do NOT map entities like (node Date is_date to node Time) or (node AccidentType is_type to node Country). This is INCORRECT.
                - The `type` field in `rels` should be a verb phrase (e.g., "occurred_in", "investigated_by").
                - The `id` field in `nodes` should be the exact text of the entity, not a description or a summary.
                - Pay attention to date and time formats (e.g., HH:MM, and DD/MM/YY).
                - The `UniqueAccident` entity should be a unique identifier for the accident.
                - The `AccidentType` entity should be the type of accident.
                - The `TrackSection` entity should be the track section where the accident occurred.
                - The `Country` entity should be the country where the accident occurred.
                - The `ContributingFactor` entity should be a factor contributing to the accident.
                - The `SystemicFactor` entity should be a systemic factor contributing to the accident.
                - The `RegulatoryBody` entity should be the regulatory body that investigated the accident.

                Schema example:
                {schema_example_str}

                Railway accident report context:
                {relevant_report_text}

                Strictly provide ONLY the JSON output, without any markdown formatting or other explanatory text.
                JSON:
                """
            )
        ],
        input_variables=runtime_input_variables,
        partial_variables={
            "entities_of_interest_str_list": str(entities_of_interest),
            "schema_example_str": schema_example
        }
    )
    return extraction_chat_prompt


def create_refinement_prompt(relevant_events_text: str, true_contr_fact_str: str,  true_sys_fact_str: str) -> ChatPromptTemplate:
    runtime_input_variables = ["extraction_result_str"]

    refinement_chat_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are an expert in refining railway accident report extractions based on provided standardized options."
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(
                """
                Your previous JSON extraction needs refinement for `AccidentType`, `ContributingFactor`, and `SystemicFactor` nodes.
                Update your previous JSON response using the standardized options provided below.

                1.  **AccidentType**: Choose ONLY ONE relevant accident category from the options below. Replace the existing `AccidentType` node's `id` in your JSON with the chosen standardized `AccidentType` name, copying it word for word.
                    AccidentType Options:
                    {relevant_events_text_str}

                2.  **ContributingFactor(s)**: Second, for ContributingFactor, the key in the dictionary is your previous prediction from the JSON, and the value is the correct factor you need to replace the value in the JSON with. Copy the factor exactly WORD FOR WORD into the JSON accordingly and replace the ContributingFactor ONLY (both in `nodes` and `rels`).
                    Standardized ContributingFactor(s):
                    {standardized_contr_fact_str}

                3.  **SystemicFactor(s)**: Similarly, update ALL `SystemicFactor` nodes in your JSON with these correct standardized names.
                    Standardized SystemicFactor(s):
                    {standardized_sys_fact_str}

                Your previous JSON extraction:
                {extraction_result_str}

                Provide only the updated JSON with the required changes, without any additional comments or text. If the factors repeat, then keep only one instance of each factor:
                """
            )
        ],
        input_variables=runtime_input_variables, 
        partial_variables={
            "relevant_events_text_str": relevant_events_text,
            "standardized_contr_fact_str": true_contr_fact_str,
            "standardized_sys_fact_str": true_sys_fact_str
        }
    )
    return refinement_chat_prompt