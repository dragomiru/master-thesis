import streamlit as st
import os
import json
import tempfile
import pandas as pd
import re

import config

# Import modules from our project structure
from data_processing import pdf_extractor, text_splitter, data_loaders
from vector_store import faiss_handler
from llm_interaction import llm_models, prompts, chains
from validation import schemas as validation_schemas
from storage import csv_logger, neo4j_handler

import streamlit.components.v1 as components
from neo4j import exceptions as neo4j_exceptions

# --- Page Configuration ---
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_FAVICON,
    layout="wide"
)

# --- Functions ---
@st.cache_resource
def load_embeddings_model(model_name):
    print(f"Cache miss: Loading embeddings model ({model_name})...")
    return faiss_handler.get_embeddings_model(model_name)

@st.cache_data
def load_static_data():
    print("Cache miss: Loading static data (categories, factors)...")
    categories_df = data_loaders.load_accident_categories(
        config.CATEGORY_A_EVENTS_CSV,
        config.CATEGORY_B_EVENTS_CSV,
        config.CATEGORY_C_EVENTS_CSV
    )
    contributing_factors_df = data_loaders.load_contributing_factors(config.CONTRIBUTING_FACTORS_CSV)
    systemic_factors_df = data_loaders.load_systemic_factors(config.SYSTEMIC_FACTORS_CSV)

    categories_dicts = data_loaders.convert_df_to_dicts(categories_df)
    contr_fact_dicts = data_loaders.convert_df_to_dicts(contributing_factors_df)
    sys_fact_dicts = data_loaders.convert_df_to_dicts(systemic_factors_df)
    
    return categories_dicts, contr_fact_dicts, sys_fact_dicts

def initialize_llm(_model_identifier): # Arguments must be hashable for cache
    print(f"Cache miss: Initializing LLM ({_model_identifier})...") # For debugging cache
    return llm_models.init_llm(model_identifier=_model_identifier)

@st.cache_resource
def get_neo4j_db_driver():
    print("Cache miss: Getting Neo4j driver...")
    if config.ENABLE_NEO4J_STORAGE:
        driver = neo4j_handler.get_neo4j_driver(
            config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD
        )
        return driver
    return None

# --- Functions for static vector stores ---
@st.cache_resource
def get_cached_vectorstore_categories(_static_categories_data, _embeddings_model):
    print("Cache miss: Creating categories vector store...")
    if not _static_categories_data or not _embeddings_model:
        st.error("Cannot create categories vector store: missing data or embeddings model.")
        return None
    event_chunks = text_splitter.split_events_into_chunks(
        _static_categories_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP
    )
    return faiss_handler.create_faiss_store_from_document_lists(
        event_chunks, _embeddings_model
    )

@st.cache_resource
def get_cached_vectorstore_contr_factors(_static_contr_factors_data, _embeddings_model):
    print("Cache miss: Creating contributing factors vector store...")
    if not _static_contr_factors_data or not _embeddings_model:
        st.error("Cannot create contributing factors vector store: missing data or embeddings model.")
        return None
    contr_factor_chunks = text_splitter.split_factors_into_chunks(
        _static_contr_factors_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP
    )
    return faiss_handler.create_faiss_store_from_document_lists(
        contr_factor_chunks, _embeddings_model
    )

@st.cache_resource
def get_cached_vectorstore_sys_factors(_static_sys_factors_data, _embeddings_model):
    print("Cache miss: Creating systemic factors vector store...")
    if not _static_sys_factors_data or not _embeddings_model:
        st.error("Cannot create systemic factors vector store: missing data or embeddings model.")
        return None
    sys_factor_chunks = text_splitter.split_factors_into_chunks(
        _static_sys_factors_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP
    )
    return faiss_handler.create_faiss_store_from_document_lists(
        sys_factor_chunks, _embeddings_model
    )

# --- Main Application UI ---
st.title(f"{config.APP_FAVICON} {config.APP_TITLE}")
with st.sidebar:
    st.header("⚙️ Processing Controls")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    available_llms = ["gemini-2.5-flash-preview-04-17", "gemini-1.5-flash-latest", "gpt-4o-mini"] 
    try:
        default_llm_index = available_llms.index(config.DEFAULT_LLM_MODEL)
    except ValueError:
        default_llm_index = 0
    
    selected_llm_model = st.selectbox(
        "Select LLM Model:",
        options=available_llms,
        index=default_llm_index,
        help="Choose the language model for extraction."
    )

    if 'enable_neo4j' not in st.session_state:
        st.session_state.enable_neo4j = config.ENABLE_NEO4J_STORAGE

    st.session_state.enable_neo4j = st.checkbox(
        "Store results in Neo4j", value=st.session_state.enable_neo4j
    )

    process_button = st.button("Process Uploaded PDF(s)", type="primary", disabled=not uploaded_files)
    
    st.markdown("---")
    st.info(f"""
    **Current Settings:**
    - LLM: `{selected_llm_model}`
    - Neo4j Storage: `{'Enabled' if st.session_state.enable_neo4j else 'Disabled'}`
    """)

# --- Load shared resources ---
embeddings_model = load_embeddings_model(config.EMBEDDINGS_MODEL_NAME)
static_categories_dicts, static_contr_factors_dicts, static_sys_factors_dicts = load_static_data()


# --- Initialize Static Vector Stores using cached functions ---
vectorstore_categories = None
vectorstore_contr_factors = None
vectorstore_sys_factors = None

if embeddings_model and static_categories_dicts and static_contr_factors_dicts and static_sys_factors_dicts:
    with st.spinner("Initializing static vector stores (cached if inputs are same)..."):
        vectorstore_categories = get_cached_vectorstore_categories(static_categories_dicts, embeddings_model)
        vectorstore_contr_factors = get_cached_vectorstore_contr_factors(static_contr_factors_dicts, embeddings_model)
        vectorstore_sys_factors = get_cached_vectorstore_sys_factors(static_sys_factors_dicts, embeddings_model)

    if not (vectorstore_categories and vectorstore_contr_factors and vectorstore_sys_factors):
        st.error("Failed to initialize one or more static vector stores. Refinement step might be impacted.")
else:
    st.error("Failed to load embeddings model or static data. Cannot create static vector stores.")


# --- Processing Logic ---
if process_button and uploaded_files:
    if not embeddings_model:
        st.error("Embeddings model not loaded. Cannot process.")
        st.stop()

    llm = initialize_llm(selected_llm_model)
    if not llm:
        st.error(f"Failed to initialize LLM: {selected_llm_model}. Check API keys and model name.")
        st.stop()

    neo4j_driver = None
    if st.session_state.enable_neo4j:
        neo4j_driver = get_neo4j_db_driver()
        if not neo4j_driver:
            st.warning("Neo4j storage is enabled, but failed to connect. Results will not be stored in Neo4j.")

    for uploaded_file in uploaded_files:
        st.markdown(f"--- \n### Processing: `{uploaded_file.name}`")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_pdf_path = tmp_file.name
        
        with st.spinner(f"Extracting text from `{uploaded_file.name}`..."):
            pdf_text = pdf_extractor.extract_summary_section(tmp_pdf_path)
            if not pdf_text:
                st.error(f"Could not extract text from `{uploaded_file.name}`.")
                os.remove(tmp_pdf_path)
                continue

        with st.spinner("Chunking report text..."):
            report_chunks = text_splitter.split_report_into_chunks(
                pdf_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
            )
            if not report_chunks:
                st.error(f"Failed to chunk text for {uploaded_file.name}.")
                os.remove(tmp_pdf_path)
                continue
        
        with st.spinner("Creating report vector store..."): # This one is per-PDF, so no global cache
            vectorstore_report = faiss_handler.create_faiss_store_from_texts(report_chunks, embeddings_model)
            if not vectorstore_report:
                st.error(f"Failed to create report vector store for {uploaded_file.name}.")
                os.remove(tmp_pdf_path)
                continue
        
        with st.spinner("Finding relevant report chunks..."):
            entity_queries_for_report = {
                "unique accident": "Description of the unique accident, its name or main event",
                "accident type": "The general category or type of railway accident",
                "track section": "Specific location, station, or track segment where the accident happened",
                "date": "Date of the accident (DD/MM/YYYY)",
                "time": "Time of the accident (HH:MM 24-hour format)",
                "country": "Country where the accident took place",
                "regulatory body": "Name of the authority or body investigating the accident",
                "contributing factor": "Key factors that contributed to the accident",
                "systemic factor": "Underlying systemic issues or failures that led to the accident",
            }
            relevant_report_text = faiss_handler.find_most_relevant_report_chunks(
                vectorstore_report, entity_queries_for_report, config.TOP_K_RELEVANT_CHUNKS_REPORT
            )
            if not relevant_report_text:
                st.warning(f"No relevant chunks found in {uploaded_file.name} for initial extraction. Using full text.")
                relevant_report_text = pdf_text

        conversation_memory = llm_models.get_conversation_memory()

        with st.spinner("Performing initial knowledge extraction with LLM..."):
            extraction_prompt_template = prompts.create_extraction_prompt()
            extraction_chain = chains.create_llm_chain(llm, extraction_prompt_template, conversation_memory)
            extraction_result_raw = chains.run_chain(extraction_chain, {"relevant_report_text": relevant_report_text})

        if not extraction_result_raw:
            st.error(f"LLM initial extraction failed for {uploaded_file.name}.")
            os.remove(tmp_pdf_path)
            continue
        
        cleaned_extraction_json_str = re.sub(r'^```json\n?|```$', '', extraction_result_raw, flags=re.MULTILINE).strip()
        try:
            initial_llm_json = json.loads(cleaned_extraction_json_str)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse initial LLM JSON output for {uploaded_file.name}: {e}")
            os.remove(tmp_pdf_path)
            continue

        with st.spinner("Preparing inputs for LLM refinement step..."):
            extracted_accident_type_guess = ""
            extracted_contr_factors_guesses = []
            extracted_sys_factors_guesses = []

            for node in initial_llm_json.get("nodes", []):
                node_type = node.get("type")
                node_id = node.get("id", "")
                if node_type == "AccidentType":
                    extracted_accident_type_guess = node_id
                elif node_type == "ContributingFactor":
                    if node_id: extracted_contr_factors_guesses.append(node_id)
                elif node_type == "SystemicFactor":
                    if node_id: extracted_sys_factors_guesses.append(node_id)
            
            relevant_category_options = ""
            if extracted_accident_type_guess and vectorstore_categories:
                relevant_category_options = faiss_handler.find_most_relevant_generic_chunks(
                    vectorstore_categories, extracted_accident_type_guess, config.TOP_K_RELEVANT_CHUNKS_ISS
                )
            
            standardized_contr_factors_map_str = "{}"
            if extracted_contr_factors_guesses and vectorstore_contr_factors:
                temp_map = {}
                for guess in extracted_contr_factors_guesses:
                    standard_match = faiss_handler.find_most_relevant_generic_chunks(
                        vectorstore_contr_factors, guess, 1
                    )
                    match_name = re.match(r"Factor:\s*(.*?)(?=\s*;\s*Definition:|$)", standard_match)
                    temp_map[guess] = match_name.group(1).strip() if match_name else guess
                standardized_contr_factors_map_str = json.dumps(temp_map)

            standardized_sys_factors_map_str = "{}"
            if extracted_sys_factors_guesses and vectorstore_sys_factors:
                temp_map = {}
                for guess in extracted_sys_factors_guesses:
                    standard_match = faiss_handler.find_most_relevant_generic_chunks(
                        vectorstore_sys_factors, guess, 1
                    )
                    match_name = re.match(r"Factor:\s*(.*?)(?=\s*;\s*Definition:|$)", standard_match)
                    temp_map[guess] = match_name.group(1).strip() if match_name else guess
                standardized_sys_factors_map_str = json.dumps(temp_map)

        with st.spinner("Performing LLM refinement..."):
            refinement_prompt_template = prompts.create_refinement_prompt(
                relevant_events_text=relevant_category_options,
                true_contr_fact_str=standardized_contr_factors_map_str,
                true_sys_fact_str=standardized_sys_factors_map_str
            )
            refinement_chain = chains.create_llm_chain(llm, refinement_prompt_template, conversation_memory)
            refinement_result_raw = chains.run_chain(refinement_chain, {"extraction_result_str": cleaned_extraction_json_str})

        if not refinement_result_raw:
            st.error(f"LLM refinement failed for {uploaded_file.name}.")
            os.remove(tmp_pdf_path)
            continue
        
        cleaned_refined_json_str = re.sub(r'^```json\n?|```$', '', refinement_result_raw, flags=re.MULTILINE).strip()
        try:
            final_llm_json = json.loads(cleaned_refined_json_str)
            st.success(f"Knowledge extraction and refinement complete for `{uploaded_file.name}`.")
            with st.expander("View Final Extracted Knowledge (JSON)"):
                st.json(final_llm_json)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse refined LLM JSON output for {uploaded_file.name}: {e}")
            st.text_area("Problematic Refined JSON Output:", value=cleaned_refined_json_str, height=200)
            os.remove(tmp_pdf_path)
            continue

        validated_graph_model = validation_schemas.validate_llm_output(final_llm_json)
        if not validated_graph_model:
            st.warning(f"Validation of the final JSON failed for {uploaded_file.name}.")
        
        if validated_graph_model:
            with st.spinner(f"Storing results for `{uploaded_file.name}`..."):
                csv_logger.append_pdf_json_result(
                    config.PDF_PROCESSING_RESULTS_CSV,
                    uploaded_file.name,
                    selected_llm_model,
                    final_llm_json
                )
                st.info(f"Results for `{uploaded_file.name}` appended to CSV.")

                if st.session_state.enable_neo4j and neo4j_driver:
                    try:
                        neo4j_handler.process_and_store_graph(
                            neo4j_driver,
                            final_llm_json,
                            pdf_text, 
                            config.NEO4J_DATABASE
                        )
                        st.info(f"Graph data for `{uploaded_file.name}` stored in Neo4j.")
                    except Exception as e:
                        st.error(f"Failed to store graph data in Neo4j for {uploaded_file.name}: {e}")
        
        os.remove(tmp_pdf_path)
        st.markdown("---")

    if st.session_state.enable_neo4j and neo4j_driver:
        neo4j_handler.close_neo4j_driver(neo4j_driver)
    
    st.balloons()
    st.success("All selected PDF(s) processed!")