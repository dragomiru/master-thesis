import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
from typing import Dict, Optional, Any

# --- Add project root to sys.path ---
import sys
import pathlib
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config

# --- Page Configuration ---
st.set_page_config(
    page_title="ERAIL DB Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ERAIL Database Comparison")
st.markdown("Compare LLM's initial extraction and refined output against the ERAIL database.")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Comparison Controls")
    uploaded_llm_results_file = st.file_uploader(
        "Upload LLM Results CSV", type="csv", accept_multiple_files=False
    )
    process_button = st.button("Run Comparison on Uploaded CSV", type="primary", disabled=not uploaded_llm_results_file)

# --- Helper function for field comparison (remains the same) ---
def perform_field_comparison(
    df: pd.DataFrame, 
    llm_col: str, 
    erail_col: str, 
    comparison_col_name: str
) -> pd.DataFrame:
    if llm_col not in df.columns:
        # st.warning(f"LLM column '{llm_col}' not found. Skipping '{comparison_col_name}'.")
        df[comparison_col_name] = "LLM_Column_Missing"
        return df
    if erail_col not in df.columns:
        # st.warning(f"ERAIL column '{erail_col}' not found. Skipping '{comparison_col_name}'.")
        df[comparison_col_name] = "ERAIL_Column_Missing"
        return df

    llm_series = df[llm_col].fillna("###NAN_PLACEHOLDER###").astype(str).str.strip()
    erail_series = df[erail_col].fillna("###NAN_PLACEHOLDER###").astype(str).str.strip()

    conditions = [
        (llm_series == "###NAN_PLACEHOLDER###") & (erail_series == "###NAN_PLACEHOLDER###"),
        (llm_series == "###NAN_PLACEHOLDER###"),
        (erail_series == "###NAN_PLACEHOLDER###"),
        (llm_series == erail_series)
    ]
    choices = ["Match (Both Missing)", "LLM_Data_Missing", "ERAIL_Data_Missing", "Match"]
    df[comparison_col_name] = np.select(conditions, choices, default="Mismatch")
    return df

# --- Merge Data for Comparison Function ---
def merge_data_for_comparison(llm_df: pd.DataFrame, erail_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merges LLM data with ERAIL data on 'ERAIL Occurrence'."""
    if llm_df.empty or erail_df.empty:
        # st.warning("One or both DataFrames for merging are empty.") # Handled by caller
        return None
    if not all(key in df.columns for df in [llm_df, erail_df] for key in ["ERAIL Occurrence"]):
        st.error("'ERAIL Occurrence' key missing in one of the DataFrames for merge.")
        return None
    try:
        merged = pd.merge(llm_df, erail_df, on="ERAIL Occurrence", how="inner")
        if merged.empty:
            # st.warning("Merge resulted in an empty DataFrame (no common IDs).") # Handled by caller
            pass
        return merged
    except Exception as e:
        st.error(f"Error during DataFrame merge: {e}")
        return None

# --- MODIFIED Helper to extract entities from a specific JSON column ---
def _extract_entities_from_json_nodes(json_data: Dict) -> Dict[str, Any]:
    """Helper to extract specific entities from LLM JSON nodes."""
    target_node_types = { # Define the entities you want to pull out for columns
        "UniqueAccident": None, "AccidentType": None, "TrackSection": None,
        "Date": None, "Time": None, "Country": None, "RegulatoryBody": None,
        "ContributingFactor": [], "SystemicFactor": []
    }
    if not isinstance(json_data, dict) or "nodes" not in json_data:
        # Return empty structure if json_data is not as expected
        return {k: ([] if isinstance(v, list) else None) for k, v in target_node_types.items()}

    for node in json_data.get("nodes", []):
        node_type, node_id_val = node.get("type"), node.get("id") # Renamed node_id to node_id_val
        if node_type in target_node_types:
            if isinstance(target_node_types[node_type], list):
                if node_id_val is not None: target_node_types[node_type].append(str(node_id_val))
            elif target_node_types[node_type] is None and node_id_val is not None:
                 target_node_types[node_type] = str(node_id_val)
    
    for factor_type in ["ContributingFactor", "SystemicFactor"]:
        if target_node_types[factor_type]:
            target_node_types[factor_type] = ", ".join(sorted(list(set(target_node_types[factor_type]))))
        else: # If list is empty after processing
            target_node_types[factor_type] = None 
    return target_node_types

# --- MODIFIED: Function to prepare LLM data, now takes json_column_name ---
def prepare_llm_data_for_comparison(
    results_df: pd.DataFrame, 
    json_column_name: str # "extraction_output" or "refined_output"
) -> Optional[pd.DataFrame]:
    """Prepares LLM results by extracting entities from the specified JSON column."""
    if json_column_name not in results_df.columns:
        st.error(f"JSON column '{json_column_name}' not found in uploaded CSV.")
        return None
    if "pdf_name" not in results_df.columns:
        st.error("'pdf_name' column missing in uploaded CSV.")
        return None
    
    comparison_list = []
    for index, row in results_df.iterrows():
        try:
            json_string = row[json_column_name]
            if pd.isna(json_string): # Handle potential empty cells for this column
                # st.warning(f"Empty JSON data in '{json_column_name}' for PDF: {row['pdf_name']} at index {index}. Skipping.")
                continue
            llm_json_data = json.loads(json_string)
            extracted_entities = _extract_entities_from_json_nodes(llm_json_data)
            
            erail_id_match = re.search(r'(IE-\d+|PL-\d+)', str(row["pdf_name"])) # Ensure pdf_name is string
            erail_occurrence_id = erail_id_match.group(1) if erail_id_match else None
            
            entry = {
                "pdf_name": row["pdf_name"], 
                "model_type": row.get("model_type"), 
                "ERAIL Occurrence": erail_occurrence_id,
                "iteration_number": row.get("iteration_number") # Keep iteration number if present
            }
            for entity_type, entity_value in extracted_entities.items():
                entry[f"LLM_{entity_type}"] = entity_value # These will be the columns compared
            comparison_list.append(entry)

        except json.JSONDecodeError:
            print(f"[WARN_COMP_PAGE] Failed to parse JSON in '{json_column_name}' for PDF: {row.get('pdf_name', 'Unknown PDF')} at index {index}")
            continue 
        except Exception as e:
            print(f"[WARN_COMP_PAGE] Error processing row for PDF: {row.get('pdf_name', 'Unknown PDF')} at index {index}, column '{json_column_name}': {e}")
            continue
        
    if not comparison_list: 
        st.warning(f"No data extracted from '{json_column_name}' for comparison.")
        return pd.DataFrame() # Return empty DataFrame if nothing was processed
    return pd.DataFrame(comparison_list)

# --- ERAIL DB Loading and Preprocessing (remains mostly the same) ---
@st.cache_data # Cache ERAIL DB loading
def load_and_preprocess_erail_db_cached(file_path: str) -> Optional[pd.DataFrame]:
    try:
        erail_df = pd.read_excel(file_path)
        if "Date of occurrence" in erail_df.columns:
            erail_df["Date of occurrence"] = pd.to_datetime(erail_df["Date of occurrence"], errors='coerce').dt.strftime("%d/%m/%Y")
        if "Time of occurrence" in erail_df.columns:
            def format_erail_time(time_val):
                if pd.isna(time_val): return None
                if isinstance(time_val, str):
                    try: return pd.to_datetime(time_val, errors='coerce').strftime('%H:%M')
                    except: return None
                if hasattr(time_val, 'strftime'): return time_val.strftime('%H:%M')
                return str(time_val)
            erail_df["Time of occurrence"] = erail_df["Time of occurrence"].apply(format_erail_time)
        return erail_df
    except FileNotFoundError:
        st.error(f"ERAIL DB file not found at: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading or preprocessing ERAIL DB: {e}")
        return None

# --- Main Page Logic ---
if process_button and uploaded_llm_results_file:
    st.header("Analysis Results")

    llm_results_df_original = None
    try:
        with st.spinner("Loading LLM processing results from uploaded CSV..."):
            llm_results_df_original = pd.read_csv(uploaded_llm_results_file)
        if llm_results_df_original.empty:
            st.warning("Uploaded LLM results CSV is empty. No data to compare.")
            st.stop()
        st.success(f"Loaded {len(llm_results_df_original)} rows from the uploaded LLM results CSV.")
    except Exception as e:
        st.error(f"Error loading uploaded LLM results CSV: {e}")
        st.exception(e)
        st.stop()

    erail_df = load_and_preprocess_erail_db_cached(config.ERAIL_DB_EXCEL)
    if erail_df is None or erail_df.empty:
        st.error("Failed to load or preprocess ERAIL database. Cannot proceed with comparison.")
        st.stop()
    
    comparison_fields_map = [
        {"llm_col": "LLM_Date", "erail_col": "Date of occurrence", "match_col": "Date_Match", "display_name": "Date"},
        {"llm_col": "LLM_Time", "erail_col": "Time of occurrence", "match_col": "Time_Match", "display_name": "Time"},
        {"llm_col": "LLM_Country", "erail_col": "Country", "match_col": "Country_Match", "display_name": "Country"},
        {"llm_col": "LLM_AccidentType", "erail_col": "Occurrence type", "match_col": "AccidentType_Match", "display_name": "Accident Type"},
        {"llm_col": "LLM_RegulatoryBody", "erail_col": "Reporting Body", "match_col": "RegulatoryBody_Match", "display_name": "Regulatory Body"},
        {"llm_col": "LLM_ContributingFactor", "erail_col": "Direct cause description (including causal and contributing factors, excluding those of systemic nature)", "match_col": "ContributingFactors_Match", "display_name": "Contributing Factors"},
        {"llm_col": "LLM_SystemicFactor", "erail_col": "Underlying and root causes description (i.e. systemic factors, if any)", "match_col": "SystemicFactors_Match", "display_name": "Systemic Factors"}
    ]
    base_display_columns = ["ERAIL Occurrence", "model_type", "iteration_number"]


    # --- Process and Display for "extraction_output" ---
    st.subheader("Comparison: Initial Extraction Output vs. ERAIL DB")
    if "extraction_output" in llm_results_df_original.columns:
        with st.spinner("Preparing 'extraction_output' data..."):
            prepared_extraction_df = prepare_llm_data_for_comparison(llm_results_df_original.copy(), "extraction_output")
        
        if prepared_extraction_df is not None and not prepared_extraction_df.empty:
            with st.spinner("Merging 'extraction_output' data with ERAIL DB..."):
                merged_extraction_df = merge_data_for_comparison(prepared_extraction_df, erail_df)
            
            if merged_extraction_df is not None and not merged_extraction_df.empty:
                st.info(f"Found {len(merged_extraction_df)} matching records for 'extraction_output' comparison.")
                current_display_cols = base_display_columns[:]
                for field_spec in comparison_fields_map:
                    merged_extraction_df = perform_field_comparison(
                        merged_extraction_df, field_spec["llm_col"], field_spec["erail_col"], field_spec["match_col"]
                    )
                    for col_key in ["llm_col", "erail_col", "match_col"]:
                        if field_spec[col_key] in merged_extraction_df.columns: current_display_cols.append(field_spec[col_key])
                
                final_display_cols_extraction = [col for i, col in enumerate(current_display_cols) if col not in current_display_cols[:i]]
                st.dataframe(merged_extraction_df[[col for col in final_display_cols_extraction if col in merged_extraction_df.columns]])

                st.text("Summary Statistics for Initial Extraction:")
                for field_spec in comparison_fields_map:
                    if field_spec["match_col"] in merged_extraction_df.columns:
                        st.text(f"  {field_spec['display_name']} ({field_spec['match_col']}):")
                        st.dataframe(merged_extraction_df[field_spec["match_col"]].value_counts())
            else:
                st.warning("No common records found after merging for 'extraction_output'.")
        else:
            st.warning("'extraction_output' data could not be prepared.")
    else:
        st.warning("Column 'extraction_output' not found in the uploaded CSV.")

    st.markdown("---") # Separator

    # --- Process and Display for "refined_output" ---
    st.subheader("Comparison: Refined Output vs. ERAIL DB")
    if "refined_output" in llm_results_df_original.columns:
        with st.spinner("Preparing 'refined_output' data..."):
            prepared_refined_df = prepare_llm_data_for_comparison(llm_results_df_original.copy(), "refined_output")

        if prepared_refined_df is not None and not prepared_refined_df.empty:
            with st.spinner("Merging 'refined_output' data with ERAIL DB..."):
                merged_refined_df = merge_data_for_comparison(prepared_refined_df, erail_df)

            if merged_refined_df is not None and not merged_refined_df.empty:
                st.info(f"Found {len(merged_refined_df)} matching records for 'refined_output' comparison.")
                current_display_cols = base_display_columns[:]
                for field_spec in comparison_fields_map:
                    merged_refined_df = perform_field_comparison(
                        merged_refined_df, field_spec["llm_col"], field_spec["erail_col"], field_spec["match_col"]
                    )
                    for col_key in ["llm_col", "erail_col", "match_col"]:
                        if field_spec[col_key] in merged_refined_df.columns: current_display_cols.append(field_spec[col_key])
                
                final_display_cols_refined = [col for i, col in enumerate(current_display_cols) if col not in current_display_cols[:i]]
                st.dataframe(merged_refined_df[[col for col in final_display_cols_refined if col in merged_refined_df.columns]])
                
                st.text("Summary Statistics for Refined Output:")
                for field_spec in comparison_fields_map:
                    if field_spec["match_col"] in merged_refined_df.columns:
                        st.text(f"  {field_spec['display_name']} ({field_spec['match_col']}):")
                        st.dataframe(merged_refined_df[field_spec["match_col"]].value_counts())
            else:
                st.warning("No common records found after merging for 'refined_output'.")
        else:
            st.warning("'refined_output' data could not be prepared.")
    else:
        st.warning("Column 'refined_output' not found in the uploaded CSV.")
else:
    st.info("Upload your LLM results CSV and click the 'Run Comparison' button to start.")