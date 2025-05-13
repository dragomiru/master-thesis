import streamlit as st
import pandas as pd
import numpy as np # For np.where
import json
import re # For extracting ERAIL ID
import os

# --- Add project root to sys.path for sibling module imports ---
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
st.markdown("Compare LLM-extracted data from processed PDFs with the ERAIL database.")
st.markdown("---")

# --- Helper function for field comparison (kept for readability) ---
def perform_field_comparison(
    df: pd.DataFrame, 
    llm_col: str, 
    erail_col: str, 
    comparison_col_name: str
) -> pd.DataFrame:
    """
    Compares two columns and adds a result column.
    Aligns with notebook cells execution_count: 206, 207, 208.
    """
    # Ensure columns exist to avoid KeyErrors
    if llm_col not in df.columns:
        st.warning(f"LLM column '{llm_col}' not found in data. Skipping this comparison.")
        df[comparison_col_name] = "LLM_Column_Missing"
        return df
    if erail_col not in df.columns:
        st.warning(f"ERAIL column '{erail_col}' not found in data. Skipping this comparison.")
        df[comparison_col_name] = "ERAIL_Column_Missing"
        return df

    # Convert to string for comparison, handling NaNs appropriately
    # Fill NaN with a unique placeholder to distinguish from actual string "nan"
    # then convert to string. This makes comparison more robust.
    llm_series = df[llm_col].fillna("###NAN_LLM###").astype(str)
    erail_series = df[erail_col].fillna("###NAN_ERAIL###").astype(str)

    conditions = [
        (llm_series == "###NAN_LLM###") & (erail_series == "###NAN_ERAIL###"),
        (llm_series == "###NAN_LLM###"),
        (erail_series == "###NAN_ERAIL###"),
        (llm_series == erail_series)
    ]
    choices = ["Match (Both Missing)", "LLM_Data_Missing", "ERAIL_Data_Missing", "Match"]
    
    df[comparison_col_name] = np.select(conditions, choices, default="Mismatch")
    print(f"[INFO_COMP_PAGE] Comparison for '{comparison_col_name}' completed.")
    return df

# --- Main Page Logic ---

if st.button("Load Data and Run Comparison", key="run_erail_comparison_button_main"):
    st.header("Analysis Results")

    # 1. Load LLM Processing Results (from pdf_processing_results.csv)
    # Corresponds to parts of notebook cell execution_count: 204
    llm_results_df = None
    results_csv_path = config.PDF_PROCESSING_RESULTS_CSV
    if not os.path.exists(results_csv_path):
        st.error(f"LLM processing results CSV not found at: {results_csv_path}")
        st.info("Please process some PDFs on the main application page first.")
        st.stop()
    
    try:
        with st.spinner("Loading LLM processing results..."):
            llm_results_df = pd.read_csv(results_csv_path)
        if llm_results_df.empty:
            st.warning("LLM processing results CSV is empty. No data to compare.")
            st.stop()
        st.success(f"Identified {len(llm_results_df)} LLM processing results in the Excel table.")
    except Exception as e:
        st.error(f"Error loading LLM results CSV ('{results_csv_path}'): {e}")
        st.stop()

    # 2. Prepare LLM data for comparison (Extract entities from JSON)
    comparison_list_for_df = []
    with st.spinner("Preparing LLM data for comparison..."):
        for index, row in llm_results_df.iterrows():
            try:
                llm_json_data = json.loads(row["json_output"])
                
                # Direct extraction logic (was _extract_entities_from_json_nodes)
                extracted_llm_entities = {
                    "UniqueAccident": None, "AccidentType": None, "TrackSection": None,
                    "Date": None, "Time": None, "Country": None, "RegulatoryBody": None,
                    "ContributingFactor": [], "SystemicFactor": []
                }
                for node in llm_json_data.get("nodes", []):
                    node_type = node.get("type")
                    node_id = node.get("id")
                    if node_type in extracted_llm_entities:
                        if isinstance(extracted_llm_entities[node_type], list):
                            extracted_llm_entities[node_type].append(node_id)
                        elif extracted_llm_entities[node_type] is None:
                            extracted_llm_entities[node_type] = node_id
                
                if extracted_llm_entities["ContributingFactor"]:
                    extracted_llm_entities["ContributingFactor"] = ", ".join(sorted(list(set(extracted_llm_entities["ContributingFactor"]))))
                else: extracted_llm_entities["ContributingFactor"] = None
                if extracted_llm_entities["SystemicFactor"]:
                    extracted_llm_entities["SystemicFactor"] = ", ".join(sorted(list(set(extracted_llm_entities["SystemicFactor"]))))
                else: extracted_llm_entities["SystemicFactor"] = None
                # End of direct extraction logic

                erail_id_match = re.search(r'(IE-\d+|PL-\d+)', row["pdf_name"])
                erail_occurrence_id = erail_id_match.group(1) if erail_id_match else None
                
                entry = {"pdf_name": row["pdf_name"], "model_type": row.get("model_type"), "ERAIL Occurrence": erail_occurrence_id}
                for entity_type, entity_value in extracted_llm_entities.items():
                    entry[f"LLM_{entity_type}"] = entity_value
                comparison_list_for_df.append(entry)

            except json.JSONDecodeError:
                print(f"[WARN_COMP_PAGE] Failed to parse JSON for PDF: {row['pdf_name']} at index {index}")
                continue
            except Exception as e:
                print(f"[WARN_COMP_PAGE] Error processing row for PDF: {row['pdf_name']} at index {index}: {e}")
                continue
        
        if not comparison_list_for_df:
            st.error("No data was extracted from LLM results for comparison.")
            st.stop()
        prepared_llm_df = pd.DataFrame(comparison_list_for_df)
    st.success(f"LLM data prepared for comparison. Shape: {prepared_llm_df.shape}")

    # 3. Load and Preprocess ERAIL DB
    erail_df = None
    with st.spinner("Loading and preprocessing ERAIL database..."):
        try:
            erail_df = pd.read_excel(config.ERAIL_DB_EXCEL)
            if "Date of occurrence" in erail_df.columns:
                erail_df["Date of occurrence"] = pd.to_datetime(erail_df["Date of occurrence"], errors='coerce').dt.strftime("%d/%m/%Y")
            if "Time of occurrence" in erail_df.columns:
                def format_erail_time(time_val):
                    if pd.isna(time_val): return None
                    if isinstance(time_val, str):
                        try: return pd.to_datetime(time_val, errors='coerce').strftime('%H:%M')
                        except: return None # Handle various string formats or unparsable
                    if hasattr(time_val, 'strftime'): # If datetime.time or datetime.datetime
                        return time_val.strftime('%H:%M')
                    return str(time_val) # Fallback
                erail_df["Time of occurrence"] = erail_df["Time of occurrence"].apply(format_erail_time)

        except FileNotFoundError:
            st.error(f"ERAIL DB file not found at: {config.ERAIL_DB_EXCEL}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading or preprocessing ERAIL DB: {e}")
            st.stop()
    
    if erail_df is None or erail_df.empty:
        st.error("Failed to load or preprocess ERAIL database, or it's empty.")
        st.stop()
    st.success(f"ERAIL DB loaded and preprocessed. Shape: {erail_df.shape}")

    # 4. Merge Data
    merged_df = None
    with st.spinner("Merging LLM results with ERAIL data..."):
        if "ERAIL Occurrence" not in prepared_llm_df.columns or "ERAIL Occurrence" not in erail_df.columns:
            st.error("'ERAIL Occurrence' column missing in one of the dataframes. Cannot merge.")
            st.stop()
        try:
            merged_df = pd.merge(prepared_llm_df, erail_df, on="ERAIL Occurrence", how="inner")
        except Exception as e:
            st.error(f"Error during merging of DataFrames: {e}")
            st.stop()

    if merged_df is None or merged_df.empty:
        st.warning("No common records found after merging LLM results with ERAIL DB. Ensure 'ERAIL Occurrence' IDs match and exist in both datasets.")
        st.stop()
    st.success(f"Successfully merged. Found {len(merged_df)} matching records for comparison.")
    
    # 5. Perform Field Comparisons
    st.subheader("Field-by-Field Comparison Results")
    
    comparison_fields_map = [
        {"llm_col": "LLM_Date", "erail_col": "Date of occurrence", "match_col": "Date_Match", "display_name": "Date"},
        {"llm_col": "LLM_Time", "erail_col": "Time of occurrence", "match_col": "Time_Match", "display_name": "Time"},
        {"llm_col": "LLM_Country", "erail_col": "Country", "match_col": "Country_Match", "display_name": "Country"},
        {"llm_col": "LLM_AccidentType", "erail_col": "Occurrence type", "match_col": "AccidentType_Match", "display_name": "Accident Type"},
        {"llm_col": "LLM_RegulatoryBody", "erail_col": "Reporting Body", "match_col": "RegulatoryBody_Match", "display_name": "Regulatory Body"},
    ]

    display_columns_for_table = ["ERAIL Occurrence", "model_type"] 

    for field_spec in comparison_fields_map:
        merged_df = perform_field_comparison(
            merged_df, 
            llm_col=field_spec["llm_col"], 
            erail_col=field_spec["erail_col"], 
            comparison_col_name=field_spec["match_col"]
        )

        if field_spec["llm_col"] in merged_df.columns: display_columns_for_table.append(field_spec["llm_col"])
        if field_spec["erail_col"] in merged_df.columns: display_columns_for_table.append(field_spec["erail_col"])
        if field_spec["match_col"] in merged_df.columns: display_columns_for_table.append(field_spec["match_col"])
            
    final_display_columns = []
    for col in display_columns_for_table:
        if col not in final_display_columns and col in merged_df.columns:
            final_display_columns.append(col)
            
    st.dataframe(merged_df[final_display_columns])

    st.subheader("Comparison Summary Statistics")
    for field_spec in comparison_fields_map:
        if field_spec["match_col"] in merged_df.columns:
            match_counts = merged_df[field_spec["match_col"]].value_counts()
            st.text(f"Summary for {field_spec['display_name']} ({field_spec['match_col']}):")
            st.dataframe(match_counts)
else:
    st.info("Click the button above to load data and run the comparison with the ERAIL database.")

