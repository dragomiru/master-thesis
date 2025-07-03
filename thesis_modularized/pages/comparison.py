# pages/Evaluation.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
from typing import Optional, Dict, Any

# --- Add project root to sys.path for sibling module imports ---
import sys
import pathlib
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config # Your main configuration file

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 LLM Performance Evaluation")
st.markdown("Analyze the accuracy, consistency, and completeness of LLM outputs by uploading your results CSV file.")

# --- CORE HELPER FUNCTIONS (Adapted from your notebook) ---

@st.cache_data # Cache ERAIL DB loading to speed up reruns
def load_and_preprocess_erail_db(file_path: str) -> Optional[pd.DataFrame]:
    """Loads the ERAIL database and preprocesses date/time columns."""
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

def prepare_llm_data(results_df: pd.DataFrame, json_column_name: str) -> Optional[pd.DataFrame]:
    """Prepares LLM results by extracting entities from the specified JSON column."""
    if json_column_name not in results_df.columns or "pdf_name" not in results_df.columns:
        st.error(f"Required columns ('{json_column_name}', 'pdf_name') not found in uploaded CSV.")
        return None
    
    comparison_list = []
    for index, row in results_df.iterrows():
        try:
            json_string = row[json_column_name]
            if pd.isna(json_string): continue
            llm_json_data = json.loads(json_string)
            
            extracted_entities = {
                "UniqueAccident": None, "AccidentType": None, "TrackSection": None,
                "Date": None, "Time": None, "Country": None, "RegulatoryBody": None,
                "ContributingFactor": [], "SystemicFactor": []
            }
            for node in llm_json_data.get("nodes", []):
                node_type, node_id = node.get("type"), node.get("id")
                if node_type in extracted_entities:
                    if isinstance(extracted_entities[node_type], list):
                        if node_id is not None: extracted_entities[node_type].append(str(node_id))
                    elif extracted_entities[node_type] is None and node_id is not None:
                        extracted_entities[node_type] = str(node_id)
            
            for factor_type in ["ContributingFactor", "SystemicFactor"]:
                if extracted_entities[factor_type]:
                    extracted_entities[factor_type] = ", ".join(sorted(list(set(extracted_entities[factor_type]))))
                else:
                    extracted_entities[factor_type] = None

            erail_id_match = re.search(r'([A-Z]{2}-\d+)', str(row["pdf_name"]))
            
            entry = {
                "pdf_name": row["pdf_name"], "model_type": row.get("model_type"),
                "iteration_number": row.get("iteration_number"),
                "ERAIL Occurrence": erail_id_match.group(1) if erail_id_match else None
            }
            
            for entity_type, value in extracted_entities.items():
                entry[f"LLM_{entity_type}"] = value
            comparison_list.append(entry)
        except (json.JSONDecodeError, TypeError):
            continue
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
        
    if not comparison_list: return pd.DataFrame()
    return pd.DataFrame(comparison_list)

def evaluate_metrics(df: pd.DataFrame, llm_col: str, truth_col: str, is_time: bool = False) -> Dict[str, Optional[float]]:
    """Calculates all evaluation metrics for a given variable and returns a dictionary."""
    results = {}
    if df.empty or llm_col not in df.columns or truth_col not in df.columns:
        return results

    temp_df = df.copy()
    llm_series = temp_df[llm_col].fillna("###NAN###").astype(str).str.strip().str.lower()
    truth_series = temp_df[truth_col].fillna("###NAN###").astype(str).str.strip().str.lower()
    results["Abs. Accuracy"] = (llm_series == truth_series).mean() * 100

    if 'ERAIL Occurrence' in temp_df.columns and 'model_type' in temp_df.columns:
        group = temp_df.groupby(['ERAIL Occurrence', 'model_type'])
        consistency_check = group.apply(lambda g: g[llm_col].nunique() == 1, include_groups=False)
        if not consistency_check.empty:
            results["Consistency"] = consistency_check.mean() * 100

    results["Completeness"] = temp_df[llm_col].apply(lambda x: pd.notna(x) and str(x).strip() != '').mean() * 100

    if is_time:
        temp_df['llm_time'] = pd.to_datetime(temp_df[llm_col], format='%H:%M', errors='coerce')
        temp_df['truth_time'] = pd.to_datetime(temp_df[truth_col], format='%H:%M', errors='coerce')
        valid_times = temp_df['llm_time'].notna() & temp_df['truth_time'].notna()
        if valid_times.any():
            time_diff = abs((temp_df.loc[valid_times, 'llm_time'] - temp_df.loc[valid_times, 'truth_time']).dt.total_seconds())
            results["2min Accuracy"] = (time_diff <= 120).sum() / valid_times.sum() * 100
            results["5min Accuracy"] = (time_diff <= 300).sum() / valid_times.sum() * 100
            
    return results

# --- CORRECTED FUNCTION SIGNATURE ---
def perform_field_comparison(df: pd.DataFrame, llm_col: str, truth_col: str, comparison_col_name: str) -> pd.DataFrame:
    """Compares two columns and adds a result column."""
    if llm_col not in df.columns:
        df[comparison_col_name] = "LLM_Column_Missing"
        return df
    if truth_col not in df.columns: # Changed from erail_col
        df[comparison_col_name] = "ERAIL_Column_Missing"
        return df

    llm_series = df[llm_col].fillna("###NAN_PLACEHOLDER###").astype(str).str.strip()
    erail_series = df[truth_col].fillna("###NAN_PLACEHOLDER###").astype(str).str.strip() # Changed from erail_col

    conditions = [
        (llm_series == "###NAN_PLACEHOLDER###") & (erail_series == "###NAN_PLACEHOLDER###"),
        (llm_series == "###NAN_PLACEHOLDER###"),
        (erail_series == "###NAN_PLACEHOLDER###"),
        (llm_series == erail_series)
    ]
    choices = ["Match (Both Missing)", "LLM_Data_Missing", "ERAIL_Data_Missing", "Match"]
    df[comparison_col_name] = np.select(conditions, choices, default="Mismatch")
    return df

def display_metrics(title: str, metrics: Dict[str, Optional[float]]):
    """Renders only the valid, calculated metrics in an intelligible way."""
    st.subheader(title)
    
    # Define the desired order of metrics
    metric_keys_ordered = ["Abs. Accuracy", "2min Accuracy", "5min Accuracy", "Consistency", "Completeness"]
    
    # Filter out metrics that were not calculated (i.e., are None or NaN)
    valid_metrics = {key: metrics.get(key) for key in metric_keys_ordered if pd.notna(metrics.get(key))}
    
    if not valid_metrics:
        st.write("No metrics calculated for this variable.")
        return
        
    # Create columns dynamically based on how many valid metrics there are
    cols = st.columns(len(valid_metrics))
    
    # Display each valid metric in its own column
    for i, (key, val) in enumerate(valid_metrics.items()):
        with cols[i]:
            st.metric(label=key, value=f"{val:.1f}%")
            st.progress(int(val))

# --- UI AND MAIN LOGIC ---
uploaded_llm_results_file = st.file_uploader("Upload your results CSV file", type="csv")

if uploaded_llm_results_file:
    with st.sidebar:
        st.header("⚙️ Evaluation Controls")
        output_column = st.radio(
            "Select Output to Analyze:", ("extraction_output", "refined_output"), index=1,
            help="Choose which LLM output column from your CSV to evaluate."
        )
        df_results_original = pd.read_csv(uploaded_llm_results_file)
        pdf_list = ["All Reports"] + sorted(df_results_original['pdf_name'].unique().tolist())
        selected_pdf = st.selectbox(
            "Filter by Report (Optional):", pdf_list,
            help="Select a single report to analyze or choose 'All Reports' for an overall evaluation."
        )
        run_evaluation_button = st.button("Run Evaluation", type="primary")

    if run_evaluation_button:
        st.header("Evaluation Results")
        
        df_to_evaluate = df_results_original[df_results_original['pdf_name'] == selected_pdf].copy() if selected_pdf != "All Reports" else df_results_original.copy()
        
        if df_to_evaluate.empty:
            st.warning("No data available for the selected report. Please choose another.")
            st.stop()

        st.info(f"Analyzing **{len(df_to_evaluate)} records** for **'{output_column}'** from **'{selected_pdf}'**.")
        
        with st.spinner("Preparing data and merging with ERAIL DB..."):
            prepared_df = prepare_llm_data(df_to_evaluate, output_column)
            erail_df = load_and_preprocess_erail_db(config.ERAIL_DB_EXCEL)
            
            if prepared_df is None or erail_df is None:
                st.error("Data preparation failed. Cannot proceed.")
                st.stop()
            
            merged_df = pd.merge(prepared_df, erail_df, on="ERAIL Occurrence", how="left")
            if merged_df.empty:
                st.warning("Could not merge any records with ERAIL DB. Check 'ERAIL Occurrence' IDs.")
                st.stop()

        st.success("Data prepared. Calculating and displaying metrics...")
        
        variables_to_evaluate = [
            {"var_name": "Date", "llm_col": "LLM_Date", "truth_col": "Date of occurrence"},
            {"var_name": "Time", "llm_col": "LLM_Time", "truth_col": "Time of occurrence", "is_time": True},
            {"var_name": "Country", "llm_col": "LLM_Country", "truth_col": "Country"},
            {"var_name": "Accident Type", "llm_col": "LLM_AccidentType", "truth_col": "Occurrence type"},
            {"var_name": "Regulatory Body", "llm_col": "LLM_RegulatoryBody", "truth_col": "Reporting Body"},
            {"var_name": "Contributing Factors", "llm_col": "LLM_ContributingFactor", "truth_col": "Direct cause description (including causal and contributing factors, excluding those of systemic nature)"},
            {"var_name": "Systemic Factors", "llm_col": "LLM_SystemicFactor", "truth_col": "Underlying and root causes description (i.e. systemic factors, if any)"},
        ]
        
        tab1, tab2 = st.tabs(["📊 Metrics Dashboard", "📄 Detailed Comparison Data"])

        with tab1:
            st.subheader("Key Performance Indicators")
            all_metrics_data = []
            for var_spec in variables_to_evaluate:
                metrics = evaluate_metrics(
                    df=merged_df, llm_col=var_spec['llm_col'], truth_col=var_spec['truth_col'],
                    is_time=var_spec.get('is_time', False)
                )
                all_metrics_data.append({"Variable": var_spec['var_name'], **metrics})
            
            summary_df = pd.DataFrame(all_metrics_data).set_index("Variable")

            kpi_cols = st.columns(3)
            with kpi_cols[0]:
                abs_acc = summary_df['Abs. Accuracy'].mean()
                st.metric("Avg. Abs. Accuracy", f"{abs_acc:.1f}%", help="Average of 'Abs. Accuracy' across all variables.")
            with kpi_cols[1]:
                consistency = summary_df['Consistency'].mean()
                st.metric("Avg. Consistency", f"{consistency:.1f}%", help="Average of 'Consistency' across all variables.")
            with kpi_cols[2]:
                completeness = summary_df['Completeness'].mean()
                st.metric("Avg. Completeness", f"{completeness:.1f}%", help="Average of 'Completeness' across all variables.")

            st.markdown("---")
            st.subheader("Detailed Metrics by Variable")
            
            for index, row in summary_df.iterrows():
                with st.container(border=True):
                    display_metrics(f"Metrics for: {row.name}", row.to_dict())

        with tab2:
            st.subheader("Detailed Comparison Table")
            st.info("This table shows the side-by-side comparison for each entry.")
            
            comparison_df_for_display = merged_df.copy()
            display_cols_ordered = ["ERAIL Occurrence", "model_type"]
            
            # --- CORRECTED LOOP ---
            for field in variables_to_evaluate:
                match_col_name = f"{field['var_name'].replace(' ', '')}_Match"
                # Call perform_field_comparison with the correct key: 'truth_col'
                comparison_df_for_display = perform_field_comparison(
                    comparison_df_for_display, 
                    field["llm_col"], 
                    field["truth_col"], # Use the correct key
                    match_col_name
                )
                # Append columns for display
                if field["llm_col"] in comparison_df_for_display.columns:
                    display_cols_ordered.append(field["llm_col"])
                if field["truth_col"] in comparison_df_for_display.columns:
                    display_cols_ordered.append(field["truth_col"])
                if match_col_name in comparison_df_for_display.columns:
                    display_cols_ordered.append(match_col_name)
            
            final_display_cols = [col for i, col in enumerate(display_cols_ordered) if col not in display_cols_ordered[:i] and col in comparison_df_for_display.columns]
            st.dataframe(comparison_df_for_display[final_display_cols])

else:
    st.info("Upload a CSV file and select your options in the sidebar to begin.")

