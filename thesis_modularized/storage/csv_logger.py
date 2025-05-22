import pandas as pd
import os
import json
from typing import Dict, Optional

def append_pdf_json_result(csv_file_path: str, pdf_name: str, model_type: str, extraction_json: Dict,
                           refined_json: Dict) -> Optional[pd.DataFrame]:
    """
    Appends the JSON output of an LLM response to a CSV file.
    - If the PDF has been processed before by the same model, it appends a new row
      with an incremented iteration number if the JSON output is different.
    - Prevents duplicate JSON entries for the same PDF, model, and iteration.
    - Ensures data is stored in rows, making querying and analysis easier.

    Args:
        csv_file_path (str): Path to the CSV file for storing results.
        pdf_name (str): Name of the processed PDF file.
        model_type (str): Identifier for the LLM used (e.g., "gpt-4o-mini").
        extraction_json (Dict): JSON response (as a Python dictionary) from the knowledge extraction.
        refined_json (Dict): JSON response (as a Python dictionary) from the knowledge extraction.

    Returns:
        Optional[pd.DataFrame]: Updated DataFrame with the new result, or None if an error occurs.
    """
    try:
        # Convert JSON response to a formatted string for consistent comparison and storage
        extraction_output_str = json.dumps(extraction_json, indent=2, sort_keys=True)
        refined_output_str = json.dumps(refined_json, indent=2, sort_keys=True)

        # Ensure the directory for the CSV exists
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Load existing results if the CSV exists
        if os.path.exists(csv_file_path):
            try:
                df = pd.read_csv(csv_file_path)
                # Ensure iteration_number is integer, handle potential float from read_csv
                if "iteration_number" in df.columns:
                    df["iteration_number"] = df["iteration_number"].fillna(0).astype(int)
                else: # If first time and column doesn't exist
                    df["iteration_number"] = 0
            except pd.errors.EmptyDataError: # If CSV is empty
                df = pd.DataFrame(columns=["pdf_name", "model_type", "iteration_number", "extraction_output", "refined_output"])
        else:
            df = pd.DataFrame(columns=["pdf_name", "model_type", "iteration_number", "extraction_output", "refined_output"])

        # Filter for the current PDF and model's past records
        pdf_model_history = df[(df["pdf_name"] == pdf_name) & (df["model_type"] == model_type)]

        # Check for duplicates: If this JSON output already exists for the same PDF & model, skip re-adding
        if not pdf_model_history.empty:
            latest_iteration_data = pdf_model_history[
                pdf_model_history["iteration_number"] == pdf_model_history["iteration_number"].max()
            ]
            if not latest_iteration_data.empty and latest_iteration_data.iloc[0]["refined_output"] == refined_output_str:
                print(f"[INFO] No changes detected in JSON for {pdf_name} by {model_type} (latest iteration), skipping new entry.")
                return df

        # Determine new iteration number
        if not pdf_model_history.empty:
            iteration_number = pdf_model_history["iteration_number"].max() + 1
        else:
            iteration_number = 1
        
        # Create new entry as a DataFrame
        new_entry_data = {
            "pdf_name": [pdf_name],
            "model_type": [model_type],
            "iteration_number": [iteration_number],
            "extraction_output": [extraction_output_str],
            "refined_output": [refined_output_str]
        }
        new_entry_df = pd.DataFrame(new_entry_data)

        # Concatenate new entry
        df = pd.concat([df, new_entry_df], ignore_index=True)

        # Save back to CSV
        df.to_csv(csv_file_path, index=False)

        print(f"[INFO] Successfully added/updated '{pdf_name}' by '{model_type}' - Iteration {iteration_number} to results at '{csv_file_path}'.")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to append PDF JSON result to CSV: {e}")
        return None
