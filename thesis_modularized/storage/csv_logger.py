import pandas as pd
import os
import json
from typing import Dict, Optional

def append_pdf_json_result(csv_file_path: str, pdf_name: str, model_type: str, llm_response_json: Dict) -> Optional[pd.DataFrame]:
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
        llm_response_json (Dict): JSON response (as a Python dictionary) from the knowledge extraction.

    Returns:
        Optional[pd.DataFrame]: Updated DataFrame with the new result, or None if an error occurs.
    """
    try:
        # Convert JSON response to a formatted string for consistent comparison and storage
        json_output_str = json.dumps(llm_response_json, indent=2, sort_keys=True)

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
                df = pd.DataFrame(columns=["pdf_name", "model_type", "iteration_number", "json_output"])
        else:
            df = pd.DataFrame(columns=["pdf_name", "model_type", "iteration_number", "json_output"])

        # Filter for the current PDF and model's past records
        pdf_model_history = df[(df["pdf_name"] == pdf_name) & (df["model_type"] == model_type)]

        # Check for duplicates: If this JSON output already exists for the same PDF & model, skip re-adding
        if not pdf_model_history.empty:
            latest_iteration_data = pdf_model_history[
                pdf_model_history["iteration_number"] == pdf_model_history["iteration_number"].max()
            ]
            if not latest_iteration_data.empty and latest_iteration_data.iloc[0]["json_output"] == json_output_str:
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
            "json_output": [json_output_str]
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

# if __name__ == '__main__':
#     print("--- Testing storage/csv_logger.py ---")
#     # Dummy data for testing
#     test_csv_path = "./test_pdf_processing_results.csv" # Temporary test file
    
#     sample_response_1 = {"nodes": [{"id": "Accident1", "type": "UniqueAccident"}], "rels": []}
#     sample_response_2 = {"nodes": [{"id": "Accident2", "type": "UniqueAccident"}], "rels": [{"source": "Accident2", "target": "Date1", "type": "has_date"}]}

#     # Clean up previous test file if it exists
#     if os.path.exists(test_csv_path):
#         os.remove(test_csv_path)

#     # Test 1: Add first entry
#     df_results = append_pdf_json_result(test_csv_path, "reportA.pdf", "gpt-4o-mini", sample_response_1)
#     if df_results is not None:
#         print(f"CSV content after 1st entry:\n{df_results}\n")
#         assert len(df_results) == 1
#         assert df_results.iloc[0]["iteration_number"] == 1

#     # Test 2: Add a different entry for the same PDF and model (new iteration)
#     df_results = append_pdf_json_result(test_csv_path, "reportA.pdf", "gpt-4o-mini", sample_response_2)
#     if df_results is not None:
#         print(f"CSV content after 2nd entry (new iteration):\n{df_results}\n")
#         assert len(df_results) == 2
#         assert df_results[(df_results["pdf_name"] == "reportA.pdf") & (df_results["model_type"] == "gpt-4o-mini")]["iteration_number"].max() == 2

#     # Test 3: Add the same entry again (should not create a new row if it's identical to the last one)
#     df_results = append_pdf_json_result(test_csv_path, "reportA.pdf", "gpt-4o-mini", sample_response_2)
#     if df_results is not None:
#         print(f"CSV content after 3rd entry (duplicate of last):\n{df_results}\n")
#         assert len(df_results) == 2 # Length should remain the same

#     # Test 4: Add entry for a different PDF
#     df_results = append_pdf_json_result(test_csv_path, "reportB.pdf", "gemini-1.5-flash", sample_response_1)
#     if df_results is not None:
#         print(f"CSV content after 4th entry (new PDF):\n{df_results}\n")
#         assert len(df_results) == 3
#         assert df_results[(df_results["pdf_name"] == "reportB.pdf")]["iteration_number"].max() == 1
    
#     # Clean up test file
#     if os.path.exists(test_csv_path):
#         os.remove(test_csv_path)
#         print(f"\n[INFO] Cleaned up test file: {test_csv_path}")

#     print("------------------------------------")