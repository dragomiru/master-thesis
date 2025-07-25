# --- General modules ---
import pandas as pd
from typing import List, Dict, Optional

# --- Functions for loading the data ---
def load_accident_categories(cat_a_path: str, cat_b_path: str, cat_c_path: str) -> Optional[pd.DataFrame]:
    """
    Loads and merges accident category event CSV files from the ISS ontology.

    Args:
        cat_a_path (str): Path to category A events CSV.
        cat_b_path (str): Path to category B events CSV.
        cat_c_path (str): Path to category C events CSV.

    Returns:
        Optional[pd.DataFrame]: Merged DataFrame of all category events, or None if loading fails.
    """
    try:
        cat_a_events = pd.read_csv(cat_a_path, encoding='latin-1')
        cat_b_events = pd.read_csv(cat_b_path, encoding='latin-1')
        cat_c_events = pd.read_csv(cat_c_path, encoding='latin-1')
        
        # Merge the DataFrames into one
        all_cat_events = pd.concat([cat_a_events, cat_b_events, cat_c_events], ignore_index=True)
        print(f"[INFO] Loaded {len(all_cat_events)} accident categories.")
        return all_cat_events
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load accident category file: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading accident categories: {e}")
        return None

def load_contributing_factors(contr_fact_path: str) -> Optional[pd.DataFrame]:
    """
    Loads contributing factors from a CSV file.

    Args:
        contr_fact_path (str): Path to the contributing factors CSV.

    Returns:
        Optional[pd.DataFrame]: DataFrame of contributing factors, or None if loading fails.
    """
    try:
        contr_factors_df = pd.read_csv(contr_fact_path, encoding='latin-1')
        print(f"[INFO] Loaded {len(contr_factors_df)} contributing factors.")
        return contr_factors_df
    except FileNotFoundError:
        print(f"[ERROR] Contributing factors file not found at: {contr_fact_path}")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading contributing factors: {e}")
        return None

def load_systemic_factors(sys_fact_path: str) -> Optional[pd.DataFrame]:
    """
    Loads systemic factors from a CSV file.

    Args:
        sys_fact_path (str): Path to the systemic factors CSV.

    Returns:
        Optional[pd.DataFrame]: DataFrame of systemic factors, or None if loading fails.
    """
    try:
        sys_factors_df = pd.read_csv(sys_fact_path, encoding='latin-1')
        print(f"[INFO] Loaded {len(sys_factors_df)} systemic factors.")
        return sys_factors_df
    except FileNotFoundError:
        print(f"[ERROR] Systemic factors file not found at: {sys_fact_path}")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading systemic factors: {e}")
        return None

def convert_df_to_dicts(df: Optional[pd.DataFrame]) -> List[Dict]:
    """
    Converts a Pandas DataFrame to a list of dictionaries.
    Returns an empty list if the DataFrame is None or empty.
    """
    if df is None or df.empty:
        return []
    return df.to_dict("records")
