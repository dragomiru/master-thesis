import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LLM Configuration ---
# Default model to use. Can be overridden by user selection in Streamlit
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# --- Neo4j Connection Details ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- File and Directory Paths ---
DATA_DIR = "./data/"

# Input PDF directory
DEFAULT_PDF_INPUT_DIR = "./reports_ie/"

# CSV data sources
CATEGORY_A_EVENTS_CSV = os.path.join(DATA_DIR, "category-a-event-types-source.csv")
CATEGORY_B_EVENTS_CSV = os.path.join(DATA_DIR, "category-b-event-types-source.csv")
CATEGORY_C_EVENTS_CSV = os.path.join(DATA_DIR, "category-c-event-types-source.csv")
CONTRIBUTING_FACTORS_CSV = os.path.join(DATA_DIR, "contributing-factors-source.csv")
SYSTEMIC_FACTORS_CSV = os.path.join(DATA_DIR, "systemic-factors-source.csv")

# Output CSV for results
PDF_PROCESSING_RESULTS_CSV = os.path.join(DATA_DIR, "pdf_processing_results.csv")

# ERAIL Database Excel
ERAIL_DB_EXCEL = os.path.join(DATA_DIR, "erail database.xlsx")

# --- Text Processing Parameters ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# For FAISS similarity search
TOP_K_RELEVANT_CHUNKS_REPORT = 3
TOP_K_RELEVANT_CHUNKS_ISS = 3

# --- Embeddings Model ---
EMBEDDINGS_MODEL_NAME = "all-mpnet-base-v2"

# --- Application Behavior Flags ---
ENABLE_NEO4J_STORAGE = True
ENABLE_ERAIL_COMPARISON = True
AUTO_PROCEED_LLM_CALLS = False

# --- Streamlit UI Configuration ---
APP_TITLE = "Railway Accident Report Analyzer"
APP_FAVICON = "🚂"

# --- Helper function to check for essential API keys ---
def check_api_keys():
    """Checks if required API keys are set, prints warnings if not."""
    if DEFAULT_LLM_MODEL.startswith("gpt") and not OPENAI_API_KEY:
        print("Warning: OpenAI API key is not set. GPT models will not function.")
        print("Please set the OPENAI_API_KEY environment variable.")
        return False
    if DEFAULT_LLM_MODEL.startswith("gemini") and not GOOGLE_API_KEY:
        print("Warning: Google API key is not set. Gemini models will not function.")
        print("Please set the GOOGLE_API_KEY environment variable.")
        return False
    return True