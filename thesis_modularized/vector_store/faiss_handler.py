# --- General modules ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Optional, Any
from config import EMBEDDINGS_MODEL_NAME

# --- Function for loading the embeddings model ---
def get_embeddings_model(model_name: str) -> Optional[Any]:
    """
    Initializes and returns a HuggingFace embeddings model.

    Args:
        model_name (str): The name of the HuggingFace model to use for embeddings.
                          Example: "all-mpnet-base-v2".

    Returns:
        Optional[Any]: The initialized embeddings model, or None if initialization fails.
    """
    try:
        print(f"[INFO] Initializing embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to initialize embeddings model '{model_name}': {e}")
        return None

# --- Functions for creating FAISS stores from texts ---
def create_faiss_store_from_texts(
    texts: List[str],
    embeddings_model: Any
) -> Optional[FAISS]:
    """
    Creates a FAISS vector store from a list of text strings.

    Args:
        texts (List[str]): A list of text documents/chunks.
        embeddings_model (Any): The pre-initialized embeddings model.

    Returns:
        Optional[FAISS]: The created FAISS vector store, or None if creation fails or input is empty.
    """
    if not texts:
        print("[WARNING] No texts provided to create FAISS store.")
        return None
    if not embeddings_model:
        print("[ERROR] Embeddings model not provided to create_faiss_store_from_texts.")
        return None
    try:
        print(f"[INFO] Creating FAISS vector store from {len(texts)} text chunks.")
        vectorstore = FAISS.from_texts(texts, embeddings_model)
        print("[INFO] FAISS vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to create FAISS vector store from texts: {e}")
        return None

def create_faiss_store_from_document_lists(
    document_lists: List[List[str]],
    embeddings_model: Any
) -> Optional[FAISS]:
    """
    Creates a FAISS vector store from a list of lists of document chunks.
    The inner lists are flattened before creating the store.

    Args:
        document_lists (List[List[str]]): A list where each element is a list of text chunks.
                                          Typically used for event_chunks or factor_chunks.
        embeddings_model (Any): The pre-initialized embeddings model.

    Returns:
        Optional[FAISS]: The created FAISS vector store, or None if creation fails or input is empty.
    """
    if not document_lists:
        print("[WARNING] No document lists provided to create FAISS store.")
        return None
    if not embeddings_model:
        print("[ERROR] Embeddings model not provided to create_faiss_store_from_document_lists.")
        return None

    flat_chunks = [chunk for sublist in document_lists for chunk in sublist if sublist] # Ensure sublist is not empty
    if not flat_chunks:
        print("[WARNING] No effective chunks found after flattening document lists.")
        return None
    
    return create_faiss_store_from_texts(flat_chunks, embeddings_model)

# --- Function to identify relevant report chunks ---
def find_most_relevant_report_chunks(
    vectorstore_reports: FAISS,
    entity_queries: Dict[str, str],
    top_k: int = 3
) -> str:
    """
    Finds the most relevant text chunks for each entity of interest from the report's vector store.

    Args:
        vectorstore_reports (FAISS): The FAISS vector store for the incident report.
        entity_queries (Dict[str, str]): Mapping of entity name to query string.
        top_k (int): Number of chunks to retrieve per entity.

    Returns:
        str: Combined unique relevant chunks as a single string.
    """
    if not vectorstore_reports:
        print("[ERROR] Report vector store not provided to find_most_relevant_report_chunks.")
        return ""
        
    retrieved_chunks_content = set() # Store page_content to ensure uniqueness

    for entity, query in entity_queries.items():
        print(f"[INFO] Searching report vector store for entity: {entity} (Query: '{query[:50]}...')")
        try:
            found_documents = vectorstore_reports.similarity_search(query, k=top_k)
            for doc in found_documents:
                retrieved_chunks_content.add(doc.page_content)
        except Exception as e:
            print(f"[ERROR] Error during similarity search for entity '{entity}': {e}")
            continue # Continue with other entities

    if not retrieved_chunks_content:
        print("[INFO] No relevant chunks found for any entity in the report.")
        return ""

    combined_text = "\n".join(list(retrieved_chunks_content)) # Convert set to list before joining
    print(f"\n[INFO] Found {len(retrieved_chunks_content)} unique relevant chunks from the report.")
    return combined_text

# --- General function used to retrieve generic chunks for accidents cats. + contr. + sys. factors ---
def find_most_relevant_generic_chunks(
    vectorstore: FAISS,
    query_input: str,
    top_k: int = 3
) -> str:
    """
    Retrieves the most relevant chunks from a generic vector store based on the query.
    This was originally `find_most_relevant_iss_chunks` for categories/factors.

    Args:
        vectorstore (FAISS): The FAISS vector store to query (e.g., for categories, factors).
        query_input (str): The query string.
        top_k (int): Number of top chunks to retrieve.

    Returns:
        str: Combined relevant chunks as a single string.
    """
    if not vectorstore:
        print(f"[ERROR] Vector store not provided to find_most_relevant_generic_chunks for query: '{query_input[:50]}...'.")
        return ""
    if not query_input:
        print("[WARNING] Empty query input provided to find_most_relevant_generic_chunks.")
        return ""
        
    print(f"[INFO] Querying generic vector store with: '{query_input[:50]}...' (top_k={top_k})")
    try:
        found_documents = vectorstore.similarity_search(query_input, k=top_k)
        chunk_contents = [doc.page_content for doc in found_documents if doc.page_content]
        
        if not chunk_contents:
            print(f"[INFO] No relevant generic chunks found for query: '{query_input[:50]}...'.")
            return ""

        combined_text = "\n".join(chunk_contents)
        return combined_text
    except Exception as e:
        print(f"[ERROR] Error during similarity search for generic query '{query_input[:50]}...': {e}")
        return ""