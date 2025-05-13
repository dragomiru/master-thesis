from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_report_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into smaller overlapping chunks using LangChain's text splitter.
    """
    if not text:
        print("[WARNING] No text provided for splitting report into chunks.")
        return []
    
    # If text is smaller than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]  

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def _split_single_text_for_events_factors(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Helper function to split a single text string."""
    if not text:
        print("[INFO] Warning: No text provided for splitting single text for events/factors.")
        return []
    if len(text) <= chunk_size:
        return [text]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def split_events_into_chunks(data: List[Dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[List[str]]:
    """
    Converts each dictionary row (representing an event) into a formatted string
    and then splits each string into smaller overlapping chunks.
    """
    if not data:
        print("[WARNING] No data provided for splitting events into chunks.")
        return []

    chunk_list_strings = [
        f"AccidentType: {row.get('Name', 'N/A')}; Definition: {row.get('Definition', 'N/A')}"
        for row in data
    ]

    all_split_chunks: List[List[str]] = []
    for text_string in chunk_list_strings:
        all_split_chunks.append(
            _split_single_text_for_events_factors(text_string, chunk_size, chunk_overlap)
        )
    return all_split_chunks

def split_factors_into_chunks(data: List[Dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[List[str]]:
    """
    Converts each dictionary row (representing a factor) into a formatted string
    and then splits each string into smaller overlapping chunks.
    """
    if not data:
        print("[WARNING] No data provided for splitting factors into chunks.")
        return []

    chunk_list_strings = [
        f"Factor: {row.get('Name', 'N/A')}; Definition: {row.get('Definition', 'N/A')}"
        for row in data
    ]
    
    all_split_chunks: List[List[str]] = []
    for text_string in chunk_list_strings:
        all_split_chunks.append(
            _split_single_text_for_events_factors(text_string, chunk_size, chunk_overlap)
        )
    return all_split_chunks
