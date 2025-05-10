import os
import getpass
import tiktoken
from typing import Optional, Any

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()

def _get_api_key_from_env_or_prompt(env_var_name: str, prompt_message: str) -> Optional[str]:
    """Tries to get an API key from environment variables, then prompts the user."""
    api_key = os.getenv(env_var_name)
    if not api_key:
        print(f"Environment variable {env_var_name} not found.")
        try:
            api_key = getpass.getpass(prompt_message)
        except Exception as e:
            print(f"Could not prompt for API key: {e}")
            return None
    return api_key

def init_llm(model_type: str, openai_api_key: Optional[str] = None, google_api_key: Optional[str] = None, temperature: float = 0.2,
    max_tokens: Optional[int] = None) -> Optional[Any]:
    """
    Initializes and returns a Langchain chat model (OpenAI or Google).

    Args:
        model_type (str): The identifier for the model.
        openai_api_key (Optional[str]): OpenAI API key.
        google_api_key (Optional[str]): Google API key.
        temperature (float): Sampling temperature for the LLM.
        max_tokens (Optional[int]): Max tokens to generate.

    Returns:
        Optional[Any]: The initialized Langchain chat model instance or None on failure.
    """
    print(f"[INFO] Initializing LLM: {model_type}")
    try:
        if model_type.startswith("gpt"):
            key = openai_api_key or _get_api_key_from_env_or_prompt("OPENAI_API_KEY", "Enter your OpenAI API Key: ")
            if not key:
                print("[ERROR] OpenAI API Key is required for GPT models.")
                return None
            os.environ["OPENAI_API_KEY"] = key
            
            model_kwargs = {"temperature": temperature}
            if max_tokens:
                model_kwargs["max_tokens"] = max_tokens
            chat_model = init_chat_model(model_name=model_type, model_provider="openai", model_kwargs=model_kwargs)

        elif model_type.startswith("gemini"):
            key = google_api_key or _get_api_key_from_env_or_prompt("GOOGLE_API_KEY", "Enter your Google AI Studio API Key: ")
            if not key:
                print("[ERROR] Google API Key is required for Gemini models.")
                return None
            
            chat_model = ChatGoogleGenerativeAI(model=model_type,google_api_key=key)
        else:
            print(f"[ERROR] Unsupported model type: {model_type}")
            return None
        
        print(f"[INFO] LLM '{model_type}' initialized successfully.")
        return chat_model
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM '{model_type}': {e}")
        return None

def count_tokens_openai(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Efficiently counts tokens in a text for a given OpenAI model.
    Returns 0 if not an OpenAI model or if an error occurs.
    """
    if not model_name.startswith("gpt"):
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        token_integers = encoding.encode(text)
        num_tokens = len(token_integers)
        return num_tokens
    except Exception as e:
        print(f"[ERROR] Could not count tokens for model {model_name}: {e}")
        return 0

def get_conversation_memory() -> ConversationBufferMemory:
    """
    Creates and returns a new conversation buffer memory.
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if __name__ == '__main__':
    print("--- Testing llm_interaction/llm_models.py ---")
    
    try:
        from config import DEFAULT_LLM_MODEL, GOOGLE_API_KEY, OPENAI_API_KEY
    except ImportError:
        print("[WARNING] config.py not found or not fully configured for testing llm_models.py")
        DEFAULT_LLM_MODEL = "gemini-1.5-flash-latest"
        GOOGLE_API_KEY = None
        OPENAI_API_KEY = None
    
    print(f"\nAttempting to initialize default model: {DEFAULT_LLM_MODEL}")
    llm = init_llm(DEFAULT_LLM_MODEL, openai_api_key=OPENAI_API_KEY, google_api_key=GOOGLE_API_KEY)
    if llm:
        print(f"[SUCCESS] Successfully initialized {DEFAULT_LLM_MODEL}")
    else:
        print(f"[FAIL] Failed to initialize {DEFAULT_LLM_MODEL}")

    # Test token counting for an OpenAI model
    gpt_model_name_for_token_test = "gpt-4o-mini"
    sample_text = "This is a test sentence for token counting."
    tokens = count_tokens_openai(sample_text, model_name=gpt_model_name_for_token_test)
    print(f"\nTokens for '{sample_text}' with '{gpt_model_name_for_token_test}': {tokens}")
    if tokens > 0:
        print("[SUCCESS] Token counting seems to work for OpenAI model.")
    else:
        print("[INFO] Token counting returned 0 (expected if not testing an OpenAI model or if tiktoken failed).")

    memory = get_conversation_memory()
    print(f"\nConversation memory initialized: {memory.memory_key}")
    print("--------------------------------------------")