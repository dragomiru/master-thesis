import os
import getpass
import tiktoken
from typing import Optional, Any
import certifi
import httpx 

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()

def _get_api_key_from_env_or_prompt(env_var_name: str, prompt_message: str) -> Optional[str]:
    """
    Tries to get an API key from environment variables.
    If not found, prompts the user securely.
    """
    api_key = os.getenv(env_var_name)
    if api_key:
        print(f"[INFO_API_KEY] Found API key in environment variable '{env_var_name}'.")
        return api_key

    print(f"Environment variable '{env_var_name}' not found for API key. Prompting user.")
    try:
        api_key = getpass.getpass(prompt_message)
        if not api_key:
            print(f"[ERROR_API_KEY] No input received for {env_var_name} from prompt.")
            return None
        return api_key
    except Exception as e:
        print(f"[ERROR_API_KEY] Could not prompt for API key '{env_var_name}': {e}")
        return None

def init_llm(model_identifier: str) -> Optional[Any]: # Simplified signature
    """
    Initializes and returns a Langchain chat model (OpenAI or Google).
    """
    if not model_identifier or not isinstance(model_identifier, str):
        print(f"[ERROR_INIT_LLM] 'model_identifier' must be a non-empty string. Received: {model_identifier}")
        return None

    print(f"[INFO_INIT_LLM] Attempting to initialize LLM: {model_identifier}")
    
    # Temporarily pop SSL_CERT_FILE and REQUESTS_CA_BUNDLE from env
    # This helps ensure that for the OpenAI call, we explicitly control the SSL context.
    original_ssl_cert_file = os.environ.pop('SSL_CERT_FILE', None)
    original_requests_ca_bundle = os.environ.pop('REQUESTS_CA_BUNDLE', None)

    chat_model = None
    try:
        if model_identifier.startswith("gpt"):
            openai_key = _get_api_key_from_env_or_prompt("OPENAI_API_KEY", "Enter your OpenAI API Key: ")
            if not openai_key:
                print("[ERROR_INIT_LLM] OpenAI API Key is essential for GPT models and could not be obtained.")
                # Restore env vars before returning if key is missing
                if original_ssl_cert_file is not None: os.environ['SSL_CERT_FILE'] = original_ssl_cert_file
                if original_requests_ca_bundle is not None: os.environ['REQUESTS_CA_BUNDLE'] = original_requests_ca_bundle
                return None
            
            # Explicitly configure the httpx client with certifi's CA bundle for OpenAI
            ca_bundle_path = certifi.where()
                
            proxies = {
                "http://": os.getenv("HTTP_PROXY"),
                "https://": os.getenv("HTTPS_PROXY"),
            }
            active_proxies = {k: v for k, v in proxies.items() if v}
            
            if active_proxies:
                print(f"[DEBUG_INIT_LLM_GPT] Configuring httpx client with proxies: {active_proxies}")
                custom_http_client = httpx.Client(verify=ca_bundle_path, proxies=active_proxies)
            else:
                print("[DEBUG_INIT_LLM_GPT] No HTTP_PROXY/HTTPS_PROXY found, configuring httpx client without explicit proxies.")
                custom_http_client = httpx.Client(verify=ca_bundle_path)

            chat_model = ChatOpenAI(
                model=model_identifier,
                api_key=openai_key,
                http_client=custom_http_client
            )
            print(f"[INFO_INIT_LLM] OpenAI model '{model_identifier}' initialized successfully.")

        elif model_identifier.startswith("gemini"):
            google_key = _get_api_key_from_env_or_prompt("GOOGLE_API_KEY", "Enter your Google AI Studio API Key: ")
            if not google_key:
                print("[ERROR_INIT_LLM] Google API Key is essential for Gemini models.")
                # Restore env vars before returning if key is missing
                if original_ssl_cert_file is not None: os.environ['SSL_CERT_FILE'] = original_ssl_cert_file
                if original_requests_ca_bundle is not None: os.environ['REQUESTS_CA_BUNDLE'] = original_requests_ca_bundle
                return None
            
            # print(f"[DEBUG_INIT_LLM_GEMINI] Using Google API Key: {'Present' if google_key else 'Not Present'}")
            
            # print(f"[DEBUG_INIT_LLM_GEMINI] Instantiating ChatGoogleGenerativeAI with model='{model_identifier}' and explicit api_key.")
            chat_model = ChatGoogleGenerativeAI(
                model=model_identifier, 
                google_api_key=google_key
                # Not passing generation_config to keep it minimal
            )
            print(f"[INFO_INIT_LLM] Gemini model '{model_identifier}' initialized successfully.")
        else:
            print(f"[ERROR_INIT_LLM] Unsupported model identifier prefix: '{model_identifier}'.")
            return None # Env vars will be restored in finally
        
        return chat_model
        
    except Exception as e: 
        print(f"[ERROR_INIT_LLM] Failed to initialize LLM '{model_identifier}': {e}")
        print(f"    Error Type: {type(e)}")
        return None
    finally:
        if original_ssl_cert_file is not None:
            os.environ['SSL_CERT_FILE'] = original_ssl_cert_file
        if original_requests_ca_bundle is not None:
            os.environ['REQUESTS_CA_BUNDLE'] = original_requests_ca_bundle

def count_tokens_openai(text: str, model_name: str) -> int:
    if not model_name or not model_name.startswith("gpt"): 
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        try:
            encoding = tiktoken.get_encoding("cl100k_base") # Fallback for gpt-3.5/4
            return len(encoding.encode(text))
        except Exception: 
            return 0 # Fallback if cl100k_base also fails
    except Exception as e:
        print(f"[ERROR_TOKEN_COUNT] Could not count tokens for model '{model_name}': {e}")
        return 0

def get_conversation_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)