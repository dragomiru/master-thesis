from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, Optional


def create_llm_chain(llm: Any, prompt: ChatPromptTemplate, memory: ConversationBufferMemory = None, 
                     verbose: bool = False) -> LLMChain:
    """
    Creates an LLMChain with the given LLM, prompt, and memory.
    """
    if not llm:
        raise ValueError("LLM must be provided to create a chain.")
    if not prompt:
        raise ValueError("Prompt must be provided to create a chain.")
        
    chain_kwargs = {"llm": llm, "prompt": prompt, "verbose": verbose}
    if memory:
        chain_kwargs["memory"] = memory
        
    return LLMChain(**chain_kwargs)

def run_chain(chain: LLMChain, inputs: Dict[str, Any]) -> Optional[str]:
    """
    Runs the given LLMChain with the provided inputs and extracts the text response.

    Args:
        chain (LLMChain): The Langchain LLMChain to run.
        inputs (Dict[str, Any]): A dictionary of inputs required by the chain's prompt.
                                (e.g., {"relevant_report_text": "..."} or {"extraction_result": "..."})

    Returns:
        Optional[str]: The text output from the LLM, or None if an error occurs or output is not found.
    """
    try:
        print(f"[INFO] Running LLM chain with input keys: {list(inputs.keys())}")        
        response = chain.invoke(inputs)
        
        if isinstance(response, dict) and "text" in response:
            return response["text"].strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            print(f"[WARNING] Unexpected response format from LLM chain: {type(response)}. Full response: {response}")
            # Attempt to find text in common places if it's a dict
            if isinstance(response, dict):
                for key in ["output", "result", "answer"]:
                    if key in response and isinstance(response[key], str):
                        return response[key].strip()
            return str(response) # Fallback to string representation

    except Exception as e:
        print(f"[ERROR] Error running LLM chain: {e}")
        return None

if __name__ == '__main__':
    print("--- Testing llm_interaction/chains.py ---")
    # This requires a mock LLM or a real one to be initialized for a full test.
    # We'll mock the behavior for demonstration.

    from langchain.llms.fake import FakeListLLM # Using a fake LLM for testing
    from langchain.prompts import PromptTemplate

    # 1. Test create_llm_chain
    mock_llm = FakeListLLM(responses=["Mocked LLM response."])
    simple_template = "Tell me about {subject}."
    simple_prompt = PromptTemplate.from_template(simple_template)
    
    # Create ChatPromptTemplate from PromptTemplate for compatibility
    system_message_prompt = SystemMessagePromptTemplate(prompt=simple_prompt)
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt])


    try:
        test_chain = create_llm_chain(llm=mock_llm, prompt=chat_prompt_template, verbose=True)
        print("[SUCCESS] LLMChain created successfully.")

        # 2. Test run_chain
        chain_inputs = {"subject": "Langchain"}
        output = run_chain(test_chain, chain_inputs)
        
        print(f"\nChain output for input '{chain_inputs}': {output}")
        assert output == "Mocked LLM response."
        print("[SUCCESS] run_chain executed and returned expected mock output.")

    except ValueError as ve:
        print(f"[FAIL] Test failed during chain creation: {ve}")
    except Exception as e:
        print(f"[FAIL] An unexpected error occurred during chain testing: {e}")
    
    print("-------------------------------------")