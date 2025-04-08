    
"""
Basic setup for PydanticAI examples.
"""

import nest_asyncio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


# Apply nest_asyncio to allow nested event loops (needed for Jupyter/interactive environments)
nest_asyncio.apply()

# Initialize the OpenAI model
def get_model():
    return OpenAIModel("gpt-4o")

def get_model_ollama():
    # _model_name = 'llama3.3:latest' #ok
    # _model_name = 'qwen2.5:14b' #ok
    # _model_name = 'llama3:latest'
    # _model_name = 'mistral:7b'
    _model_name = 'qwq:32b' #ok
    # _model_name = 'gemma3:27b' 
    return OpenAIModel(model_name=_model_name, 
                       provider=OpenAIProvider(base_url='http://localhost:11434/v1')
                       )