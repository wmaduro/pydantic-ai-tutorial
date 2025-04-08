    
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

class OllamaModel:
    LLAMA3_3 = "llama3.3:latest" #ok
    QWEN2_5_14b = "qwen2.5:14b" #ok
    LLAMA3 = "llama3:latest"
    MISTRAL = "mistral:7b"
    QWQ_32b = "qwq:32b" #ok
    GEMMA3_27b = "gemma3:27b"
    
    def __init__(self, model_name=QWQ_32b):
        self.model_name = model_name
    
    def get_model(self):
        return OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url='http://localhost:11434/v1')
        )