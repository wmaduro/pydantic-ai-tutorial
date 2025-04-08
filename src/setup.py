"""
Basic setup for PydanticAI examples.
"""

import nest_asyncio
from pydantic_ai.models.openai import OpenAIModel

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/interactive environments)
nest_asyncio.apply()

# Initialize the OpenAI model
def get_model():
    return OpenAIModel("gpt-4o")
