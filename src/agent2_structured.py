"""
Example 2: Agent with Structured Response

This example shows how to get structured, type-safe responses from the agent.
Key concepts:
- Using Pydantic models to define response structure
- Type validation and safety
- Field descriptions for better model understanding
"""

from setup import get_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class ResponseModel(BaseModel):
    """Structured response with metadata."""
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")

def run_structured_agent():
    model = get_model()
    
    # Create an agent that returns structured data
    agent2 = Agent(
        model=model,
        result_type=ResponseModel,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses."
        ),
    )

    print("Running query with structured response...")
    response = agent2.run_sync("How can I track my order #12345?")
    print("\nStructured response data:")
    print(response.data.model_dump_json(indent=2))

if __name__ == "__main__":
    run_structured_agent()
