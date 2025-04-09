"""
Example 5: Agent with Reflection and Self-Correction

This example demonstrates advanced agent capabilities with self-correction.
Key concepts:
- Implementing self-reflection
- Handling errors gracefully with retries
- Using ModelRetry for automatic retries
- Decorator-based tool registration
"""

from typing import Dict, List, Optional
from setup import OllamaModel, get_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from utils.markdown import to_markdown

# Define customer schema
class CustomerDetails(BaseModel):
    """Structure for incoming customer queries."""
    customer_id: str
    name: str
    email: str
    orders: Optional[List[dict]] = None

class ResponseModel(BaseModel):
    """Structured response with metadata."""
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")

def run_self_correction_agent():
    model = OllamaModel(OllamaModel.QWQ_32b).get_model()
    
    # Simulated database of shipping information
    shipping_info_db: Dict[str, str] = {
        "#12345": "Shipped on 2024-12-01",
        "#67890": "Out for delivery",
    }

    # Create a customer
    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
    )

    # Agent with reflection and self-correction
    agent = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses. "
            "Use tools to look up relevant information. "
            "Always greet the customer and provide a helpful response."
        ),
    )

    @agent.tool_plain()  # Add plain tool via decorator
    def get_shipping_status(order_id: str) -> str:
        """Get the shipping status for a given order ID."""
        shipping_status = shipping_info_db.get(order_id)
        if shipping_status is None:
            raise ModelRetry(
                f"No shipping information found for order ID {order_id}. "
                "Make sure the order ID starts with a #: e.g, #624743 "
                "Self-correct this if needed and try again."
            )
        return shipping_info_db[order_id]

    print("Running query with self-correction...")
    response = agent.run_sync(
        user_prompt="What's the status of my last order 12345?", deps=customer
    )

    print("\nAll messages:")
    print(response.all_messages())
    print("\nStructured response data:")
    print(response.data.model_dump_json(indent=2))

if __name__ == "__main__":
    run_self_correction_agent()
