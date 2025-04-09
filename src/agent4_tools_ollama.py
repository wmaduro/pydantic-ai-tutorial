"""
Example 4: Agent with Tools

This example shows how to enhance agents with custom tools.
Key concepts:
- Creating and registering tools
- Accessing context in tools
"""

from typing import Dict, List, Optional
from setup import OllamaModel, get_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from utils.markdown import to_markdown

# Define order schema
class Order(BaseModel):
    """Structure for order details."""
    order_id: str
    status: str
    items: List[str]

# Define customer schema
class CustomerDetails(BaseModel):
    """Structure for incoming customer queries."""
    customer_id: str
    name: str
    email: str
    orders: Optional[List[Order]] = None

class ResponseModel(BaseModel):
    """Structured response with metadata."""
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")

def run_tools_agent():
    model = OllamaModel(OllamaModel.QWEN2_5_14b).get_model()
    
    # Simulated database of shipping information
    shipping_info_db: Dict[str, str] = {
        "12345": "Shipped on 2024-12-01",
        "67890": "Out for delivery",
    }

    def get_shipping_info(ctx: RunContext[CustomerDetails]) -> str:
        """Get the customer's shipping information."""
        return shipping_info_db[ctx.deps.orders[0].order_id]

    # Agent with structured output, dependencies, and tools
    agent = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses. "
            "Use tools to look up relevant information."
            "Always great the customer and provide a helpful response."
        ),
        tools=[Tool(get_shipping_info, takes_ctx=True)],  # Add tool via kwarg
    )

    @agent.system_prompt
    async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
        return f"Customer details: {to_markdown(ctx.deps)}"

    # Create a customer with order details
    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
        orders=[
            Order(order_id="12345", status="shipped", items=["Blue Jeans", "T-Shirt"]),
        ],
    )

    print("Running query with tools...")
    response = agent.run_sync(
        user_prompt="What's the status of my last order?", deps=customer
    )

    print("\nAll messages:")
    print(response.all_messages())
    print("\nStructured response data:")
    print(response.data.model_dump_json(indent=2))

    print(
        "\nCustomer Details:\n"
        f"Name: {customer.name}\n"
        f"Email: {customer.email}\n\n"
        "Response Details:\n"
        f"{response.data.response}\n\n"
        "Status:\n"
        f"Follow-up Required: {response.data.follow_up_required}\n"
        f"Needs Escalation: {response.data.needs_escalation}"
    )

if __name__ == "__main__":
    run_tools_agent()
