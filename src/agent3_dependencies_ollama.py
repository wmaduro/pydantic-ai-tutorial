from typing import List, Optional
from setup import OllamaModel, get_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
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

def run_dependencies_agent():
    model = OllamaModel(OllamaModel.QWEN2_5_14b).get_model()
    
    # Agent with structured output and dependencies
    agent = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        # system_prompt=(
        #     "You are an intelligent customer support agent. "
        #     "Analyze queries carefully and provide structured responses. "
        #     "Always great the customer and provide a helpful response."
        # ),
    )

    # Add dynamic system prompt based on dependencies
    @agent.system_prompt
    async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
        lixo = f"Customer details: {to_markdown(ctx.deps)}"
        # print(f'=================== {lixo}')        
        return lixo

    # Create a customer with order details
    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
        orders=[
            Order(order_id="12345", status="shipped", items=["Blue Jeans", "T-Shirt"]),
            Order(order_id="12346", status="pending", items=["Snikers", "Jandals"]),
            Order(order_id="12347", status="returned", items=["Sunnies"]),
            Order(order_id="12348", status="returned", items=["Black Jeans"]),
            Order(order_id="12349", status="returned", items=["Blue Jeans"]),
        ],
    )

    print("Running query with dependencies...")
    # response = agent.run_sync(user_prompt="What did I order?", deps=customer)
    # response = agent.run_sync(user_prompt="did I order a flip flop?", deps=customer)
    # response = agent.run_sync(user_prompt="list the orders that have flip flops?", deps=customer)
    
    # response = agent.run_sync(user_prompt="list the orders that have flip flops?", deps=customer)
    # response = agent.run_sync(user_prompt="give me only the number of the order where I bought a foowear.", deps=customer)

    # response = agent.run_sync(user_prompt="did I buy anything that I could use with my new hat?", deps=customer)

    response = agent.run_sync(user_prompt="list all products that I ordered and show only their names, " \
    "not the order information. " \
    "show the names in uppercase" \
    "remove duplication from the list" \
    "show only the product names" \
    # "it they have compound names, show them as a single word separate by #" \
    , deps=customer)

    # print("\nAll messages:")
    # print(response.all_messages())
    print("\n------> Structured response data:")
    print(response.data.model_dump_json(indent=2))

    print(
        "\n------> Customer Details:\n"
        f"Name: {customer.name}\n"
        f"Email: {customer.email}\n\n"
        "Response Details:\n"
        f"{response.data.response}\n\n"
        "Status:\n"
        f"Follow-up Required: {response.data.follow_up_required}\n"
        f"Needs Escalation: {response.data.needs_escalation}"
    )

if __name__ == "__main__":
    run_dependencies_agent()
