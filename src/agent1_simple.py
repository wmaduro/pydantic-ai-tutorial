"""
Example 1: Simple Agent - Hello World Example

This example demonstrates the basic usage of PydanticAI agents.
Key concepts:
- Creating a basic agent with a system prompt
- Running synchronous queries
- Accessing response data, message history, and costs
"""

from setup import get_model
from pydantic_ai import Agent

def run_simple_agent():
    model = get_model()
    
    # Create a basic agent with a system prompt
    agent1 = Agent(
        model=model,
        system_prompt="You are a helpful customer support agent. Be concise and friendly.",
    )

    # Example usage of basic agent
    print("Running first query...")
    response = agent1.run_sync("How can I track my order #12345?")
    print("\nResponse data:")
    print(response.data)
    print("\nAll messages:")
    print(response.all_messages())
    print("\nCost:")
    print(response.cost())

    # Continuing the conversation with message history
    print("\nRunning follow-up query with message history...")
    response2 = agent1.run_sync(
        user_prompt="What was my previous question?",
        message_history=response.new_messages(),
    )
    print("\nResponse data:")
    print(response2.data)

if __name__ == "__main__":
    run_simple_agent()
