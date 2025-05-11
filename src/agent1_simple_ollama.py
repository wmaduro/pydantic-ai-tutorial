"""
Example 1: Simple Agent - Hello World Example

This example demonstrates the basic usage of PydanticAI agents.
Key concepts:
- Creating a basic agent with a system prompt
- Running synchronous queries
- Accessing response data, message history, and costs
"""

from setup import OllamaModel
from pydantic_ai import Agent
from colorama import init, Fore

init()

def run_simple_agent():
    model = OllamaModel(OllamaModel.QWEN2_5_14b).get_model()
    
    # Create a basic agent with a system prompt
    agent1 = Agent(
        model=model,
        # system_prompt="You are a helpful customer support agent. Be concise and friendly.",
        system_prompt="The response must be in one sentence, max 10 words.",
    )

    # Example usage of basic agent
    print("Running first query...")
    response = agent1.run_sync("How can I track my order #12345?")
    print("\nResponse data:")
    print(response.data)
    # print("\nAll messages:")
    # print(response.all_messages())

    try:
        print(f'\nCost: {response.cost()}')
    except:
        print(Fore.RED + "\nCost: Not available" + Fore.RESET)
        
    # Continuing the conversation with message history
    print("\nRunning follow-up query with message history...")
    response2 = agent1.run_sync(
        user_prompt="What was my previous question?",
        message_history=response.new_messages(),
    )
    print("\nResponse data:")
    print(Fore.LIGHTGREEN_EX + response2.data)

if __name__ == "__main__":
    run_simple_agent()
