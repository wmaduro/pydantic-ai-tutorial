"""
Main script to run all PydanticAI examples.
"""

import sys

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "1" or example == "simple":
            print_header("Example 1: Simple Agent")
            from agent1_simple import run_simple_agent
            run_simple_agent()
            
        elif example == "2" or example == "structured":
            print_header("Example 2: Structured Response")
            from agent2_structured import run_structured_agent
            run_structured_agent()
            
        elif example == "3" or example == "dependencies":
            print_header("Example 3: Agent with Dependencies")
            from agent3_dependencies import run_dependencies_agent
            run_dependencies_agent()
            
        elif example == "4" or example == "tools":
            print_header("Example 4: Agent with Tools")
            from agent4_tools import run_tools_agent
            run_tools_agent()
            
        elif example == "5" or example == "self-correction":
            print_header("Example 5: Agent with Self-Correction")
            from agent5_self_correction import run_self_correction_agent
            run_self_correction_agent()
            
        else:
            print(f"Unknown example: {example}")
            print("Available examples: 1/simple, 2/structured, 3/dependencies, 4/tools, 5/self-correction")
    else:
        print("Please specify which example to run:")
        print("python run_examples.py [example_number or name]")
        print("Available examples: 1/simple, 2/structured, 3/dependencies, 4/tools, 5/self-correction")
