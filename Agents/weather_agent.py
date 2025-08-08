import os
import requests
import json
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
load_dotenv()
# --- 1. Define the Tool ---
@tool
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a specified city.
    Args:
        city (str): The name of the city for which to get the weather.
    """
    print(f"--- Calling Weather Tool for {city} ---")
    if not isinstance(city, str):
        return "Error: City must be a string."
        
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        current_condition = data['current_condition'][0]
        weather_description = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        temp_f = current_condition['temp_F']
        humidity = current_condition['humidity']
        wind_speed_kmph = current_condition['windspeedKmph']
        
        return json.dumps({
            "city": city,
            "temperature_celsius": temp_c,
            "temperature_fahrenheit": temp_f,
            "description": weather_description,
            "humidity_percent": humidity,
            "wind_speed_kmph": wind_speed_kmph
        })
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing weather data. The city might be invalid. Details: {e}"

# --- 2. Define Agent State and Graph Nodes ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    """Invokes the LLM to generate a response or decide on a tool call."""
    print("--- Calling LLM ---")
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# --- 3. Set Up the Model and Tools ---
tools = [get_weather]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
tool_node = ToolNode(tools)

# --- 4. Define Conditional Logic ---
def should_continue(state: AgentState) -> str:
    """Determines whether to continue with a tool call or end the turn."""
    print("--- Checking for Tool Calls ---")
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# --- 5. Construct the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()

# --- 6. Main Interaction Loop (Corrected) ---
def main():
    """Manages the conversation loop with the user."""
    print("🤖 Weather AI Agent is ready. Type 'quit' or 'exit' to stop.")
    
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("🤖 Goodbye!")
            break
        
        human_message = HumanMessage(content=user_input)
        conversation_history.append(human_message)

        graph_input = {"messages": conversation_history}
        
        final_response = None
        print("\n--- Agent Thinking ---")
        for chunk in app.stream(graph_input):
            for key, value in chunk.items():
                print(f"Output from node '{key}':")
                if 'messages' in value:
                    for message in value['messages']:
                        # THIS IS THE CORRECTED LOGIC:
                        # We check if the message is an AIMessage and if it has no tool calls.
                        # This is the signature of a final human-readable response.
                        if isinstance(message, AIMessage) and not message.tool_calls:
                            final_response = message
                        
                        # Print the representation of each step for transparency
                        message.pretty_print()
            print("---")
            
        if final_response:
            conversation_history.append(final_response)
            print(f"\nAI: {final_response.content}\n")
        else:
            print("\nAI: I seem to have run into an issue processing that. Please try again.\n")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("🔴 ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run the agent.")
    else:
        main()