import json
import time
from typing import Dict, List, Any
from galileo.experiments import run_experiment
from dotenv import load_dotenv
from galileo import galileo_context
from galileo.projects import get_project, create_project

load_dotenv("../.env")


def weather_conversation_function(input_data: Dict[str, Any]) -> str:
    """
    Process a multi-turn weather conversation based on input data.
    This function handles the conversation and returns the final results as a JSON string.
    """
    # Extract conversation data from the input
    conversation = input_data.get("conversation", [])
    session_id = input_data.get("session_id", "test-session")

    logger = galileo_context.get_logger_instance()

    # Initialize results container
    results = {
        "session_id": session_id,
        "turns_completed": 0,
        "turns_results": [],
        "success": False,
    }

    # Process each conversation turn
    for turn in conversation:
        turn_id = turn["turn_id"]
        turn_start_time = time.time()

        # Start a workflow span for this turn
        workflow_name = f"turn_{turn_id}_workflow"
        logger.add_workflow_span(
            input=turn["user_input"],
            name=workflow_name,
            metadata={
                "turn_id": str(turn_id),
                "session_id": session_id,
            },
            tags=[f"turn_{turn_id}", "conversation_turn"],
        )

        # Define the weather API tool with proper schema
        weather_tool = {
            "title": "get_weather",
            "description": "Get current weather information for a specific location",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather information for",
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units (celsius or fahrenheit)",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
            "response_schema": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "Current temperature",
                    },
                    "condition": {
                        "type": "string",
                        "description": "Weather condition (e.g., Sunny, Cloudy, Rainy)",
                    },
                    "humidity": {
                        "type": "number",
                        "description": "Humidity percentage",
                    },
                    "wind_speed": {
                        "type": "number",
                        "description": "Wind speed in km/h",
                    },
                },
            },
        }

        # Simulate thinking/planning with LLM (tool selection)
        thinking_start_time = time.time()
        thinking_output = simulate_llm_thinking(
            turn["user_input"], conversation[: turn_id - 1]
        )
        thinking_duration_ns = int((time.time() - thinking_start_time) * 1_000_000_000)

        # Generate tool selection output
        if "tool_calls" in turn:
            tool_selection_output = json.dumps(
                {
                    "selected_tools": [
                        {"tool": "get_weather", "parameters": tool_call}
                        for tool_call in turn["tool_calls"]
                    ]
                },
                indent=2,
            )
        else:
            tool_selection_output = json.dumps({"selected_tools": []}, indent=2)

        # Log the LLM tool selection span
        logger.add_llm_span(
            input=turn["user_input"],
            output=tool_selection_output,
            model="simulated-llm",
            tools=[weather_tool],
            duration_ns=thinking_duration_ns,
            name="tool_selection",
            tags=["tool_selection", f"turn_{turn_id}"],
        )

        # Process tool calls if present
        tool_results = []
        if "tool_calls" in turn:
            for i, tool_input in enumerate(turn["tool_calls"]):
                tool_start_time = time.time()
                tool_input_str = json.dumps(tool_input)
                tool_result = simulate_weather_api(tool_input)
                tool_duration_ns = int((time.time() - tool_start_time) * 1_000_000_000)
                tool_call_id = f"{session_id}_turn{turn_id}_tool{i}"

                logger.add_tool_span(
                    input=tool_input_str,
                    output=json.dumps(tool_result),
                    name=f"weather_api_{tool_input.get('location', 'unknown')}",
                    duration_ns=tool_duration_ns,
                    tool_call_id=tool_call_id,
                    tags=[
                        "weather_api",
                        tool_input.get("location", "unknown"),
                    ],
                )

                print("Tool result added to logger")

                tool_results.append(tool_result)

        # Generate response based on tool results
        response_start_time = time.time()
        assistant_response = generate_response(
            turn["user_input"], tool_results, conversation[: turn_id - 1]
        )
        response_duration_ns = int((time.time() - response_start_time) * 1_000_000_000)

        # Create a structured input for response generation
        response_input = {
            "user_query": turn["user_input"],
            "tool_results": tool_results,
            "conversation_context": (
                "Previous conversation context from user"
                if turn_id > 1
                else "Initial conversation"
            ),
        }

        # Log the LLM response generation span
        logger.add_llm_span(
            input=json.dumps(response_input, indent=2),
            output=assistant_response,
            model="simulated-response-model",
            tools=[weather_tool],
            duration_ns=response_duration_ns,
            name="response_generation",
            tags=["response_generation", f"turn_{turn_id}"],
        )

        # Calculate turn duration
        turn_duration_ns = int((time.time() - turn_start_time) * 1_000_000_000)

        # Conclude the workflow span for this turn
        logger.conclude(
            output=assistant_response,
            duration_ns=turn_duration_ns,
        )

        # Record turn results
        turn_result = {
            "turn_id": turn_id,
            "user_input": turn["user_input"],
            "thinking": thinking_output,
            "tool_results": tool_results,
            "assistant_response": assistant_response,
            "processing_time_ms": int((time.time() - turn_start_time) * 1000),
        }
        results["turns_results"].append(turn_result)
        results["turns_completed"] += 1

    # Set success flag
    results["success"] = True if results["turns_completed"] > 0 else False
    results["conversation_summary"] = (
        "Completed multi-turn conversation about weather in different cities"
    )

    return json.dumps(results)


def simulate_llm_thinking(user_input: str, history: List[Dict]) -> str:
    """Simulate LLM thinking process"""
    if "San Francisco" in user_input:
        return "I need to check the weather in San Francisco."
    elif "New York" in user_input:
        return "User is asking about weather in New York."
    else:
        return f"Analyzing user query about {user_input}"


def simulate_weather_api(tool_input: Dict) -> Dict:
    """Simulate weather API call"""
    location = tool_input.get("location", "Unknown")

    # Return mock data based on location
    if location == "San Francisco":
        return {
            "temperature": 18,
            "condition": "Partly Cloudy",
            "humidity": 72,
            "wind_speed": 12,
        }
    elif location == "New York":
        return {
            "temperature": 15,
            "condition": "Rainy",
            "humidity": 85,
            "wind_speed": 20,
        }
    else:
        return {
            "temperature": 20,
            "condition": "Clear",
            "humidity": 60,
            "wind_speed": 10,
        }


def generate_response(
    user_input: str, tool_results: List[Dict], history: List[Dict]
) -> str:
    """Generate assistant response based on weather data"""
    # In a real application, you would use OpenAI here
    if tool_results:
        weather = tool_results[0]
        if "San Francisco" in user_input:
            return f"The current weather in San Francisco is {weather['condition'].lower()} with a temperature of {weather['temperature']}°C. The humidity is {weather['humidity']}% and wind speed is {weather['wind_speed']} km/h."
        elif "New York" in user_input or "How about New York" in user_input:
            return f"In New York, it's currently {weather['condition'].lower()} with a temperature of {weather['temperature']}°C. The humidity is quite high at {weather['humidity']}% with wind speeds of {weather['wind_speed']} km/h."
        else:
            location = (
                user_input.split("weather in ")[-1].split("?")[0]
                if "weather in" in user_input
                else "the requested location"
            )
            return f"The weather in {location} is {weather['condition'].lower()} with a temperature of {weather['temperature']}°C."
    else:
        return "I don't have current weather information for that location."


# Create example dataset for the experiment
weather_dataset = [
    {
        "session_id": "test-session-1",
        "conversation": [
            {
                "turn_id": 1,
                "user_input": "What's the weather in San Francisco?",
                "tool_calls": [{"location": "San Francisco", "units": "celsius"}],
            },
            {
                "turn_id": 2,
                "user_input": "How about New York?",
                "tool_calls": [{"location": "New York", "units": "celsius"}],
            },
        ],
    },
    {
        "session_id": "test-session-2",
        "conversation": [
            {
                "turn_id": 1,
                "user_input": "What's the weather in Paris?",
                "tool_calls": [{"location": "Paris", "units": "celsius"}],
            }
        ],
    },
    {
        "session_id": "test-session-3",
        "conversation": [
            {
                "turn_id": 1,
                "user_input": "What's the weather in Tokyo?",
                "tool_calls": [{"location": "Tokyo", "units": "celsius"}],
            },
            {
                "turn_id": 2,
                "user_input": "And what about London?",
                "tool_calls": [{"location": "London", "units": "celsius"}],
            },
            {
                "turn_id": 3,
                "user_input": "Is it warmer in Dubai?",
                "tool_calls": [{"location": "Dubai", "units": "celsius"}],
            },
        ],
    },
]

# Run the experiment
if __name__ == "__main__":
    project_name = "test"
    if not bool(get_project(name=project_name)):
            print(f"Creating project: {project_name}")
            create_project(project_name)
            
    # Use microseconds in the timestamp to ensure uniqueness
    experiment_name = f"weather-conversation-experiment-{int(time.time() * 1000000)}"
    METRICS = [
        "tool_selection_quality",
        "agentic_session_success",
    ]

    # No galileo_context here - we're creating new loggers for each turn
    results = run_experiment(
        experiment_name,
        dataset=weather_dataset,
        function=weather_conversation_function,
        metrics=METRICS,
        project=project_name,
    )
    print(f"Experiment completed with {len(results)} data points")
