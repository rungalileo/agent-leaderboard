import json
import time
from typing import Dict, List, Any
from galileo.experiments import run_experiment
from dotenv import load_dotenv
from galileo import galileo_context

# load_dotenv("../.env")


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

        # Simulate thinking/planning with LLM
        thinking_output = simulate_llm_thinking(
            turn["user_input"], conversation[: turn_id - 1]
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
                    metadata={
                        "location": tool_input.get("location", "unknown"),
                        "units": tool_input.get("units", "celsius"),
                        "turn_id": str(turn_id),
                        "session_id": session_id,
                        "tool_index": str(i),
                        "tool_type": "weather_api",
                    },
                    tool_call_id=tool_call_id,
                    tags=[
                        "weather_api",
                        tool_input.get("location", "unknown"),
                    ],
                )

                print("Tool result added to logger")

                tool_results.append(tool_result)

        # Generate response
        assistant_response = generate_response(
            turn["user_input"], tool_results, conversation[: turn_id - 1]
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
    # Use microseconds in the timestamp to ensure uniqueness
    experiment_name = f"weather-conversation-experiment-{int(time.time() * 1000000)}"

    # No galileo_context here - we're creating new loggers for each turn
    results = run_experiment(
        experiment_name=experiment_name,
        dataset=weather_dataset,
        function=weather_conversation_function,
        metrics=["tool_selection_quality"],
        project="agent-leaderboard-test",
    )
    print(f"Experiment completed with {len(results)} data points")
