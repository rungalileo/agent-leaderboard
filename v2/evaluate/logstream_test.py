import time
import json
import uuid
from galileo import GalileoLogger
from dotenv import load_dotenv

load_dotenv("../.env")


def test_galileo_logging():
    """
    Enhanced test function to demonstrate Galileo logging with multi-turn conversation.
    Simulates a complete conversation flow with multiple user interactions.
    Uses fixed input/output values for testing.
    """
    # Initialize the Galileo logger
    logger = GalileoLogger(project="test-project", log_stream="test-multi-turn-stream")

    print("Initialized Galileo logger")

    # Start a session trace
    session_id = f"test-session-{uuid.uuid4()}"
    trace_input = {
        "session_id": session_id,
        "scenario": "multi-turn-weather-conversation",
        "user_id": "test-user-456",
        "initial_context": {
            "user_location": "Unknown",
            "preferences": {"units": "celsius"},
        },
    }

    # Make sure we have an active trace before the loop
    if not logger.traces:
        logger.start_trace(
            input=json.dumps(trace_input),
            name="multi_turn_conversation",
            tags=["test", "demo", "multi-turn"],
            metadata={"environment": "development", "version": "0.1.0"},
        )

    # Conversation turns data
    conversation = [
        {
            "turn_id": 1,
            "user_input": "What's the weather in San Francisco?",
            "thinking": "I need to check the weather in San Francisco.",
            "tool_calls": [{"location": "San Francisco", "units": "celsius"}],
            "tool_results": [
                {
                    "temperature": 18,
                    "condition": "Partly Cloudy",
                    "humidity": 72,
                    "wind_speed": 12,
                }
            ],
            "assistant_response": "The current weather in San Francisco is partly cloudy with a temperature of 18°C. The humidity is 72% and wind speed is 12 km/h.",
        },
        {
            "turn_id": 2,
            "user_input": "How about New York?",
            "thinking": "User is now asking about weather in New York after previously asking about San Francisco.",
            "tool_calls": [{"location": "New York", "units": "celsius"}],
            "tool_results": [
                {
                    "temperature": 15,
                    "condition": "Rainy",
                    "humidity": 85,
                    "wind_speed": 20,
                }
            ],
            "assistant_response": "In New York, it's currently rainy with a temperature of 15°C. The humidity is quite high at 85% with wind speeds of 20 km/h.",
        },
    ]

    # Process each conversation turn
    for turn in conversation:
        turn_id = turn["turn_id"]
        print(f"\nProcessing turn {turn_id}")

        # Start a new trace for each turn
        turn_start_time = time.time()
        try:
            turn_span = logger.add_workflow_span(
                input=json.dumps(
                    {"user_input": turn["user_input"], "turn_id": turn_id}
                ),
                output=None,
                name=f"conversation_turn_{turn_id}",
                tags=["turn", f"turn-{turn_id}"],
                metadata={"turn_number": str(turn_id), "session_id": session_id},
            )

            if turn_span is None:
                raise ValueError("add_workflow_span returned None")

        except Exception as e:
            print(f"Error creating workflow span: {e}")
            # Fallback to using the trace itself
            turn_span = logger.current_parent() or logger.traces[-1]

        # Add a safety check
        if turn_span is None:
            print("WARNING: turn_span is None, creating a trace instead")
            # Try using start_trace as a fallback
            temp_trace = logger.start_trace(
                input=json.dumps(
                    {"user_input": turn["user_input"], "turn_id": turn_id}
                ),
                name=f"conversation_turn_{turn_id}",
                tags=["turn", f"turn-{turn_id}"],
                metadata={"turn_number": turn_id, "session_id": session_id},
            )
            turn_span = temp_trace  # Use the trace as the parent span

        # Add user message span
        user_span = logger.add_llm_span(
            input=json.dumps({"text": turn["user_input"]}),
            output=json.dumps({"parsed_intent": f"intent_turn_{turn_id}"}),
            name=f"user_message_{turn_id}",
            tags=["user-message"],
            metadata={"turn_id": str(turn_id)},
            model="user-message",
        )
        print(f"Added user message span for turn {turn_id}")

        # Add thinking/planning span
        thinking_start_time = time.time()
        time.sleep(0.2)  # Simulate processing time
        thinking_duration_ns = int((time.time() - thinking_start_time) * 1_000_000_000)

        thinking_span = logger.add_llm_span(
            input=json.dumps(
                {
                    "user_message": turn["user_input"],
                    "conversation_history": [
                        t for t in conversation if t["turn_id"] < turn_id
                    ],
                }
            ),
            output=turn["thinking"],
            model="gpt-4",
            duration_ns=thinking_duration_ns,
            name=f"thinking_planning_{turn_id}",
            metadata={"temperature": "0.7", "turn_id": str(turn_id)},
            tags=["llm", "reasoning", "planning"],
        )
        print(f"Added thinking span for turn {turn_id}")

        # Add tool spans if this turn has tool calls
        if "tool_calls" in turn and "tool_results" in turn:
            for idx, (tool_input, tool_output) in enumerate(
                zip(turn["tool_calls"], turn["tool_results"])
            ):
                tool_start_time = time.time()
                time.sleep(0.3)  # Simulate tool processing time
                tool_duration_ns = int((time.time() - tool_start_time) * 1_000_000_000)

                tool_span = logger.add_tool_span(
                    input=json.dumps(tool_input),
                    output=json.dumps(tool_output),
                    name=f"weather_api_call_{turn_id}_{idx}",
                    duration_ns=tool_duration_ns,
                    metadata={
                        "api_version": "v1.0",
                        "location": tool_input.get("location", "unknown"),
                    },
                    tags=["tool", "weather-api"],
                )
                print(f"Added tool span {idx} for turn {turn_id}")

        # Add response generation span
        response_start_time = time.time()
        time.sleep(0.4)  # Simulate LLM processing time
        response_duration_ns = int((time.time() - response_start_time) * 1_000_000_000)

        # Construct the input context based on available information
        context_input = {
            "user_query": turn["user_input"],
            "conversation_history": [t for t in conversation if t["turn_id"] < turn_id],
        }

        if "tool_results" in turn:
            context_input["tool_results"] = turn["tool_results"]

        response_span = logger.add_llm_span(
            input=json.dumps(context_input),
            output=turn["assistant_response"],
            model="gpt-4",
            duration_ns=response_duration_ns,
            name=f"generate_response_{turn_id}",
            metadata={
                "temperature": "0.5",
                "max_tokens": "1024",
                "turn_id": str(turn_id),
            },
            tags=["llm", "response-generation"],
        )
        print(f"Added response generation span for turn {turn_id}")

        # Close the turn span
        turn_duration_ns = int((time.time() - turn_start_time) * 1_000_000_000)
        logger.conclude(
            output=json.dumps({"assistant_response": turn["assistant_response"]}),
            duration_ns=turn_duration_ns,
        )
        print(f"Completed turn {turn_id}")

    # Conclude the session trace
    final_output = {
        "session_id": session_id,
        "turns_completed": len(conversation),
        "conversation_summary": "Completed multi-turn conversation about weather in different cities",
        "success": True,
    }

    logger.conclude(output=json.dumps(final_output))

    print("\nConcluded multi-turn conversation trace")
    print("Test completed successfully")

    logger.flush()


if __name__ == "__main__":
    test_galileo_logging()
