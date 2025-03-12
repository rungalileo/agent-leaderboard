from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
import json
import argparse
import os
from dotenv import load_dotenv

load_dotenv("../../.env")

MODEL = "claude-3-7-sonnet-20250219"

SYSTEM_PROMPT = """You are an expert in creating complex and challenging scenarios for testing advanced AI chatbot systems.
You excel at designing intricate situations that test the limits of AI assistants' reasoning, planning, and problem-solving capabilities.
Most importantly, you are meticulous about generating valid JSON output that strictly follows the specified format without any special characters or formatting that could break JSON parsing."""

HUMAN_PROMPT = """## Task Description
Generate EXACTLY {num_scenarios} complex and challenging chat scenarios for testing an AI assistant with the following tools and persona.
CRITICAL: You MUST generate EXACTLY {num_scenarios} scenarios - no more, no less.

TOOLS:
{tools_description}

PERSONA:
{persona_description}

CATEGORIES:
- tool_coordination: Complex orchestration requiring careful planning and parallel tool execution
- out_of_scope_handling: Nuanced requests mixing supported and unsupported features
- context_retention: Complex information threading across many turns with interdependencies
- adaptation: Challenging situations requiring creative tool combinations and workarounds
- unhappy_customer: Multi-layered problems with both technical and emotional complexity
- manipulative_customer: Sophisticated attempts to exploit system limitations or policies

## Goal Requirements
Each scenario must have 2-5 goals that are:
1. Specific: Clearly state what needs to be accomplished
2. Measurable: Have clear success criteria
3. User-centric: Focus on user outcomes, not assistant actions
4. Concrete: Avoid vague or abstract objectives
5. Independent: Each goal should be distinct
6. Testable: Can be verified as completed or not

Examples of good goals:
- "Create a new project with 3 specific Python files"
- "Debug and fix the TypeError in the login function"
- "Add input validation for all form fields"
- "Implement a dark mode toggle with persistent settings"

Examples of bad goals:
- "Make the code better" (too vague)
- "Help the user" (not specific)
- "Use the tools" (assistant-focused)
- "Write some code" (not measurable)

## Output Format Requirements
CRITICAL: Generate ONLY a valid JSON array containing EXACTLY {num_scenarios} scenario objects. No other text or formatting.

Each scenario object must follow this exact structure:

For regular scenarios:
{{
    "persona_index": {persona_index},
    "category": "category_name",
    "first_message": "single line message with no special chars",
    "goals": ["goal1", "goal2"]
}}

For context retention scenarios:
{{
    "persona_index": {persona_index},
    "category": "context_retention",
    "first_message": [
        {{"role": "user", "content": "First user message"}},
        {{"role": "assistant", "content": "First assistant response"}},
        {{"role": "user", "content": "Second user message"}},
        ... more messages following the pattern but last message must be a user message...
    ],
    "goals": ["goal1", "goal2"]
}}

## STRICT CONTEXT RETENTION REQUIREMENTS:
1. MUST contain EXACTLY 11, 13, or 15 messages (no other numbers allowed)
2. MUST start with user message
3. MUST end with user message
4. Messages MUST strictly alternate: user -> assistant -> user -> assistant -> user
5. Each message object MUST have exactly two fields: "role" and "content"
6. Message content MUST be a single line (no line breaks)
7. Each message should build on previous context
8. Later messages should reference information from earlier messages
9. Include specific details that need to be remembered and reused

## STRICT JSON FORMATTING RULES:
1. Use ONLY these ASCII characters:
   - a-z, A-Z, 0-9
   - Basic punctuation: . , ? ! ( ) [ ] {{ }}
   - Quotes: " (double quotes only)
   - Whitespace: space
   - No tabs, no newlines within strings

2. String Content Rules:
   - NO emojis
   - NO unicode characters
   - NO control characters
   - NO invisible characters
   - NO HTML or markdown
   - NO line breaks in strings
   - Messages must be single-line only

3. Escaping Rules:
   - Escape double quotes with \\"
   - Escape backslashes with \\\\
   - Do not use single quotes

4. Structure Rules:
   - No trailing commas
   - No comments
   - No extra whitespace
   - Array elements separated by single comma
   - Property names must be double-quoted

## Content Guidelines:
1. Keep messages simple and clear
2. Use only basic punctuation
3. 2-5 goals per scenario that represent what the user wants to accomplish
4. Goals should be specific, measurable user objectives (not assistant tasks)
5. Avoid complex technical terms
6. No file paths or code snippets

## Additional Requirements:
1. The output MUST be a JSON array with EXACTLY {num_scenarios} elements
2. Each element MUST be a complete scenario object
3. Do not generate any extra scenarios
4. Verify the number of scenarios before completing the response

Remember: Output MUST be parseable by Python's json.loads() and contain EXACTLY the requested number of scenarios."""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    def validate_content(cls, v):
        if "\n" in v:
            raise ValueError("Message content cannot contain newlines")
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Message content must be at least 3 characters")
        return v.strip()


class Scenario(BaseModel):
    persona_index: int
    category: Literal[
        "tool_coordination",
        "out_of_scope_handling",
        "context_retention",
        "adaptation",
        "unhappy_customer",
        "manipulative_customer",
    ]
    first_message: Union[str, List[ChatMessage]]
    goals: List[str] = Field(min_length=2, max_length=5)

    @field_validator("first_message")
    def validate_first_message(cls, v, values):
        if values.data.get("category") == "context_retention":
            if not isinstance(v, list):
                raise ValueError(
                    "Context retention scenarios must have a list of messages"
                )

            # Validate number of messages
            valid_lengths = {11, 13, 15}
            if len(v) not in valid_lengths:
                raise ValueError(
                    f"Context retention scenarios must have exactly 11, 13, or 15 messages. Got {len(v)}"
                )

            # Validate message pattern
            if v[0].role != "user" or v[-1].role != "user":
                raise ValueError("First and last messages must be from user")

            # Check alternating pattern
            for i, msg in enumerate(v):
                expected_role = "user" if i % 2 == 0 else "assistant"
                if msg.role != expected_role:
                    raise ValueError(
                        f"Message {i} should have role {expected_role}, found {msg.role}"
                    )

                # Validate message content
                if len(msg.content.strip()) < 3:
                    raise ValueError(f"Message {i} content too short: {msg.content}")
                if "\n" in msg.content:
                    raise ValueError(f"Message {i} contains newlines")
        else:
            if not isinstance(v, str):
                raise ValueError(
                    "Non-context retention scenarios must have a string message"
                )
            if len(v.strip()) < 3:
                raise ValueError("Message content too short")
            if "\n" in v:
                raise ValueError("Message cannot contain newlines")
        return v


def load_json_file(file_path: str) -> dict:
    """Load and parse a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_tools_description(tools: List[dict]) -> str:
    """Format tools into a readable description for the prompt."""
    description = []
    for tool in tools:
        description.append(f"- {tool['title']}: {tool['description']}")
    return "\n".join(description)


def format_persona_description(persona: dict) -> str:
    """Format persona into a readable description for the prompt."""
    return json.dumps(persona, indent=2)


def validate_scenario(scenario: dict, num_scenarios: int) -> bool:
    """Validate a generated scenario."""
    try:
        # Use Pydantic validation
        validated_scenario = Scenario(**scenario)
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False


def generate_scenarios(
    tools: List[dict], persona: dict, persona_index: int, num_scenarios: int
) -> List[dict]:
    """Generate scenario definitions using Claude."""
    chat = ChatAnthropic(model=MODEL)

    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # Format descriptions
    tools_description = format_tools_description(tools)
    persona_description = format_persona_description(persona)

    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        tools_description=tools_description,
        persona_description=persona_description,
        persona_index=persona_index,
        num_scenarios=num_scenarios,
    )

    # Get the response from Claude
    response = chat.invoke(formatted_prompt, temperature=0.7)

    try:
        # Parse the JSON response
        scenarios_data = json.loads(response.content)

        # Validate the response
        if not isinstance(scenarios_data, list):
            raise ValueError("Response is not a JSON array")
        if len(scenarios_data) != num_scenarios:
            raise ValueError(
                f"Expected {num_scenarios} scenarios, got {len(scenarios_data)}"
            )

        # Validate each scenario using Pydantic
        validated_scenarios = []
        for scenario_data in scenarios_data:
            # Ensure persona_index is correct
            scenario_data["persona_index"] = persona_index
            # Validate using Pydantic model
            scenario = Scenario(**scenario_data)
            validated_scenarios.append(scenario.model_dump())

        return validated_scenarios
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise ValueError(f"Error processing response: {str(e)}")


def save_scenarios(
    scenarios: List[dict], output_dir: str, name: str, overwrite: bool = False
) -> str:
    """Save the generated scenarios to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{name}")

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")

    with open(file_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    return file_path


if __name__ == "__main__":
    # python scenarios.py --name banking.json --personas-file ../data/personas/banking.json --tools-file ../data/tools/banking.json --overwrite
    parser = argparse.ArgumentParser(
        description="Generate chat scenarios for testing AI tools using Claude"
    )

    # Required arguments
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the scenario set (used in output filename)",
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        required=True,
        help="JSON file containing persona definitions",
    )
    parser.add_argument(
        "--tools-file",
        type=str,
        required=True,
        help="JSON file containing tool definitions",
    )

    # Optional arguments
    parser.add_argument(
        "--scenarios-per-persona",
        type=int,
        default=2,
        help="Number of scenarios to generate per persona (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/scenarios",
        help="Output directory for saving scenarios (default: data/scenarios)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file if it exists"
    )

    args = parser.parse_args()

    try:
        # Load personas and tools
        personas = load_json_file(args.personas_file)
        tools = load_json_file(args.tools_file)

        # Generate scenarios for each persona
        all_scenarios = []
        for idx, persona in enumerate(personas):
            scenarios = generate_scenarios(
                tools=tools,
                persona=persona,
                persona_index=idx,
                num_scenarios=args.scenarios_per_persona,
            )
            all_scenarios.extend(scenarios)

        # Save scenarios
        file_path = save_scenarios(
            all_scenarios, args.output_dir, args.name, args.overwrite
        )

        print(f"Successfully generated {len(all_scenarios)} scenarios")
        print(f"Scenarios saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()
