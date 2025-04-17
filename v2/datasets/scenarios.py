from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Literal, Union, Optional
from pydantic import BaseModel, Field, field_validator
import json
import argparse
import os
import random
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("../.env")

MODEL = "claude-3-7-sonnet-20250219"
temperature = 1.0

SYSTEM_PROMPT = """You are an expert in creating complex and challenging scenarios for testing advanced AI chatbot systems.
You excel at designing intricate situations that test the limits of AI assistants' reasoning, planning, and problem-solving capabilities.
Most importantly, you are meticulous about generating valid JSON output that strictly follows the specified format without any special characters or formatting that could break JSON parsing."""

HUMAN_PROMPT = """## Task Description
Generate EXACTLY {num_scenarios} diverse, complex and challenging chat scenarios for testing an AI assistant with the following tools and persona.
CRITICAL: You MUST generate EXACTLY {num_scenarios} scenarios - no more, no less.

TOOLS:
{tools_description}

PERSONA:
{persona_description}

DOMAIN:
{domain}

CATEGORIES:
{categories_instruction}

## Goal Requirements
Each scenario must have 3-5 goals that are:
1. Specific: Clearly state what needs to be accomplished
2. Measurable: Have clear success criteria
3. User-centric: Focus on user outcomes, not assistant actions
4. Concrete: Avoid vague or abstract objectives
5. Independent: Each goal should be distinct
6. Testable: Can be verified as completed or not

## Output Format Requirements
CRITICAL: Generate ONLY a valid JSON array containing EXACTLY {num_scenarios} scenario objects. No other text or formatting.

Each scenario object must follow this exact structure:

{{
    "persona_index": {persona_index},
    "category": "{category}",
    "first_message": "single line message with no special chars",
    "goals": ["goal1", "goal2"]
}}

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
1. Create scenarios which leverage multiple tools
2. Create scenarios which have 3-5 goals that represent what the user wants to accomplish
3. Goals should be specific, measurable user objectives (not assistant tasks)
4. No file paths or code snippets

## Additional Requirements:
1. The output MUST be a JSON array with EXACTLY {num_scenarios} elements
2. Each element MUST be a complete scenario object  

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


# Define available categories with descriptions
CATEGORIES_DICT = {
    "tool_coordination": "Complex orchestration requiring careful planning and parallel tool execution",
    "out_of_scope_handling": "Nuanced requests mixing supported and unsupported features",
    "adaptation": "Challenging situations requiring cascading dependencies, conditional logic, creative tool combinations and validations",
    "unhappy_customer": "Multi-layered problems with both technical and emotional complexity",
    "manipulative_customer": "Sophisticated attempts to exploit system limitations or policies",
}


class Scenario(BaseModel):
    persona_index: int
    category: Literal[
        "tool_coordination",
        "out_of_scope_handling",
        "adaptation",
        "unhappy_customer",
        "manipulative_customer",
    ]
    first_message: str
    goals: List[str] = Field(min_length=2, max_length=5)

    @field_validator("first_message")
    def validate_first_message(cls, v):
        if not isinstance(v, str):
            raise ValueError("First message must be a string")
        if len(v.strip()) < 3:
            raise ValueError("Message content too short")
        if "\n" in v:
            raise ValueError("Message cannot contain newlines")
        return v


def load_json_file(file_path: str) -> dict:
    """Load and parse a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_domain_file_paths(domain: str) -> tuple:
    """Construct file paths for personas and tools based on domain."""
    personas_file = f"../data/personas/{domain}.json"
    tools_file = f"../data/tools/{domain}.json"
    return personas_file, tools_file


def format_tools_description(tools: List[dict]) -> str:
    """Format tools into a readable description for the prompt."""
    description = []
    for tool in tools:
        description.append(f"- {tool['title']}: {tool['description']}")
    return "\n".join(description)


def format_persona_description(persona: dict) -> str:
    """Format persona into a readable description for the prompt."""
    return json.dumps(persona, indent=2)


def format_categories_instruction(category: str) -> str:
    """Format categories instruction for a specific category."""
    return (
        f"You MUST generate scenarios ONLY for the category: {category}\n\nCategory definitions:\n"
        + "\n".join([f"- {cat}: {desc}" for cat, desc in CATEGORIES_DICT.items()])
    )


def validate_scenario(scenario: dict, category: str) -> bool:
    """Validate a generated scenario."""
    try:
        # Use Pydantic validation
        validated_scenario = Scenario(**scenario)
        # Ensure category matches
        if validated_scenario.category != category:
            raise ValueError(
                f"Scenario category '{validated_scenario.category}' does not match requested category '{category}'"
            )
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False


def generate_scenarios(
    tools: List[dict],
    persona: dict,
    persona_index: int,
    num_scenarios: int,
    domain: str,
    category: str,
) -> List[dict]:
    """Generate scenario definitions using Claude."""
    chat = ChatAnthropic(model=MODEL)

    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # Randomly shuffle the tools to increase variety in generated scenarios
    shuffled_tools = tools.copy()
    random.shuffle(shuffled_tools)

    # Format descriptions
    tools_description = format_tools_description(shuffled_tools)
    persona_description = format_persona_description(persona)
    categories_instruction = format_categories_instruction(category)

    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        tools_description=tools_description,
        persona_description=persona_description,
        persona_index=persona_index,
        num_scenarios=num_scenarios,
        categories_instruction=categories_instruction,
        category=category,
        domain=domain,
    )

    # Get the response from Claude
    response = chat.invoke(formatted_prompt, temperature=temperature)

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

            # Check if category is the requested category
            if scenario_data.get("category") != category:
                raise ValueError(
                    f"Generated scenario has category {scenario_data.get('category')} but should be {category}"
                )

            # Validate using Pydantic model
            scenario = Scenario(**scenario_data)
            validated_scenarios.append(scenario.model_dump())

        return validated_scenarios
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise ValueError(f"Error processing response: {str(e)}")


def save_scenarios(
    scenarios: List[dict], domain: str, category: str, overwrite: bool = False
) -> str:
    """Save the generated scenarios to a JSON file."""
    output_dir = os.path.join("../data/scenarios", domain)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{category}.json")

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")

    with open(file_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    return file_path


if __name__ == "__main__":
    # python scenarios.py --domain banking --category tool_coordination --overwrite
    parser = argparse.ArgumentParser(
        description="Generate chat scenarios for testing AI tools using Claude"
    )

    # Required arguments
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain for the scenarios (e.g., banking, healthcare)",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        choices=list(CATEGORIES_DICT.keys()),
        help="Category to generate scenarios for (if not specified, all categories will be generated)",
    )

    # Optional arguments
    parser.add_argument(
        "--scenarios-per-persona",
        type=int,
        default=2,
        help="Number of scenarios to generate per persona (default: 2)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file if it exists"
    )

    args = parser.parse_args()

    try:
        # Automatically construct file paths based on domain
        personas_file, tools_file = get_domain_file_paths(args.domain)

        # Load personas and tools
        personas = load_json_file(personas_file)
        tools = load_json_file(tools_file)

        # Verify expected counts
        assert len(personas) == 25, f"Expected 25 personas, but found {len(personas)}"
        assert len(tools) == 20, f"Expected 20 tools, but found {len(tools)}"

        # Determine which categories to generate
        categories_to_generate = (
            [args.category] if args.category else list(CATEGORIES_DICT.keys())
        )

        # Generate scenarios for each category
        for category in tqdm(categories_to_generate, desc="Categories"):
            # Initialize an empty list for all scenarios in this category
            all_scenarios = []

            # Generate scenarios for each persona with progress bar
            for idx, persona in tqdm(
                enumerate(personas), desc="Personas", total=len(personas)
            ):
                # Generate scenarios for the current persona
                scenarios = generate_scenarios(
                    tools=tools,
                    persona=persona,
                    persona_index=idx,
                    num_scenarios=args.scenarios_per_persona,
                    domain=args.domain,
                    category=category,
                )
                # Append the generated scenarios to the all_scenarios list
                all_scenarios.extend(scenarios)

            # Save scenarios
            file_path = save_scenarios(
                all_scenarios, args.domain, category, args.overwrite
            )

            print(
                f"Successfully generated {len(all_scenarios)} scenarios for category '{category}'"
            )
            print(f"Scenarios saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()
