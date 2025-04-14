from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
import json
import re
import os
import glob
import argparse
from dotenv import load_dotenv

load_dotenv("../.env")

MODEL = "claude-3-7-sonnet-20250219"

SYSTEM_PROMPT = """You are an expert in creating realistic user personas for testing AI chatbot systems.
You excel at understanding how different types of people interact with AI assistants, their expectations, comfort levels, and behavioral patterns."""

HUMAN_PROMPT = """## Task Description
Generate {num_personas} detailed user personas representing different types of people who interact with AI chatbots. These personas should be domain-agnostic but will be used to test AI systems that have access to the following tools:

{tools_description}

Each persona should represent a distinct type of AI chatbot user, with specific interaction patterns, expectations, and behaviors.

## Output Format Requirements
Your response must be a valid JSON array containing exactly {num_personas} persona objects. Each object must follow this structure:
```json
{{
  "name": "Persona's full name",
  "age": integer,
  "occupation": "Current job or role",
  "background": "Brief background story",
  "personality_traits": ["trait1", "trait2", "trait3"],
  "ai_interaction": {{
    "comfort": "skeptical|cautious|comfortable|enthusiastic",
    "tech_proficiency": "low|medium|high",
    "learning_style": "experimental|methodical|cautious"
  }},
  "query_style": {{
    "verbosity": "terse|concise|detailed|verbose",
    "clarity": "vague|clear|precise"
  }},
  "error_response": {{
    "patience": "frustrated|patient|adaptive",
    "retry_willingness": "low|medium|high"
  }},
  "preferences": {{
    "tone": "formal|casual|friendly|professional",
    "detail_level": "brief|balanced|comprehensive",
    "format": ["text", "lists", "structured"]
  }}
}}
```

## Persona Design Guidelines:
1. Core Characteristics:
   - Create diverse personas across age, technical proficiency, and AI comfort levels
   - Include varied professional backgrounds and exposure to technology
   - Consider different learning styles and problem-solving approaches

2. AI Interaction Patterns:
   - Define how they typically phrase queries (verbosity, clarity)
   - Specify their response to AI mistakes or limitations
   - Consider their comfort level with AI systems

3. Behavioral Aspects:
   - How they handle errors and their patience level
   - Their willingness to retry after failures
   - Their preferred tone and detail level in communications

4. Ensure personas:
   - Are realistic and relatable
   - Represent different user archetypes
   - Have distinct interaction patterns
   - Include both positive and challenging aspects

Consider how different personality traits and experiences would affect their interaction with AI chatbots.

REMEMBER: Your response must be a single JSON array that can be parsed by json.loads(). Do not include any explanatory text or markdown formatting."""


def load_tools(tools_path: str) -> Dict[str, List[dict]]:
    """Load tool definitions from a single JSON file or directory."""
    tools = {}

    # If it's a file, load it directly
    if os.path.isfile(tools_path):
        industry = os.path.basename(tools_path).replace(".json", "")
        with open(tools_path, "r") as f:
            tools[industry] = json.load(f)
        return tools

    # If it's a directory, load all JSON files (existing behavior)
    for file_path in glob.glob(os.path.join(tools_path, "*.json")):
        industry = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r") as f:
            tools[industry] = json.load(f)
    return tools


def format_tools_description(tools: Dict[str, List[dict]]) -> str:
    """Format tools into a readable description for the prompt."""
    description = []
    for industry, tool_list in tools.items():
        description.append(f"\n{industry.upper()} TOOLS:")
        for tool in tool_list:
            description.append(f"- {tool['title']}: {tool['description']}")
    return "\n".join(description)


def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string before parsing."""
    # Remove any markdown code blocks
    json_str = re.sub(r"```json\s*|\s*```", "", json_str)

    # Remove any comments
    json_str = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.DOTALL)

    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    return json_str.strip()


def validate_persona_schema(persona: dict) -> bool:
    """Validate that a persona follows the required schema."""
    required_fields = {
        "name",
        "age",
        "occupation",
        "background",
        "personality_traits",
        "ai_interaction",
        "query_style",
        "error_response",
        "preferences",
    }

    if not all(field in persona for field in required_fields):
        return False

    if not isinstance(persona["personality_traits"], list):
        return False

    if not isinstance(persona["ai_interaction"], dict):
        return False

    if not isinstance(persona["query_style"], dict):
        return False

    if not isinstance(persona["error_response"], dict):
        return False

    if not isinstance(persona["preferences"], dict):
        return False

    # Validate AI interaction profile
    ai_profile = persona["ai_interaction"]
    if not all(
        field in ai_profile
        for field in ["comfort", "tech_proficiency", "learning_style"]
    ):
        return False

    return True


def generate_personas(tools: Dict[str, List[dict]], num_personas: int) -> List[dict]:
    """Generate persona definitions using Claude."""
    chat = ChatAnthropic(model=MODEL)

    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # Format tools description
    tools_description = format_tools_description(tools)

    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        tools_description=tools_description, num_personas=num_personas
    )

    # Get the response from Claude
    response = chat.invoke(formatted_prompt, temperature=0.7, max_tokens=4000)

    try:
        # Clean and parse the JSON response
        json_str = clean_json_string(response.content)
        personas = json.loads(json_str)

        # Validate the response
        if not isinstance(personas, list):
            raise ValueError("Response is not a JSON array")
        if len(personas) != num_personas:
            raise ValueError(f"Expected {num_personas} personas, got {len(personas)}")

        # Validate each persona's schema
        for i, persona in enumerate(personas):
            if not validate_persona_schema(persona):
                raise ValueError(
                    f"Persona at index {i} does not follow the required schema"
                )

        return personas
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON response: {e}\nResponse content: {json_str}"
        )
    except Exception as e:
        raise ValueError(f"Error processing response: {str(e)}")


def save_personas(
    personas: List[dict], output_dir: str, name: str, overwrite: bool = False
) -> str:
    """Save the generated personas to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{name}.json")

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")
    elif os.path.exists(file_path):
        print(f"Warning: Overwriting existing file: {file_path}")

    with open(file_path, "w") as f:
        json.dump(personas, f, indent=2)
    return file_path


if __name__ == "__main__":
    # python personas.py --name banking --tools-file ../data/tools/banking.json
    parser = argparse.ArgumentParser(
        description="Generate user personas for testing AI tools using Claude"
    )

    # Required arguments
    parser.add_argument(
        "--file-name",
        type=str,
        required=True,
        help="Name for the persona set (used in output filename)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num-personas",
        type=int,
        default=3,
        help="Number of personas to generate (default: 3)",
    )

    # Add tools file argument group
    tools_group = parser.add_mutually_exclusive_group()
    tools_group.add_argument(
        "--tools-file",
        type=str,
        help="Single JSON file containing tool definitions",
    )
    tools_group.add_argument(
        "--tools-dir",
        type=str,
        default="data/tools",
        help="Directory containing tool definition JSON files (default: data/tools)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/personas",
        help="Output directory for saving personas (default: data/personas)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file if it exists"
    )

    args = parser.parse_args()

    try:
        # Load existing tools from either file or directory
        tools_path = args.tools_file if args.tools_file else args.tools_dir
        tools = load_tools(tools_path)
        if not tools:
            raise ValueError(f"No tool definitions found in {tools_path}")

        # Generate personas
        personas = generate_personas(tools, args.num_personas)

        # Save personas
        file_path = save_personas(personas, args.output_dir, args.name, args.overwrite)

        print(f"Successfully generated and saved {args.num_personas} personas")
        print(f"Personas saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()

# python personas.py --file-name banking.json --num-personas 20 --tools-file ../data/tools/banking.json --overwrite
