from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
import json
import re
import os
import glob
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError

load_dotenv("../.env")

MODEL = "claude-3-7-sonnet-20250219"


class Persona(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=18, le=100)
    occupation: str = Field(..., min_length=5)
    personality_traits: List[str] = Field(..., min_items=1, max_items=3)
    tone: str = Field(..., pattern="^(formal|casual|friendly|professional|unprofessional)$")
    detail_level: str = Field(..., pattern="^(brief|balanced|comprehensive)$")


SYSTEM_PROMPT = """You are an expert in creating realistic user personas for testing AI chatbot systems.
You excel at understanding how different types of people interact with AI assistants, their expectations, comfort levels, and behavioral patterns."""

HUMAN_PROMPT = """## Task Description
Generate 1 detailed user persona representing a type of person who interacts with AI chatbots. This persona should be domain-agnostic but will be used to test AI systems that have access to the following tools:

{tools_description}

The persona should represent a distinct type of AI chatbot user, with specific interaction patterns, expectations, and behaviors.

{existing_personas_section}

## Output Format Requirements
Your response must be a valid JSON object (not an array) following this EXACT schema:

{schema}

## Persona Design Guidelines:
   - Create a diverse persona across age, technical proficiency, and AI comfort levels
   - Include varied professional backgrounds and exposure to technology
   - Consider different learning styles and problem-solving approaches
   - Make it realistic and relatable
   - Represent a specific user archetype
   - Have distinct interaction patterns
   - Include both positive and challenging aspects
   - IMPORTANT: Ensure this persona is distinctly different from any previously generated personas

Consider how different personality traits and experiences would affect their interaction with AI chatbots.

REMEMBER: Your response must be a single JSON object that exactly matches the schema above and can be parsed by json.loads(). Do not include any explanatory text or markdown formatting."""


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


def format_existing_personas_section(existing_personas: List[dict]) -> str:
    """Format existing personas into a readable section for the prompt."""
    if not existing_personas:
        return ""
    
    section = ["## Previously Generated Personas"]
    section.append("To ensure diversity, here are the personas that have already been generated:")
    section.append("")
    
    for i, persona in enumerate(existing_personas, 1):
        section.append(f"**Persona {i}:** {persona['name']}")
        section.append(f"- Age: {persona['age']}, Occupation: {persona['occupation']}")
        section.append(f"- Personality: {', '.join(persona['personality_traits'])}")
        section.append("")
    
    section.append("**CRITICAL:** Generate a persona that is significantly different from all the above personas in terms of:")
    section.append("- Age group and life stage")
    section.append("- Professional background and industry")
    section.append("")
    
    return "\n".join(section)


def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string before parsing."""
    # Remove any markdown code blocks
    json_str = re.sub(r"```json\s*|\s*```", "", json_str)

    # Remove any comments
    json_str = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.DOTALL)

    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    return json_str.strip()


def generate_personas(tools: Dict[str, List[dict]], num_personas: int, n_persona_per_iter: int = 5) -> List[dict]:
    """Generate persona definitions using Claude with progress tracking and Pydantic validation."""
    chat = ChatAnthropic(model=MODEL)

    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # Format tools description
    tools_description = format_tools_description(tools)
    
    # Generate schema from Pydantic model
    schema = Persona.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    personas = []
    
    # Calculate number of batches needed
    num_batches = (num_personas + n_persona_per_iter - 1) // n_persona_per_iter
    
    # Generate personas in batches with progress bar
    for batch_idx in tqdm(range(num_batches), desc="Generating persona batches", unit="batch"):
        # Calculate how many personas to generate in this batch
        start_idx = batch_idx * n_persona_per_iter
        end_idx = min(start_idx + n_persona_per_iter, num_personas)
        personas_in_batch = end_idx - start_idx
        
        tqdm.write(f"Batch {batch_idx + 1}/{num_batches}: Generating {personas_in_batch} personas...")
        
        # Generate each persona in the current batch
        for i in range(personas_in_batch):
            current_persona_idx = start_idx + i
            max_retries = 10
            retry_count = 0
            persona_generated = False
            
            # Format existing personas section for this iteration
            existing_personas_section = format_existing_personas_section(personas)
            
            while not persona_generated and retry_count < max_retries:
                try:
                    # Format the prompt with existing personas
                    formatted_prompt = prompt_template.format_messages(
                        tools_description=tools_description,
                        existing_personas_section=existing_personas_section,
                        schema=schema_str
                    )

                    # Get the response from Claude
                    response = chat.invoke(formatted_prompt, temperature=1.0, max_tokens=4000)

                    # Clean and parse the JSON response
                    json_str = clean_json_string(response.content)
                    persona_data = json.loads(json_str)

                    # Validate using Pydantic model
                    persona = Persona(**persona_data)
                    
                    # Convert back to dict for consistency with existing code
                    personas.append(persona.model_dump())
                    persona_generated = True
                    
                except json.JSONDecodeError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        tqdm.write(f"    Retrying persona {current_persona_idx+1} (attempt {retry_count + 1}/{max_retries}): JSON parsing failed - {str(e)}")
                    else:
                        raise ValueError(f"Failed to parse JSON for persona {current_persona_idx+1} after {max_retries} attempts: {str(e)}")
                        
                except ValidationError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        tqdm.write(f"    Retrying persona {current_persona_idx+1} (attempt {retry_count + 1}/{max_retries}): Schema validation failed - {str(e)}")
                    else:
                        raise ValueError(f"Failed to generate valid persona {current_persona_idx+1} after {max_retries} attempts: {str(e)}")
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        tqdm.write(f"    Retrying persona {current_persona_idx+1} (attempt {retry_count + 1}/{max_retries}): Unexpected error - {str(e)}")
                    else:
                        raise ValueError(f"Unexpected error generating persona {current_persona_idx+1} after {max_retries} attempts: {str(e)}")
        
        tqdm.write(f"  Completed batch {batch_idx + 1}: Generated {personas_in_batch} personas (Total: {len(personas)}/{num_personas})")

    return personas


def save_personas(
    personas: List[dict], domain: str, overwrite: bool = False
) -> str:
    """Save the generated personas to a JSON file."""
    # Create domain directory if it doesn't exist
    domain_dir = os.path.join("../data", domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    file_path = os.path.join(domain_dir, "personas.json")

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")
    elif os.path.exists(file_path):
        print(f"Warning: Overwriting existing file: {file_path}")

    with open(file_path, "w") as f:
        json.dump(personas, f, indent=2)
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate user personas for testing AI tools using Claude"
    )

    # Required arguments
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name (used for tools path and output filename)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num-personas",
        type=int,
        default=3,
        help="Number of personas to generate (default: 3)",
    )

    parser.add_argument(
        "--n-persona-per-iter",
        type=int,
        default=5,
        help="Number of personas to generate per iteration/batch (default: 5)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file if it exists"
    )

    args = parser.parse_args()

    try:
        # Load tools from data/{domain}/tools.json
        tools_path = os.path.join("../data", args.domain, "tools.json")
        tools = load_tools(tools_path)
        if not tools:
            raise ValueError(f"No tool definitions found in {tools_path}")

        # Generate personas
        personas = generate_personas(tools, args.num_personas, args.n_persona_per_iter)

        # Save personas
        file_path = save_personas(personas, args.domain, args.overwrite)

        print(f"Successfully generated and saved {args.num_personas} personas")
        print(f"Personas saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()

# python personas.py --domain banking --num-personas 5 --overwrite
