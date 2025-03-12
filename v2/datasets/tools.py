from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List
import json
import re
import os
import argparse
from dotenv import load_dotenv

load_dotenv("../../.env")

MODEL = "claude-3-7-sonnet-20250219"

SYSTEM_PROMPT = """You are an expert AI system designer specializing in creating function definitions for AI tools. 
You excel at understanding industry-specific needs and translating them into well-structured JSONSchema function definitions."""

HUMAN_PROMPT = """## Task Description
Generate exactly {num_tools} function definitions for a conversational AI chatbot in the {industry} industry. These tools must follow the JSONSchema format and enable the chatbot to perform domain-specific tasks and respond to user queries effectively.

## Output Format Requirements
Your response must be a valid JSON array containing exactly {num_tools} objects. Each object must follow this exact structure:
```json
{{
  "description": "Specific function purpose.",
  "properties": {{
    "parameter_name": {{
      "description": "Detailed description of the parameter ending with a period.",
      "type": "string|number|integer|boolean|array|object",
      "title": "Parameter_Name_Title_Case"
    }},
    "array_parameter": {{
      "description": "Description for array parameter.",
      "type": "array",
      "items": {{
        "type": "string|number|integer|boolean"
      }},
      "title": "Array_Parameter_Title_Case"
    }}
  }},
  "required": ["list", "of", "required", "parameter", "names"],
  "title": "function_name_in_snake_case",
  "type": "object",
  "response_schema": {{
    "description": "Description of the response format.",
    "type": "object",
    "properties": {{
      "field_name": {{
        "description": "Description of the response field ending with a period.",
        "type": "string|number|integer|boolean|array|object"
      }}
    }},
    "required": ["list", "of", "required", "response", "fields"]
  }}
}}
```

IMPORTANT JSON FORMATTING RULES:
1. ALL property names must be in double quotes
2. ALL string values must be in double quotes
3. NO trailing commas after the last item in objects or arrays
4. NO comments in the JSON
5. Arrays and objects must be properly closed
6. The output must be a single JSON array containing {num_tools} function objects
7. Parameter titles must be in snake_case with underscores
8. Function titles (tool names) must be in snake_case
9. Parameter descriptions must end with a period
10. Array parameters must include the "items" field with the element type
11. Each tool must include a response_schema defining the expected response format

## Function Design Guidelines:
1. Parameter Design:
   - Use clear, specific parameter names
   - Each parameter needs a clear description ending with a period
   - Use enums for fixed options when appropriate
   - Arrays must specify their item types
   - Parameter titles must be in Title_Case

2. Function Scope:
   - Each function should have one clear purpose
   - Functions should be specific to the {industry} industry
   - Group related functions under the same system/API name
   - Cover common user tasks and operations

3. Categories to Include as Needed:
   - Information retrieval
   - Data manipulation
   - Transactions
   - Status checking
   - User preferences
   - Support/escalation
   - Authentication/authorization
   - Reporting/analytics
   - Business operations
   - Customer support
   - Account management
   - Inventory management
   - Order management
   - Payment processing
   - Shipping and delivery

Consider the typical operations, pain points, terminology, and regulatory requirements specific to the {industry} industry.

REMEMBER: Your response must be a single JSON array that can be parsed by json.loads(). Do not include any explanatory text or markdown formatting."""


def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string before parsing."""
    # Remove any markdown code blocks
    json_str = re.sub(r"```json\s*|\s*```", "", json_str)

    # Remove any comments
    json_str = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.DOTALL)

    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    return json_str.strip()


def validate_tool_schema(tool: dict) -> bool:
    """Validate that a tool follows the required schema."""
    required_fields = {
        "description",
        "properties",
        "required",
        "title",
        "type",
        "response_schema",
    }
    if not all(field in tool for field in required_fields):
        return False

    if not isinstance(tool["properties"], dict):
        return False

    if not isinstance(tool["required"], list):
        return False

    if tool["type"] != "object":
        return False

    # Validate response schema
    response_schema = tool.get("response_schema")
    if not isinstance(response_schema, dict):
        return False
    if not all(
        field in response_schema
        for field in ["description", "type", "properties", "required"]
    ):
        return False
    if not isinstance(response_schema["properties"], dict):
        return False
    if not isinstance(response_schema["required"], list):
        return False

    # Validate each parameter
    for param_name, param_info in tool["properties"].items():
        if not all(field in param_info for field in ["description", "type", "title"]):
            return False
        if not param_info["description"].endswith("."):
            return False
        if param_info["type"] == "array" and "items" not in param_info:
            return False

    return True


def generate_tools(industry: str, num_tools: int) -> List[dict]:
    """
    Generate tool definitions for a specific industry using Claude 3.7 Sonnet.

    Args:
        industry (str): The industry for which to generate tools
        num_tools (int): Number of tools to generate

    Returns:
        List[dict]: List of tool definitions in JSONSchema format
    """
    # Initialize the ChatAnthropic client with Claude 3.7 Sonnet
    chat = ChatAnthropic(model=MODEL)

    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    # Format the prompt with the industry and number of tools
    formatted_prompt = prompt_template.format_messages(
        industry=industry, num_tools=num_tools
    )

    # Get the response from Claude
    response = chat.invoke(formatted_prompt, temperature=0.3, max_tokens=4000)

    try:
        # Clean and parse the JSON response
        json_str = clean_json_string(response.content)
        tools = json.loads(json_str)

        # Validate the response
        if not isinstance(tools, list):
            raise ValueError("Response is not a JSON array")
        if len(tools) != num_tools:
            raise ValueError(f"Expected {num_tools} tools, got {len(tools)}")

        # Validate each tool's schema
        for i, tool in enumerate(tools):
            if not validate_tool_schema(tool):
                raise ValueError(
                    f"Tool at index {i} does not follow the required schema"
                )

        return tools
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON response: {e}\nResponse content: {json_str}"
        )
    except Exception as e:
        raise ValueError(f"Error processing response: {str(e)}")


def save_tools(
    tools: List[dict], industry: str, output_dir: str, overwrite: bool = True
) -> str:
    """
    Save the generated tools to a JSON file in the specified output directory.

    Args:
        tools (List[dict]): List of tool definitions to save
        industry (str): Industry name used for the filename
        output_dir (str): Output directory for saving tools
        overwrite (bool): Whether to overwrite existing file if it exists

    Returns:
        str: Path to the saved file

    Raises:
        FileExistsError: If file exists and overwrite is False
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{industry.lower()}.json")

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")
    elif os.path.exists(file_path):
        print(f"Warning: Overwriting existing file: {file_path}")

    with open(file_path, "w") as f:
        json.dump(tools, f, indent=2)
    return file_path


if __name__ == "__main__":
    # python tools.py --industry banking --num-tools 3 --overwrite
    parser = argparse.ArgumentParser(
        description="Generate tool definitions for a specific industry using Claude 3.7 Sonnet"
    )

    # Required arguments
    parser.add_argument(
        "--industry",
        type=str,
        required=True,
        help="Industry for which to generate tools (e.g., banking, healthcare)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num-tools",
        type=int,
        default=5,
        help="Number of tools to generate (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing file if it exists (default: False)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/tools",
        help="Output directory for saving tools (default: data/tools)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for Claude's response generation (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for Claude's response (default: 4000)",
    )

    args = parser.parse_args()

    try:
        # Generate tools with specified parameters
        tools = generate_tools(args.industry, args.num_tools)

        # Save tools with specified parameters
        file_path = save_tools(tools, args.industry, args.output_dir, args.overwrite)

        print(
            f"Successfully generated and saved {args.num_tools} tools for {args.industry} industry"
        )
        print(f"Tools saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()
