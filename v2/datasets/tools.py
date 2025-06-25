from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List
import json
import re
import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv("../.env")


MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are an expert AI system designer specializing in creating function definitions for AI tools. 
You excel at understanding domain-specific needs and translating them into well-structured JSONSchema function definitions."""

HUMAN_PROMPT = """## Task Description
Generate exactly 1 function definition for a customer-facing conversational AI chatbot in the {domain} domain. This tool must follow the JSONSchema format and enable the chatbot to perform domain-specific tasks and respond to customer queries effectively.

{existing_tools_section}

## Output Format Requirements
Your response must be a valid JSON object (NOT an array) representing exactly 1 function, with NO explanatory text before or after the JSON. Your entire response must be parseable with json.loads().

The object must follow this exact structure:
```json
{{
  "description": "Specific function purpose and information present in its response.",
  "properties": {{
    "parameter_name": {{
      "description": "Detailed description of the parameter ending with a period.",
      "type": "string|number|integer|boolean|array|object",
      "title": "Parameter_Name_Title_Case",
      "enum": ["option1", "option2", "option3"]
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
  "response_schema": {{ // minimum fields required for a response schema
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

## CRITICAL JSON FORMATTING RULES:
1. ALL property names must be in double quotes
2. ALL string values must be in double quotes
3. NO trailing commas after the last item in objects or arrays
4. NO comments in the JSON
5. Arrays and objects must be properly closed with matching brackets and braces
6. The output must be a single JSON object representing 1 function
7. Parameter titles must be in snake_case with underscores
8. Function titles (tool names) must be in snake_case
10. Array parameters must include the "items" field with the element type

## Function Design Guidelines:
1. Parameter Design:
   - Use clear, specific parameter names
   - Each parameter needs a clear description ending with a period
   - Use enums for fixed options when appropriate
   - Arrays must specify their item types
   - Parameter titles must be in Title_Case
   - Aim for a good mix of required and optional parameters whenever possible:
     * Include 2-4 required parameters for essential functionality
     * Include 1-3 optional parameters for enhanced features, filtering, or customization
     * Optional parameters should provide meaningful value without being strictly necessary
     * Consider parameters like filters, limits, sorting options, formatting preferences, or additional details as good candidates for optional parameters

2. Function Scope:
   - Each function should have one clear purpose
   - Functions should be specific to the {domain} domain and customer interactions
   - Group related functions under the same system/API name
   - Cover common customer tasks, inquiries, and service operations

3. Response Schema Design:
   - Keep response schemas simple and focused on essential information
   - Limit to 1-5 key fields maximum to avoid overly complex responses
   - Include only the most important data that would be immediately useful to customers
   - Avoid nested objects or arrays unless absolutely necessary

4. Categories to Include as Needed:
   - Customer information retrieval
   - Product/service information
   - Transactions
   - Status checking
   - User preferences
   - Support/escalation
   - Authentication/authorization
   - Account management
   - Order management
   - Payment processing
   - Shipping and delivery
   - FAQ and knowledge base
   - Appointment scheduling
   - Customer feedback
   - Service troubleshooting

Consider the typical customer needs, questions, pain points, terminology, and service expectations specific to the {domain} domain.

REMEMBER: Your response must be a single JSON object that can be parsed by json.loads(). Do not include any explanatory text or markdown formatting."""


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


def generate_tools(domain: str, num_tools: int) -> List[dict]:
    """
    Generate tool definitions for a specific domain using Claude 3.7 Sonnet.
    Tools are generated one by one to avoid duplicates and provide progress tracking.

    Args:
        domain (str): The domain for which to generate tools
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

    generated_tools = []
    
    # Generate tools one by one with progress bar
    with tqdm(total=num_tools, desc=f"Generating {domain} tools") as pbar:
        for i in range(num_tools):
            # Create existing tools section for the prompt
            if generated_tools:
                existing_tools_text = "## Previously Generated Tools\nDo NOT generate any of the following tools that have already been created:\n\n"
                for j, tool in enumerate(generated_tools, 1):
                    existing_tools_text += f"{j}. **{tool['title']}**: {tool['description']}\n"
                existing_tools_text += "\nMake sure your new tool is different from all the above tools in both name and functionality.\n"
            else:
                existing_tools_text = ""
            
            # Format the prompt with the domain and existing tools
            formatted_prompt = prompt_template.format_messages(
                domain=domain, 
                existing_tools_section=existing_tools_text
            )

            # Get the response from Claude
            response = chat.invoke(formatted_prompt, temperature=0.8, max_tokens=1500)

            try:
                # Clean and parse the JSON response
                json_str = clean_json_string(response.content)
                tool = json.loads(json_str)

                # Validate the response
                if not isinstance(tool, dict):
                    raise ValueError("Response is not a JSON object")

                # Validate the tool's schema
                if not validate_tool_schema(tool):
                    raise ValueError("Tool does not follow the required schema")

                # Check for duplicate tool names
                existing_titles = [t['title'] for t in generated_tools]
                if tool['title'] in existing_titles:
                    raise ValueError(f"Duplicate tool name: {tool['title']}")

                generated_tools.append(tool)
                pbar.set_postfix({'current_tool': tool['title']})
                pbar.update(1)

            except json.JSONDecodeError as e:
                print(f"\nWarning: Failed to parse JSON for tool {i+1}: {e}")
                print(f"Response content: {json_str}")
                # Continue to next iteration instead of failing completely
                pbar.set_postfix({'status': 'JSON parse error - retrying'})
                continue
            except Exception as e:
                print(f"\nWarning: Error processing tool {i+1}: {str(e)}")
                # Continue to next iteration instead of failing completely
                pbar.set_postfix({'status': 'Processing error - retrying'})
                continue

    if len(generated_tools) < num_tools:
        print(f"\nWarning: Only generated {len(generated_tools)} out of {num_tools} requested tools")
    
    return generated_tools


def save_tools(
    tools: List[dict], domain: str, output_dir: str, overwrite: bool = True
) -> str:
    """
    Save the generated tools to a JSON file in the specified output directory.

    Args:
        tools (List[dict]): List of tool definitions to save
        domain (str): Industry name used for the filename
        output_dir (str): Output directory for saving tools
        overwrite (bool): Whether to overwrite existing file if it exists

    Returns:
        str: Path to the saved file

    Raises:
        FileExistsError: If file exists and overwrite is False
    """
    file_path = os.path.join(output_dir, domain.lower(), "tools.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")
    elif os.path.exists(file_path):
        print(f"Warning: Overwriting existing file: {file_path}")

    with open(file_path, "w") as f:
        json.dump(tools, f, indent=2)
    return file_path


if __name__ == "__main__":
    # python tools.py --domain banking --num-tools 3 --overwrite
    parser = argparse.ArgumentParser(
        description="Generate tool definitions for a specific domain using Claude 3.7 Sonnet"
    )

    # Required arguments
    parser.add_argument(
        "--domain",
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
        default="../data",
        help="Output directory for saving tools (default: data)",
    )

    args = parser.parse_args()

    try:
        # Generate tools with specified parameters
        tools = generate_tools(args.domain, args.num_tools)

        # Save tools with specified parameters
        file_path = save_tools(tools, args.domain, args.output_dir, args.overwrite)

        print(
            f"Successfully generated and saved {args.num_tools} tools for {args.domain} domain"
        )
        print(f"Tools saved to: {file_path}")

    except FileExistsError as e:
        print(f"Error: {str(e)}")
        print("Use --overwrite to overwrite existing file")
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()

# python tools.py --domain banking --num-tools 20 --overwrite
