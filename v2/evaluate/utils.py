import time
import json
import re
import logging
import os
from colorama import init, Fore, Style, Back
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any


# Initialize colorama
init(autoreset=True)

# Check if terminal supports Unicode
UNICODE_SUPPORT = True
try:
    # Try detecting terminal capability
    term = os.environ.get("TERM", "")
    if "xterm" in term or "rxvt" in term or "vt100" in term:
        UNICODE_SUPPORT = True
    # Force ASCII mode for specific environments
    if os.environ.get("FORCE_ASCII_LOGS", "false").lower() in ("true", "1", "yes"):
        UNICODE_SUPPORT = False
except Exception:
    UNICODE_SUPPORT = False

# Box drawing characters - with ASCII fallback
if UNICODE_SUPPORT:
    BOX = {
        "tl": "┌",
        "tr": "┐",
        "bl": "└",
        "br": "┘",
        "h": "━",
        "v": "┃",
        "ltee": "├",
        "rtee": "┤",
    }
else:
    BOX = {
        "tl": "+",
        "tr": "+",
        "bl": "+",
        "br": "+",
        "h": "-",
        "v": "|",
        "ltee": "+",
        "rtee": "+",
    }


class ConversationHistoryManager:
    """Manages conversation history and related operations for agent simulations."""

    @staticmethod
    def format_for_display(history: List[Dict[str, str]]) -> str:
        """Format conversation history for display or logging."""
        formatted = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            formatted += f"{role.upper()}: {content}\n\n"
        return formatted

    @staticmethod
    def add_message(
        history: List[Dict[str, str]], role: str, content: str
    ) -> List[Dict[str, str]]:
        """Add a message to the conversation history."""
        history.append({"role": role, "content": content})
        return history

    @staticmethod
    def get_last_user_message(
        history: List[Dict[str, str]], default_message: str = ""
    ) -> str:
        """Get the most recent user message from the history."""
        for msg in reversed(history):
            if msg["role"] == "user":
                return msg["content"]
        return default_message

    @staticmethod
    def to_langchain_messages(
        history: List[Dict[str, str]], system_prompt: str = None
    ) -> List[Any]:
        """Convert conversation history to LangChain message format."""
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # Add conversation history
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        return messages

    @staticmethod
    def filter_for_final_response(history: List[Dict[str, str]]) -> List[Any]:
        """Filter history for generating a final response, excluding the last assistant message if applicable."""
        if len(history) >= 2 and history[-2]["role"] == "assistant":
            filtered_history = [
                (
                    HumanMessage(content=msg["content"])
                    if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                )
                for msg in history[:-2]
            ]
        else:
            filtered_history = [
                (
                    HumanMessage(content=msg["content"])
                    if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                )
                for msg in history
            ]
        return filtered_history


# Set up logging
def setup_logger():
    """Configure and return a logger with custom formatting"""
    logger = logging.getLogger("agent_simulation")
    logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler with formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a custom formatter
    formatter = logging.Formatter(
        f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} | "
        f"{Fore.YELLOW}%(levelname)s{Style.RESET_ALL} | "
        f"{Fore.WHITE}%(message)s{Style.RESET_ALL}"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def log_header(title, width=100, style=Fore.GREEN):
    """Create a header bar with centered title"""
    # Calculate visible length of the title (excluding ANSI color codes)
    visible_length = len(re.sub(r"\x1b\[[0-9;]*m", "", title))

    # Adjust width calculation to account for the full title display
    # We need space for: padding + " Title " + padding
    padding = max(width - visible_length - 2, 0)  # 2 spaces (one on each side of title)
    left_padding = padding // 2
    right_padding = padding - left_padding

    if UNICODE_SUPPORT:
        header = f"\n{style}{BOX['h'] * left_padding} {Back.BLACK}{title}{Style.RESET_ALL}{style} {BOX['h'] * right_padding}{Style.RESET_ALL}"
    else:
        header = f"\n{style}{BOX['h'] * left_padding} {title} {BOX['h'] * right_padding}{Style.RESET_ALL}"

    return header


def log_section(title, content=None, style=Fore.GREEN, width=100, title_style=None):
    """Create a bordered section with title and optional content"""
    title_style = title_style or style

    # Calculate visible length of the title (excluding ANSI color codes)
    visible_title_length = len(re.sub(r"\x1b\[[0-9;]*m", "", title))

    # Create header
    # Format: ┌─ Title ─────┐
    # We need: corner + dash + space + title + space + remaining_dashes + corner
    header = f"\n{style}{BOX['tl']}{BOX['h']}{Style.RESET_ALL} {title_style}{title}{Style.RESET_ALL} "

    # Precise calculation for the remaining horizontal lines
    # width = 1 (left corner) + 1 (dash) + 1 (space) + title_length + 1 (space) + remaining + 1 (right corner)
    # so remaining = width - title_length - 5
    header_remaining = max(width - visible_title_length - 5, 0)
    header += f"{style}{BOX['h'] * header_remaining}{BOX['tr']}{Style.RESET_ALL}"

    # If no content provided, just return the header
    if content is None:
        return header

    # Process content lines
    lines = []
    if isinstance(content, str):
        content_lines = content.split("\n")
        for line in content_lines:
            # Handle line wrapping for long lines
            while len(line) > width - 4:
                lines.append(line[: width - 7] + "...")
                line = line[width - 7 :]
            lines.append(line)
    else:
        lines = [str(content)]

    # Format each content line with side borders
    formatted_content = ""
    for line in lines:
        # Calculate visible length of the line (excluding ANSI color codes)
        visible_length = len(re.sub(r"\x1b\[[0-9;]*m", "", line))

        # Precise calculation for padding
        # width = 1 (border) + 1 (space) + line_length + padding + 1 (space) + 1 (border)
        # so padding = width - line_length - 4
        padding = max(width - visible_length - 4, 0)
        formatted_content += f"{style}{BOX['v']}{Style.RESET_ALL} {line}{' ' * padding} {style}{BOX['v']}{Style.RESET_ALL}\n"

    # Create footer with precise width
    # width = 1 (left corner) + (width-2) horizontal lines + 1 (right corner)
    footer = f"{style}{BOX['bl']}{BOX['h'] * (width - 2)}{BOX['br']}{Style.RESET_ALL}"

    # Combine all components
    return f"{header}\n{formatted_content}{footer}"


# Function to format JSON for display
def format_json_for_display(json_obj, max_length=None):
    """Format JSON for display with truncation if needed"""
    json_str = json.dumps(json_obj)
    if max_length is not None and len(json_str) > max_length:
        return json_str[:max_length] + "..."
    return json_str


def ensure_string(value: Any) -> str:
    """
    Ensure that a value is a string suitable for Galileo logging.

    Args:
        value: Any type of value

    Returns:
        A string representation of the value
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        # For dictionaries, lists, etc. use JSON
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        # For other types, convert to string
        return str(value)
    except Exception as e:
        print(f"Error converting to string: {str(e)}")
        return f"[Unconvertible data: {type(value).__name__}]"
