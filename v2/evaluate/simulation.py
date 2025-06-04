import json
import random
import time
import os
from typing import Dict, List, Any
from colorama import Fore, Style
from tqdm import tqdm
import pandas as pd

import config
from llm_handler import LLMHandler
from langchain_core.messages import HumanMessage
from galileo import galileo_context
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment, get_experiment
from galileo.projects import get_project, create_project
from dotenv import load_dotenv
from utils import (
    setup_logger,
    log_header,
    log_section,
    ensure_string,
    ConversationHistoryManager,
)

from simulator.llm_agent import LLMAgent
from simulator.tool_simulator import ToolSimulator
from simulator.user_simulator import UserSimulator

load_dotenv("../.env")

# Initialize logger without verbose flag by default
logger = setup_logger(verbose=False)

class AgentSimulation:

    def __init__(
        self,
        agent_model: str,
        domain: str,
        category: str,
        log_to_galileo: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the agent simulation.

        Args:
            agent_model: The model name for the agent being evaluated
            domain: The domain for the simulation (e.g., 'banking', 'healthcare')
            category: The category of the simulation (e.g., 'tool_coordination')
            log_to_galileo: Whether to log to Galileo
            verbose: Whether to print verbose logs
        """
        self.agent_model = agent_model
        self.domain = domain
        self.category = category
        self.log_to_galileo = log_to_galileo
        self.verbose = verbose

        # Initialize conversation history manager
        self.history_manager = ConversationHistoryManager()

        # Reinitialize the logger with the verbose flag
        global logger
        logger = setup_logger(verbose=verbose)

        # Initialize LLM handler
        self.llm_handler = LLMHandler()

        # Load data
        self.tools_with_output_schema = self._load_tools()
        # remove response_schema key from tools
        self.tools = [
            {k: v for k, v in tool.items() if k != "response_schema"}
            for tool in self.tools_with_output_schema
        ]
        self.personas = self._load_personas()
        self.scenarios = self._load_scenarios()

        # Initialize LLMs with tools 
        self.agent_llm = self.llm_handler.get_llm(
            model_name=agent_model,
            temperature=config.AGENT_TEMPERATURE,
            max_tokens=config.AGENT_MAX_TOKENS,
            tools=self.tools,  
        )

        self.simulator_llm = self.llm_handler.get_llm(
            model_name=config.SIMULATOR_MODEL,
            temperature=config.SIMULATOR_TEMPERATURE,
            max_tokens=config.SIMULATOR_MAX_TOKENS,
        )

        # Initialize Galileo logger
        if log_to_galileo:
            self.galileo_logger = galileo_context.get_logger_instance()
            logger.info(
                log_section(
                    "INITIALIZATION",
                    f"Galileo logger initialized successfully",
                    style=Fore.CYAN,
                )
            )
        else:
            self.galileo_logger = None

        # Tool simulator
        self.tool_simulator = ToolSimulator(
            domain=domain,
            category=category,
            simulator_llm=self.simulator_llm,
            tools=self.tools_with_output_schema,
            galileo_logger=self.galileo_logger,
            verbose=verbose,
        )

        # Agent
        self.agent = LLMAgent(
            model_name=agent_model,
            agent_llm=self.agent_llm,
            tool_simulator=self.tool_simulator,
            domain=domain,
            category=category,
            galileo_logger=self.galileo_logger,
            verbose=verbose,
            history_manager=self.history_manager,
        )

        # User simulator
        self.user_simulator = UserSimulator(
            simulator_llm=self.simulator_llm,
            history_manager=self.history_manager,
        )

        # Create configuration information string
        config_info = (
            f"{Fore.CYAN}Agent Model:{Style.RESET_ALL} {self.agent_model}\n"
            f"{Fore.CYAN}Simulator Model:{Style.RESET_ALL} {config.SIMULATOR_MODEL}\n"
            f"{Fore.CYAN}Domain:{Style.RESET_ALL} {self.domain}\n"
            f"{Fore.CYAN}Category:{Style.RESET_ALL} {self.category}\n"
            f"{Fore.CYAN}Max Turns:{Style.RESET_ALL} {config.MAX_TURNS}\n"
            f"{Fore.CYAN}Parallel Tool Execution:{Style.RESET_ALL} Enabled"
        )

        logger.info(
            log_header(
                f"SIMULATION INITIALIZED: {self.agent_model} - {self.domain}/{self.category}",
                style=Fore.CYAN,
            )
        )
        logger.info(log_section("CONFIGURATION", config_info, style=Fore.CYAN))

    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load tools for the specified domain."""
        path = config.FILE_PATHS["tools"].format(domain=self.domain)
        with open(path, "r") as f:
            return json.load(f)

    def _load_personas(self) -> List[Dict[str, Any]]:
        """Load personas for the specified domain."""
        path = config.FILE_PATHS["personas"].format(domain=self.domain)
        with open(path, "r") as f:
            return json.load(f)

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load scenarios for the specified domain and category."""
        path = config.FILE_PATHS["scenarios"].format(
            domain=self.domain, category=self.category
        )
        with open(path, "r") as f:
            return json.load(f)

    def run_simulation(self, scenario_idx: int = 0) -> Dict[str, Any]:
        """
        Run a full simulation for a single scenario.

        Returns:
            Dictionary containing the complete simulation results
        """
        # Get scenario and persona
        scenario = self.scenarios[scenario_idx]
        persona_idx = scenario.get("persona_index", 0)
        persona = self.personas[persona_idx]

        # Create system prompt with tools
        system_prompt = self.agent.update_agent_prompt(self.tools)

        # Initialize conversation history with system prompt
        conversation_history = (
            self.history_manager.initialize_history_with_system_prompt(system_prompt)
        )

        # Get initial message from scenario
        if isinstance(scenario.get("first_message"), list):
            # If first_message is a list, add all messages to history
            for msg in scenario["first_message"]:
                if msg.get("role") and msg.get("content"):
                    self.history_manager.add_message(
                        conversation_history, msg["role"], msg["content"]
                    )
            initial_user_message = conversation_history[-1]["content"]
        else:
            initial_user_message = scenario.get("first_message", "Hello")
            # Only add the user message if conversation_history doesn't already have it
            if (
                not conversation_history
                or conversation_history[-1].get("role") != "user"
            ):
                self.history_manager.add_message(
                    conversation_history, "user", initial_user_message
                )

        # Simulation loop
        turn_count = 0
        tool_outputs = []  # This will be used for simulating user responses
        all_tool_results = []  # This tracks all tool results across turns
        simulation_start_time = time.time()

        # Store all workflow inputs and outputs for Galileo context
        all_workflow_inputs = []
        all_workflow_outputs = []

        # Initialize results container
        results = {"turns_completed": 0, "success": False}

        # Log simulation start
        sim_start_info = (
            f"{Fore.CYAN}Domain:{Style.RESET_ALL} {self.domain} | "
            f"{Fore.CYAN}Category:{Style.RESET_ALL} {self.category} | "
            f"{Fore.CYAN}Scenario:{Style.RESET_ALL} {scenario_idx}"
        )
        print(log_header("SIMULATION START", style=Fore.BLUE))
        print(log_section("CONFIG", sim_start_info, style=Fore.BLUE))

        # Get initial user message to start the conversation
        current_user_message = initial_user_message

        # Create progress bar for turns
        turn_progress = tqdm(
            total=config.MAX_TURNS, desc="Simulation turns", leave=True
        )

        while turn_count < config.MAX_TURNS:
            turn_count += 1
            turn_start_time = time.time()
            turn_progress.update(1)

            # Log turn start with clear separator
            logger.info(log_header(f"TURN {turn_count}", style=Fore.GREEN))

            # Start a workflow span for this turn
            workflow_name = f"turn_{turn_count}_workflow"
            if self.galileo_logger:
                # For the first turn, use the scenario's first_message as input to ensure complete context
                input_message = (
                    scenario.get("first_message", current_user_message)
                    if turn_count == 1
                    else current_user_message
                )

                # Convert conversation history to LangChain messages to ensure consistent formatting
                langchain_messages = self.history_manager.to_langchain_messages(
                    conversation_history
                )

                # Add current user message if not in history yet
                if (
                    not conversation_history
                    or conversation_history[-1].get("role") != "user"
                ):
                    langchain_messages.append(
                        HumanMessage(content=current_user_message)
                    )

                # Create cumulative workflow input with history from previous turns
                workflow_input = "=== CONVERSATION HISTORY ===\n\n"

                # Add system prompt if present
                if (
                    conversation_history
                    and conversation_history[0].get("role") == "system"
                ):
                    system_content = conversation_history[0].get("content", "")
                    workflow_input += f"SYSTEM: {system_content}\n\n"

                # Reconstruct all previous turns from conversation history
                current_turn = 1
                message_index = 0

                # Skip system message at beginning if present
                if (
                    conversation_history
                    and conversation_history[0].get("role") == "system"
                ):
                    message_index = 1

                # Process all complete user-assistant exchanges
                while message_index + 1 < len(conversation_history):
                    # Get user message
                    if conversation_history[message_index].get("role") == "user":
                        workflow_input += f"=== TURN {current_turn} ===\n\n"
                        workflow_input += f"Human: {conversation_history[message_index].get('content')}\n\n"
                        message_index += 1
                    else:
                        message_index += 1
                        continue

                    # Get assistant response
                    if (
                        message_index < len(conversation_history)
                        and conversation_history[message_index].get("role")
                        == "assistant"
                    ):
                        workflow_input += f"Assistant: {conversation_history[message_index].get('content')}\n\n"
                        message_index += 1
                        current_turn += 1

                # Check if there's a user message left without a response
                # (this should be the last message before the current turn)
                if (
                    message_index < len(conversation_history)
                    and conversation_history[message_index].get("role") == "user"
                ):
                    workflow_input += f"=== TURN {current_turn} ===\n\n"
                    workflow_input += f"Human: {conversation_history[message_index].get('content')}\n\n"
                    current_turn += 1

                # Add current turn with the input
                # Only add if it's not already included in the conversation history
                if (
                    conversation_history[-1].get("role") != "user"
                    or conversation_history[-1].get("content") != input_message
                ):
                    workflow_input += f"=== TURN {turn_count} ===\n\n"
                    workflow_input += f"Human: {input_message}\n\n"

                # Store workflow input for next turn
                all_workflow_inputs.append(workflow_input)

                # Use the constructed history for Galileo logging
                self.galileo_logger.add_workflow_span(
                    input=workflow_input,
                    name=workflow_name,
                    tags=["conversation_turn", self.domain, self.category],
                )

            # Log user message
            logger.info(
                log_section("USER MESSAGE", current_user_message, style=Fore.BLUE)
            )

            # Run the agent and get response - using refactored agent
            final_response, tool_results = self.agent.run_agent(
                user_message=current_user_message,
                conversation_history=conversation_history,
                tools=self.tools,
                workflow_input=(
                    workflow_input if self.galileo_logger else None
                ),  # Pass workflow_input to agent
                previous_tool_outputs=tool_outputs,  # Pass the accumulated tool outputs
            )

            # Store this turn's inputs and outputs for future Galileo logging
            if (
                hasattr(self.agent, "workflow_context")
                and self.agent.workflow_context["inputs"]
            ):
                # Get the most recent input/output from the agent's workflow context
                all_workflow_inputs.append(self.agent.workflow_context["inputs"][-1])
                if self.agent.workflow_context["outputs"]:
                    all_workflow_outputs.append(
                        self.agent.workflow_context["outputs"][-1]
                    )

            # Calculate turn duration
            turn_duration_sec = time.time() - turn_start_time
            turn_duration_ns = int(turn_duration_sec * 1_000_000_000)

            # Process tools if any were detected
            if tool_results:
                tools_info = (
                    f"{Fore.YELLOW}Detected {len(tool_results)} tools:{Style.RESET_ALL} "
                    + ", ".join([t["tool_name"] for t in tool_results])
                )
                logger.info(log_section("TOOLS", tools_info, style=Fore.YELLOW))

                # Add to the persistent tool outputs list for user simulation
                tool_outputs.extend(tool_results)

                # Add to overall tool results list
                all_tool_results.extend(tool_results)
            else:
                logger.info(
                    log_section(
                        "TOOLS",
                        f"{Fore.YELLOW}NO TOOLS DETECTED{Style.RESET_ALL}",
                        style=Fore.YELLOW,
                    )
                )

            # Log agent response
            logger.info(log_section("AGENT RESPONSE", final_response, style=Fore.GREEN))

            # Add agent response to conversation history
            self.history_manager.add_message(
                conversation_history, "assistant", final_response
            )

            # Record turn results - only include final response, not detailed tool results
            turn_result = {
                "turn_id": turn_count,
                "user_input": current_user_message,
                "assistant_response": final_response,
                "processing_time_ms": int(turn_duration_sec * 1000),
            }
            # results["turns_results"].append(turn_result)
            results["turns_completed"] += 1

            # Simulate user response - using refactored user simulator
            user_response = self.user_simulator.simulate_user(
                persona=persona,
                scenario=scenario,
                conversation_history=conversation_history,
                tool_outputs=tool_outputs,  # Pass all accumulated tool outputs
            )

            # Add user response to conversation history
            self.history_manager.add_message(
                conversation_history, "user", user_response
            )

            # Update current user message for next turn
            current_user_message = user_response

            # Log turn completion
            logger.info(
                log_section(
                    "TURN END", f"Duration: {turn_duration_sec:.4f}s", style=Fore.GREEN
                )
            )

            # Conclude the workflow span for this turn
            if self.galileo_logger:
                self.galileo_logger.conclude(
                    output=f"Assistant: {ensure_string(final_response)}",
                    duration_ns=turn_duration_ns,
                )

            # Check if conversation should end
            if "CONVERSATION_COMPLETE" in user_response:
                logger.info(
                    log_section(
                        "COMPLETION",
                        f"Conversation complete after {turn_count} turns",
                        style=Fore.MAGENTA,
                    )
                )
                break

        # Close the progress bar
        turn_progress.close()

        # Log simulation end with clear separator
        total_duration = time.time() - simulation_start_time
        sim_end_info = (
            f"{Fore.CYAN}Total turns:{Style.RESET_ALL} {turn_count} | "
            f"{Fore.CYAN}Total time:{Style.RESET_ALL} {total_duration:.2f}s"
        )
        logger.info(log_header("SIMULATION COMPLETE", style=Fore.BLUE))
        logger.info(log_section("SUMMARY", sim_end_info, style=Fore.BLUE))

        results["success"] = True if results["turns_completed"] > 0 else False
        results["total_duration_ms"] = int(total_duration * 1000)

        return results


def create_experiment_runner(
    agent_model: str,
    domain: str,
    category: str,
    verbose: bool = False,
    log_to_galileo: bool = False,
):
    """
    Create a runner function for Galileo experiments.

    Args:
        agent_model: The model to use for the agent
        domain: The domain for experiments (e.g., 'banking', 'healthcare')
        category: The category of scenarios to run
        verbose: Whether to print verbose logs
        log_to_galileo: Whether to log to Galileo

    Returns:
        A function that can be passed to run_experiment
    """

    def runner(input_data):
        """
        Runner function for experiments that processes a single test case.

        Args:
            input_data: A dictionary containing test case data

        Returns:
            Results of the simulation as a JSON string
        """
        # Extract scenario index from input
        scenario_idx = input_data.get("scenario_idx", 0)

        # Print runner configuration
        runner_config = (
            f"{Fore.CYAN}Runner Configuration:{Style.RESET_ALL}\n"
            f"{Fore.CYAN}Agent Model:{Style.RESET_ALL} {agent_model}\n"
            f"{Fore.CYAN}Domain:{Style.RESET_ALL} {domain}\n"
            f"{Fore.CYAN}Category:{Style.RESET_ALL} {category}\n"
            f"{Fore.CYAN}Scenario Index:{Style.RESET_ALL} {scenario_idx}\n"
            f"{Fore.CYAN}Log to Galileo:{Style.RESET_ALL} {log_to_galileo}\n"
            f"{Fore.CYAN}Verbose:{Style.RESET_ALL} {verbose}\n"
            f"{Fore.CYAN}Parallel Tool Execution:{Style.RESET_ALL} Enabled"
        )
        print(log_section("RUN CONFIGURATION", runner_config, style=Fore.YELLOW))

        # Indicate progress
        print(f"Running scenario {scenario_idx}...")

        # Initialize simulation
        simulation = AgentSimulation(
            agent_model=agent_model,
            domain=domain,
            category=category,
            log_to_galileo=log_to_galileo,
            verbose=verbose,
        )

        # Get the first message from the scenario to include as input
        if scenario_idx < len(simulation.scenarios):
            scenario = simulation.scenarios[scenario_idx]
            first_message = scenario.get("first_message", "")
            # If it's provided as a list, extract the last content
            if isinstance(first_message, list) and first_message:
                first_message = first_message[-1].get("content", "")
        else:
            first_message = ""

        # Run the simulation and get results
        results = simulation.run_simulation(scenario_idx=scenario_idx)

        logger.info(
            log_section(
                "COMPLETE",
                f"Simulation complete for scenario {scenario_idx}",
                style=Fore.GREEN,
            )
        )

        # Return results as JSON string like in simple_test.py
        return json.dumps(results)

    return runner


def run_simulation_experiments(
    models: List[str],
    domains: List[str],
    categories: List[str],
    dataset_name: str = None,
    project_name: str = None,
    metrics: List[str] = config.METRICS,
    verbose: bool = False,
    log_to_galileo: bool = False,
    add_timestamp: bool = True,
):
    """
    Run experiments for all combinations of models, domains, and categories.

    Args:
        models: List of model names to evaluate
        domains: List of domains to evaluate
        categories: List of categories to evaluate
        dataset_name: Name of the dataset to use
        project_name: Galileo project name
        metrics: List of metrics to evaluate
        verbose: Whether to print verbose logs
        log_to_galileo: Whether to log to Galileo
        add_timestamp: Whether to add timestamp to experiment name

    Returns:
        Dictionary of experiment results
    """
    formatted_results = {}

    # Format model names for display
    model_names = ", ".join([m.split("/")[-1] if "/" in m else m for m in models])

    # Print detailed experiment configuration
    experiment_config = (
        f"{Fore.CYAN}Models:{Style.RESET_ALL} {model_names}\n"
        f"{Fore.CYAN}Domains:{Style.RESET_ALL} {', '.join(domains)}\n"
        f"{Fore.CYAN}Categories:{Style.RESET_ALL} {', '.join(categories)}\n"
        f"{Fore.CYAN}Dataset Name:{Style.RESET_ALL} {dataset_name if dataset_name else 'Auto-generated'}\n"
        f"{Fore.CYAN}Project:{Style.RESET_ALL} {project_name if project_name else 'Auto-generated from model names'}\n"
        f"{Fore.CYAN}Metrics:{Style.RESET_ALL} {', '.join(metrics)}\n"
        f"{Fore.CYAN}Verbose Logging:{Style.RESET_ALL} {verbose}\n"
        f"{Fore.CYAN}Log to Galileo:{Style.RESET_ALL} {log_to_galileo}\n"
        f"{Fore.CYAN}Add Timestamp:{Style.RESET_ALL} {add_timestamp}\n"
        f"{Fore.CYAN}Parallel Tool Execution:{Style.RESET_ALL} Enabled\n"
        f"{Fore.CYAN}Agent Temperature:{Style.RESET_ALL} {config.AGENT_TEMPERATURE}\n"
        f"{Fore.CYAN}Agent Max Tokens:{Style.RESET_ALL} {config.AGENT_MAX_TOKENS}\n"
        f"{Fore.CYAN}Simulator Model:{Style.RESET_ALL} {config.SIMULATOR_MODEL}\n"
        f"{Fore.CYAN}Simulator Temperature:{Style.RESET_ALL} {config.SIMULATOR_TEMPERATURE}\n"
        f"{Fore.CYAN}Simulator Max Tokens:{Style.RESET_ALL} {config.SIMULATOR_MAX_TOKENS}\n"
        f"{Fore.CYAN}Max Turns:{Style.RESET_ALL} {config.MAX_TURNS}"
    )

    logger.info(log_header("STARTING EXPERIMENTS", style=Fore.MAGENTA))
    logger.info(
        log_section("GLOBAL CONFIGURATION", experiment_config, style=Fore.MAGENTA)
    )

    # Print the total number of experiment combinations
    total_experiments = len(models) * len(domains) * len(categories)
    print(
        log_section(
            "EXPERIMENT SUMMARY",
            f"Running {total_experiments} total experiments ({len(models)} models × {len(domains)} domains × {len(categories)} categories)",
            style=Fore.MAGENTA,
        )
    )

    # Use tqdm to track progress across all experiments
    experiment_combinations = [
        (model, domain, category)
        for model in models
        for domain in domains
        for category in categories
    ]

    # Store original project_name state
    project_name_specified = project_name is not None

    for model, domain, category in tqdm(
        experiment_combinations, desc="Running experiments", total=total_experiments
    ):
        # Set project_name based on model if not specified
        current_project_name = project_name
        if not project_name_specified:
            # Extract the model name from the full path if applicable
            current_project_name = model.replace("/", "-")

        # Check if project exists and create it if it doesn't
        time.sleep(random.randint(0, 20))
        time.sleep(random.randint(0, 20))
        if not bool(get_project(name=current_project_name)):
            logger.info(f"Creating project: {current_project_name}")
            create_project(current_project_name)

        # Create experiment name, add timestamp only if flag is set
        if add_timestamp:
            timestamp = int(time.time())
            experiment_name = (
                f"{model.replace('/', '-')}-{domain}-{category}-{timestamp}"
            )
        else:
            experiment_name = f"{model.replace('/', '-')}-{domain}-{category}"

        # Check if experiment already exists
        project_id = get_project(name=current_project_name).id
        experiment_exists = bool(
            get_experiment(project_id=project_id, experiment_name=experiment_name)
        )

        if experiment_exists:
            logger.warning(
                log_section(
                    "EXPERIMENT EXISTS",
                    f"Skipping experiment '{experiment_name}' as it already exists in project '{current_project_name}'",
                    style=Fore.YELLOW,
                )
            )
            continue

        # Print detailed experiment configuration
        exp_config = (
            f"{Fore.CYAN}Experiment Name:{Style.RESET_ALL} {experiment_name}\n"
            f"{Fore.CYAN}Model:{Style.RESET_ALL} {model}\n"
            f"{Fore.CYAN}Domain:{Style.RESET_ALL} {domain}\n"
            f"{Fore.CYAN}Category:{Style.RESET_ALL} {category}\n"
            f"{Fore.CYAN}Project:{Style.RESET_ALL} {current_project_name}\n"
            f"{Fore.CYAN}Log to Galileo:{Style.RESET_ALL} {log_to_galileo}\n"
            f"{Fore.CYAN}Verbose:{Style.RESET_ALL} {verbose}\n"
            f"{Fore.CYAN}Parallel Tool Execution:{Style.RESET_ALL} Enabled"
        )
        logger.info(
            log_section("EXPERIMENT CONFIGURATION", exp_config, style=Fore.YELLOW)
        )

        # Create the runner function for this specific combination
        runner = create_experiment_runner(
            agent_model=model,
            domain=domain,
            category=category,
            verbose=verbose,
            log_to_galileo=log_to_galileo,
        )

        # Get or create dataset
        if dataset_name:
            dataset = get_dataset(name=dataset_name)
        else:
            # Create custom dataset from scenarios
            simulation = AgentSimulation(
                agent_model=model,
                domain=domain,
                category=category,
                log_to_galileo=False,
                verbose=verbose,
            )

            # Convert scenarios to dataset format - include first_message
            dataset = []
            for i, scenario in tqdm(
                enumerate(simulation.scenarios),
                desc="Processing scenarios",
                total=len(simulation.scenarios),
            ):
                first_message = scenario.get("first_message", "")
                # Handle both string and list formats
                if isinstance(first_message, list) and first_message:
                    first_message_text = first_message[-1].get("content", "")
                else:
                    first_message_text = first_message

                dataset.append({"scenario_idx": i, "first_message": first_message_text})

            logger.info(
                log_section(
                    "DATASET",
                    f"Created dataset with {len(dataset)} scenarios",
                    style=Fore.CYAN,
                )
            )

        # Run the experiment
        logger.info(
            log_section(
                "RUNNING EXPERIMENT",
                f"Running experiment with {len(dataset)} scenarios",
                style=Fore.CYAN,
            )
        )
        result = run_experiment(
            experiment_name=experiment_name,
            project=current_project_name,
            dataset=dataset,
            function=runner,
            metrics=metrics,
        )
        exp_data = result["experiment"]
        formatted_results[exp_data.name] = {
            "model_name": model,
            "category": category,
            "domain": domain,
            "project_id": exp_data.project_id,
            "id": exp_data.id,
            "name": exp_data.name,
            "created_at": str(exp_data.created_at),
            "link": result["link"],
            "message": result["message"],
        }

        # Print experiment completion details
        exp_result_info = (
            f"{Fore.CYAN}Experiment Name:{Style.RESET_ALL} {experiment_name}\n"
            f"{Fore.CYAN}Experiment ID:{Style.RESET_ALL} {exp_data.id}\n"
            f"{Fore.CYAN}Project ID:{Style.RESET_ALL} {exp_data.project_id}\n"
            f"{Fore.CYAN}Result Link:{Style.RESET_ALL} {result['link']}\n"
            f"{Fore.CYAN}Message:{Style.RESET_ALL} {result['message']}"
        )
        logger.info(
            log_section("EXPERIMENT COMPLETE", exp_result_info, style=Fore.YELLOW)
        )

    # Print overall experiment results summary
    logger.info(log_header("ALL EXPERIMENTS COMPLETED", style=Fore.MAGENTA))
    summary_info = (
        f"{Fore.CYAN}Total Experiments:{Style.RESET_ALL} {len(formatted_results)}\n"
        f"{Fore.CYAN}Models Tested:{Style.RESET_ALL} {model_names}\n"
        f"{Fore.CYAN}Domains Tested:{Style.RESET_ALL} {', '.join(domains)}\n"
        f"{Fore.CYAN}Categories Tested:{Style.RESET_ALL} {', '.join(categories)}"
    )
    print(log_section("EXPERIMENT SUMMARY", summary_info, style=Fore.MAGENTA))

    # Save results to CSV files organized by model name
    save_results_to_csv(formatted_results)

    return formatted_results


def save_results_to_csv(formatted_results: Dict[str, Dict[str, Any]]):
    """
    Save formatted experiment results to CSV files organized by model name.

    Args:
        formatted_results: Dictionary of experiment results
    """
    # Create experiments directory if it doesn't exist
    os.makedirs("../data/experiments", exist_ok=True)

    # Group results by model name
    model_results = {}
    for exp_name, result in formatted_results.items():
        # Extract model name from experiment name (format: model-domain-category-timestamp)
        model_name = result["model_name"]

        if model_name not in model_results:
            model_results[model_name] = []

        # Add this result to the model's results list
        result_row = {
            "domain": result["domain"],
            "category": result["category"],
            "link": result["link"],
            "experiment_name": exp_name,
            "project_id": result["project_id"],
            "experiment_id": result["id"],
            "created_at": result["created_at"],
            "message": result["message"],
        }
        model_results[model_name].append(result_row)

    # Save each model's results to its own CSV file
    for model_name, results in tqdm(
        model_results.items(), desc="Saving results to CSV"
    ):
        csv_path = f"../data/experiments/{model_name}.csv"
        df = pd.DataFrame(results)

        # Check if file exists and append if it does
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # Combine with new results
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on experiment_id
            combined_df = combined_df.drop_duplicates(
                subset=["experiment_id"], keep="last"
            )
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"Updated results for {model_name} in {csv_path}")
        else:
            # Create new file
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved results for {model_name} to {csv_path}")
