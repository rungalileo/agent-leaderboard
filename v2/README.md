# Agent Leaderboard v2

This is version 2 of the Agent Leaderboard project, featuring enhanced dataset generation and comprehensive evaluation capabilities for AI agents across multiple domains.

## Overview

The v2 system consists of:
- **Dataset Generation**: Create synthetic datasets with tools, personas, and scenarios for different domains
- **Agent Evaluation**: Simulate conversations between AI agents and users to evaluate performance
- **Results Analysis**: Collect and analyze metrics on agent performance

## Environment Setup

### Prerequisites
- Python 3.12
- API keys for LLM providers (OpenAI, Anthropic, etc.)

### 1. Install Dependencies

```bash
cd v2
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with your API keys:

```bash
# Required for dataset generation and evaluation
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional - for other LLM providers
GOOGLE_API_KEY=your_google_key_here
TOGETHER_API_KEY=your_together_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
MISTRAL_API_KEY=your_mistral_key_here
COHERE_API_KEY=your_cohere_key_here
XAI_API_KEY=your_xai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional - for Galileo logging
GALILEO_API_KEY=your_galileo_key_here
GALILEO_PROJECT_NAME=your_project_name
```

## Dataset Creation

### Quick Start: Generate Complete Dataset

Use the provided script to generate a complete dataset for a domain:

```bash
cd datasets
export domain=telecom && bash generate.sh
```

This script generates:
1. 20 tools for the finance domain
2. 100 personas
3. Scenarios for adaptive tool use category

### Custom Dataset Generation

#### 1. Generate Tools

Create domain-specific function definitions:

```bash
cd datasets
python tools.py --domain banking --num-tools 25 --overwrite
```

Options:
- `--domain`: Target domain (banking, healthcare, investment, telecom, etc.)
- `--num-tools`: Number of tools to generate (default: 20)
- `--overwrite`: Overwrite existing tools file

#### 2. Generate Personas

Create diverse user personas:

```bash
python personas.py --domain banking --num-personas 150 --overwrite
```

Options:
- `--domain`: Target domain
- `--num-personas`: Number of personas to generate (default: 100)
- `--overwrite`: Overwrite existing personas file

#### 3. Generate Scenarios

Create test scenarios for evaluation:

```bash
python scenarios.py --domain banking --categories adaptive_tool_use --overwrite
```

Options:
- `--domain`: Target domain
- `--categories`: Scenario categories (adaptive_tool_use, scope_management, empathetic_resolution, extreme_scenario_recovery, adversarial_input_mitigation)
- `--overwrite`: Overwrite existing scenarios file

### Available Domains

- **banking**: Financial services, transfers, account management
- **healthcare**: Patient records, appointments, health information
- **investment**: Portfolio management, trading, research
- **telecom**: Service management, troubleshooting, plan changes
- **automobile**: Vehicle services, maintenance, diagnostics
- **insurance**: Policy management, claims, coverage

### Generated Data Structure

Datasets are saved in `data/{domain}/`:
```
data/
├── banking/
│   ├── tools.json          # Function definitions for banking tools
│   ├── personas.json       # User personas for testing
│   └── adaptive_tool_use.json  # Test scenarios
└── healthcare/
    ├── tools.json
    ├── personas.json
    └── adaptive_tool_use.json
```

## Running Experiments

For evaluation across multiple models:

```bash
python run_parallel_experiments.py \
  --models "gpt-4.1-mini-2025-04-14,claude-3-7-sonnet-20250219" \
  --domains "banking,healthcare" \
  --categories "adaptive_tool_use" \
  --max-processes-per-model 2 \
  --log-to-galileo
```

### Experiment Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--models` | Comma-separated list of models to evaluate | `"gpt-4.1-mini-2025-04-14,claude-3-7-sonnet-20250219"` |
| `--domains` | Comma-separated list of domains | `"banking,healthcare,investment"` |
| `--categories` | Comma-separated list of scenario categories | `"adaptive_tool_use,scope_management"` |
| `--dataset-name` | Specific dataset name (optional) | `"banking_scenarios_v1"` |
| `--project-name` | Galileo project name for logging | `"agent-leaderboard-test"` |
| `--metrics` | Evaluation metrics | `"tool_selection_quality,agentic_session_success"` |
| `--verbose` | Enable detailed logging | |
| `--log-to-galileo` | Enable Galileo logging | |
| `--add-timestamp` | Add timestamp to experiment names | |
| `--max-processes` | Max parallel processes (parallel mode only) | `2` |

## Evaluation Categories

### adaptive_tool_use
Complex scenarios requiring sophisticated tool orchestration, conditional logic, and creative combinations to handle cascading dependencies and evolving requirements.

### scope_management  
Nuanced requests that mix legitimate tasks with subtly inappropriate or impossible requests, testing boundary recognition and graceful degradation.

### empathetic_resolution
Multi-layered customer issues combining urgent technical problems with emotional distress, requiring both precise tool usage and empathetic communication.

### extreme_scenario_recovery
High-stakes crisis situations with incomplete information, time pressure, and cascading failures requiring adaptive reasoning and rapid prioritization.

### adversarial_input_mitigation
Sophisticated social engineering and manipulation attempts disguised as legitimate requests, testing security awareness and boundary enforcement.

## Configuration

Key configuration options in `evaluate/config.py`:

```python
# LLM Configuration  
SIMULATOR_MODEL = "gpt-4.1-mini-2025-04-14"
AGENT_TEMPERATURE = 0.0
AGENT_MAX_TOKENS = 4000

# Simulation Configuration
MAX_TURNS = 15  # Maximum conversation turns

# Evaluation Metrics
METRICS = [
    "tool_selection_quality",
    "agentic_session_success",
]
```

## Results Analysis

Results are saved in `results/` directory with experiment metadata and can be analyzed using:

```bash
cd results
jupyter notebook get_score.ipynb
```

## Supported Models

The system supports models from various providers:

- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3 family
- **Google**: Gemini models  
- **Together AI**: Various open source models
- **Fireworks**: Optimized models
- **Mistral**: Mistral family
- **Cohere**: Command models
- **xAI**: Grok models
- **DeepSeek**: DeepSeek models

## Example Workflow

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   # Configure .env file with API keys
   ```

2. **Generate Dataset**:
   ```bash
   cd datasets
   python tools.py --domain banking --num-tools 30
   python personas.py --domain banking --num-personas 200  
   python scenarios.py --domain banking --categories adaptive_tool_use
   ```

3. **Run Evaluation**:
   ```bash
   cd ../evaluate
   python run_experiment.py \
     --models "gpt-4.1-mini-2025-04-14" \
     --domains "banking" \
     --categories "adaptive_tool_use" \
     --verbose
   ```

4. **Analyze Results**:
   ```bash
   cd ../results
   jupyter notebook get_score.ipynb
   ```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env` file
2. **Memory Issues**: Reduce `--max-processes` for parallel experiments
3. **Dataset Not Found**: Generate datasets before running experiments
4. **Model Not Supported**: Check that the model name matches the provider's format

### Debugging

Use `--verbose` flag for detailed logging:
```bash
python run_experiment.py --verbose --models "gpt-4.1-mini-2025-04-14" --domains "banking" --categories "adaptive_tool_use"
```

For more details, check the simulation logs and conversation history in the results output. 