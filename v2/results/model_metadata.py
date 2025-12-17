# Model metadata with costs and vendor information
# api_version: "v1" uses old metric names (average_agentic_session_success, average_tool_selection_quality)
# api_version: "v2" uses new metric names (average_Action Completion - Agent Leaderboard, average_Tool Selection Quality - Agent Leaderboard)
model_metadata = {
    "claude-sonnet-4-20250514": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "Anthropic",
        "input_cost_per_m_token": 3,
        "output_cost_per_m_token": 15,
        "release_date": "2025-05"
    },
    "gemini-2.5-pro": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "Google",
        "input_cost_per_m_token": 1.25,
        "output_cost_per_m_token": 10,
        "release_date": "2025-03"
    },
    "gemini-2.5-flash": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "Google",
        "input_cost_per_m_token": 0.3,
        "output_cost_per_m_token": 2.5,
        "release_date": "2025-06"
    },
    "mistral-small-2506": {
        "api_version": "v1",
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Mistral",
        "input_cost_per_m_token": 0.1,
        "output_cost_per_m_token": 0.3,
        "release_date": "2025-06"
    },
    "deepseek-ai/DeepSeek-V3": {
        "api_version": "v1",
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Deepseek",
        "input_cost_per_m_token": 0.27,
        "output_cost_per_m_token": 1.1,
        "release_date": "2024-12"
    },
    "amazon.nova-pro-v1:0": {
        "api_version": "v1",
        "model_name": "nova-pro-v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "Amazon",
        "input_cost_per_m_token": 0.8,
        "output_cost_per_m_token": 3.2,
        "release_date": "2024-12"
    },
    # "amazon.nova-lite-v1:0": {
    #     "api_version": "v1",
    #     "model_name": "nova-lite-v1",
    #     "model_type": "Proprietary",
    #     "output_type": "Normal",
    #     "vendor": "Amazon",
    #     "input_cost_per_m_token": 0.06,
    #     "output_cost_per_m_token": 0.24
    # },
    "Qwen/Qwen2.5-72B-Instruct-Turbo": {
        "api_version": "v1",
        "model_name": "Qwen2.5-72B-Instruct",
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Alibaba",
        "input_cost_per_m_token": 0.9,
        "output_cost_per_m_token": 0.9,
        "release_date": "2024-09"
    },
    # "magistral-small-2506": {
    #     "api_version": "v1",
    #     "model_type": "Open source",
    #     "output_type": "Reasoning",
    #     "vendor": "Mistral",
    #     "input_cost_per_m_token": 0.5,
    #     "output_cost_per_m_token": 1.5
    # },
    # "magistral-medium-2506": {
    #     "api_version": "v1",
    #     "model_type": "Proprietary",
    #     "output_type": "Reasoning",
    #     "vendor": "Mistral",
    #     "input_cost_per_m_token": 2,
    #     "output_cost_per_m_token": 5
    # },
    "gpt-4.1-2025-04-14": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "OpenAI",
        "input_cost_per_m_token": 2,
        "output_cost_per_m_token": 8,
        "release_date": "2025-04"
    },
    "gpt-4.1-mini-2025-04-14": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "OpenAI",
        "input_cost_per_m_token": 0.4,
        "output_cost_per_m_token": 1.6,
        "release_date": "2025-04"
    },
    "gpt-4.1-nano-2025-04-14": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "OpenAI",
        "input_cost_per_m_token": 0.1,
        "output_cost_per_m_token": 0.4,
        "release_date": "2025-04"
    },
    # "arcee-ai/caller": {  
    #     "api_version": "v1",
    #     "model_type": "Open source",
    #     "output_type": "Normal",
    #     "vendor": "Arcee",
    #     "input_cost_per_m_token": 0.55,
    #     "output_cost_per_m_token": 0.85
    # },
    "grok-4-0709": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "xAI",
        "input_cost_per_m_token": 3,
        "output_cost_per_m_token": 15,
        "release_date": "2025-07"
    },
    "moonshotai/Kimi-K2-Instruct": {
        "api_version": "v1",
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Moonshot AI",
        "input_cost_per_m_token": 1,
        "output_cost_per_m_token": 3,
        "release_date": "2025-07"
    },
    # "GLM-4.5-Air-FP8": {
    #     "api_version": "v1",
    #     "model_name": "GLM-4.5-Air",
    #     "model_type": "Open source",
    #     "output_type": "Reasoning",
    #     "vendor": "Zai",
    #     "input_cost_per_m_token": 0.2,
    #     "output_cost_per_m_token": 1.1
    # },
    # "gemini-2.5-flash-lite": {
    #     "api_version": "v1",
    #     "model_name": "gemini-2.5-flash-lite",
    #     "model_type": "Proprietary",
    #     "output_type": "Reasoning",
    #     "vendor": "Google",
    #     "input_cost_per_m_token": 0.1,
    #     "output_cost_per_m_token": 0.4
    # },
    # "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {
    #     "api_version": "v1",
    #     "model_name": "Qwen3-235B-A22B-Instruct-2507",
    #     "model_type": "Open source",
    #     "output_type": "Reasoning",
    #     "vendor": "Alibaba",
    #     "input_cost_per_m_token": 0.2,
    #     "output_cost_per_m_token": 0.6
    # },
    # "Qwen3-235B-A22B-Thinking-2507": {
    #     "api_version": "v1",
    #     "model_type": "Open source",
    #     "output_type": "Reasoning",
    #     "vendor": "Alibaba",
    #     "input_cost_per_m_token": 0.65,
    #     "output_cost_per_m_token": 3.0
    # },
    # "Llama-3.3-70B-Instruct-Turbo": {
    #     "api_version": "v1",
    #     "model_name": "Llama-3.3-70B-Instruct",
    #     "model_type": "Open source",
    #     "output_type": "Normal",
    #     "vendor": "Meta",
    #     "input_cost_per_m_token": 0.88,
    #     "output_cost_per_m_token": 0.88
    # },
    "mistral-medium-2508": {
        "api_version": "v1",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "Mistral",
        "input_cost_per_m_token": 0.4,
        "output_cost_per_m_token": 2,
        "release_date": "2025-08"
    },
    # v2 models (new metric names)
    "grok-4-1-fast-non-reasoning": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Normal",
        "vendor": "xAI",
        "input_cost_per_m_token": 0.2,
        "output_cost_per_m_token": 0.5,
        "release_date": "2025-11"
    },
    "grok-4-1-fast-reasoning": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "xAI",
        "input_cost_per_m_token": 0.2,
        "output_cost_per_m_token": 0.5,
        "release_date": "2025-11"
    },
    "gpt-5.2-2025-12-11": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "OpenAI",
        "input_cost_per_m_token": 1.25,
        "output_cost_per_m_token": 10,
        "release_date": "2025-12"
    },
    "claude-haiku-4-5-20251001": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "Anthropic",
        "input_cost_per_m_token": 1,
        "output_cost_per_m_token": 5,
        "release_date": "2025-10"
    },
    "claude-sonnet-4-5-20250929": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "Anthropic",
        "input_cost_per_m_token": 3,
        "output_cost_per_m_token": 15,
        "release_date": "2025-09"
    },
    "claude-opus-4-5-20251101": {
        "api_version": "v2",
        "model_type": "Proprietary",
        "output_type": "Reasoning",
        "vendor": "Anthropic",
        "input_cost_per_m_token": 5,
        "output_cost_per_m_token": 25,
        "release_date": "2025-11"
    }
}
