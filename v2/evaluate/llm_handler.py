import os

from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
# from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_writer import ChatWriter
from langchain_deepseek import ChatDeepSeek
from langchain_xai import ChatXAI

class LLMHandler:
    """
    A class to handle different LLM providers (Mistral, Anthropic, and Google) using LangChain integrations.
    """

    def __init__(self):
        self.available_models = {
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-7-sonnet-20250219",
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
            ],
            "mistral": [
                "mistral-small-2506",
                "mistral-medium-2505",
                "magistral-small-2506",
                "magistral-medium-2506",
            ],
            "google": [
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-lite-001",
                "gemini-2.5-flash-lite-preview-06-17",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ],
            "together": [
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-R1-0528-tput",
                "deepseek-ai/DeepSeek-V3",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "Qwen/Qwen3-235B-A22B-fp8-tput",
                "arcee-ai/caller",
                "arcee_ai/arcee-spotlight",
                "arcee-ai/AFM-4.5B-Preview",
                "google/gemma-3n-E4B-it",
                "moonshotai/Kimi-K2-Instruct",
            ],
            "openai": [
                "gpt-4o-2024-11-20",
                "gpt-4o-mini",
                "o1-2024-12-17",
                "o3-mini-2025-01-31",
                "gpt-4.1-2025-04-14",
                "gpt-4.1-mini-2025-04-14",
                "gpt-4.1-nano-2025-04-14",
            ],
            "fireworks": [
                "accounts/fireworks/models/qwen-qwq-32b-preview",
                "accounts/fireworks/models/qwen2p5-72b-instruct",
                "accounts/fireworks/models/deepseek-r1",
                "accounts/fireworks/models/llama4-maverick-instruct-basic",
                "accounts/fireworks/models/llama4-scout-instruct-basic",
            ],
            "bedrock": [
                "amazon.nova-pro-v1:0",
                "amazon.nova-micro-v1:0",
                "amazon.nova-lite-v1:0",
            ],
            "cohere": [
                "command-a-03-2025",
            ],
            "nvidia": [
                "nvidia/llama-3.3-nemotron-super-49b-v1",
                "nvidia/llama-3.1-nemotron-nano-8b-v1",
            ],
            "writer": [
                "palmyra-x-004",
                "palmyra-x5"
            ],
            "deepseek": [
                "deepseek-chat",
            ],
            "ibm": [
                "ibm-granite/granite-3.3-8b-instruct",
            ],
            "xai": [
                "grok-3",
                "grok-3-mini",
                "grok-4-0709",
            ],
        }       

        self.model_name_to_provider = {name:provider for provider, models in self.available_models.items() for name in models}

    def _detect_provider(self, model_name: str) -> str:
        """
        Automatically detects the provider based on the model name.
        """
        if model_name in self.model_name_to_provider:
            return self.model_name_to_provider[model_name]
        raise ValueError(f"Could not detect provider for model name: {model_name}")


    def get_llm(
        self,
        model_name: str, 
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4000,
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> BaseChatModel:
        """
        Creates and returns an LLM instance based on the provider and model name.

        Args:
            model_name: The specific model to use
            temperature: Temperature parameter for generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: Optional API key (if not set in environment variables)
            tools: Optional list of tools to bind to the LLM
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            A LangChain chat model instance
        """
        # Auto-detect provider if not specified
        provider = self._detect_provider(model_name)

        # Set API key if provided
        if api_key:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

        model_params = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}

        # remove temperature if model name contains o1
        if ("o1" in model_name) or ("o3" in model_name):
            model_params.pop("temperature", None)

        # set temperature to 0.6 if model name contains r1
        if ("r1" in model_name) or ("nvidia/llama-3.3-nemotron" in model_name):
            model_params["temperature"] = 0.6

        if "deepseek" in model_name:
            model_params["temperature"] = 0.0

        # Remove None values
        model_params = {k: v for k, v in model_params.items() if v is not None}

        # Create the base LLM
        llm = None

        if provider == "anthropic":
            llm = ChatAnthropic(model_name=model_name, **model_params)
        elif provider == "mistral":
            llm = ChatMistralAI(model_name=model_name, **model_params)
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(model=model_name, **model_params)
        elif provider == "together":
            llm = ChatTogether(model=model_name, **model_params)
        elif provider == "openai":
            llm = ChatOpenAI(model=model_name, **model_params)
        # elif provider == "fireworks":
            # llm = ChatFireworks(model=model_name, **model_params)
        elif provider == "bedrock":
            llm = ChatBedrock(model_id=model_name, **model_params)
        elif provider == "cohere":
            llm = ChatCohere(model=model_name, **model_params)
        elif provider == "nvidia":
            llm = ChatNVIDIA(model=model_name, **model_params)
        elif provider == "writer":
            llm = ChatWriter(model=model_name, **model_params)
        elif provider == "deepseek":
            llm = ChatDeepSeek(model=model_name, **model_params)
        elif provider == "vertexai":
            llm = ChatVertexAI(model=model_name, **model_params)
        elif provider == "xai":
            llm = ChatXAI(model=model_name, **model_params)
        elif provider == "ibm":
            base_url ='https://galileo-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct-gll/v1'
            llm = ChatOpenAI(
                model=model_name,
                max_retries=3,
                api_key=os.getenv("RITS_API_KEY"),
                base_url=base_url,
                default_headers={'RITS_API_KEY': os.getenv("RITS_API_KEY")},
                **model_params
            )

        # Bind tools if provided
        if tools and llm:
            llm = llm.bind_tools(tools)

        llm_with_retry = llm.with_retry(
        retry_if_exception_type=(Exception,),      # Exception types to retry on
        wait_exponential_jitter=True,              # Use exponential backoff with jitter
        stop_after_attempt=7                       # Maximum number of attempts
    )

        return llm_with_retry

    @staticmethod
    def get_token_usage_info(response):
        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"] 
        if "output_token_details" in response.usage_metadata:
            if "reasoning" in response.usage_metadata["output_token_details"]:
                output_tokens += response.usage_metadata["output_token_details"]["reasoning"]
        return input_tokens, output_tokens