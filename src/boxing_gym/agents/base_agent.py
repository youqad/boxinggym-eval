from typing import (
    Any,
    Dict,
    List, 
)

from abc import ABC, abstractmethod

# cost per token for each model input (USD, updated December 2025)
MODEL_COST_PER_INPUT = {
    # OpenAI models
    'gpt-4': 3e-05,                      # $30/1M tokens
    'gpt-4o': 2.5e-06,                   # $2.50/1M tokens
    'gpt-4o-mini': 1.5e-07,              # $0.15/1M tokens
    'gpt-4o-2024-05-13': 2.5e-06,        # $2.50/1M tokens
    'gpt-4o-2024-08-06': 2.5e-06,        # $2.50/1M tokens
    'gpt-4-turbo': 1e-05,                # $10/1M tokens
    'gpt-4-turbo-2024-04-09': 1e-05,     # $10/1M tokens
    'gpt-4-1106-preview': 1e-05,         # $10/1M tokens (GPT-4 Turbo preview)
    'openai/gpt-4-1106-preview': 1e-05,  # $10/1M tokens (prefixed)
    'gpt-4-1106-vision-preview': 1e-05,  # $10/1M tokens
    'gpt-4.1': 2e-06,                    # $2/1M tokens
    'gpt-3.5-turbo': 5e-07,              # $0.50/1M tokens
    'gpt-3.5-turbo-0125': 5e-07,         # $0.50/1M tokens
    'gpt-5': 1.25e-06,                   # $1.25/1M tokens
    'gpt-5-mini': 2.5e-07,               # $0.25/1M tokens
    'gpt-5-nano': 5e-08,                 # $0.05/1M tokens
    'gpt-5.1': 1.25e-06,                 # $1.25/1M tokens
    'gpt-5.1-mini': 2.5e-07,             # $0.25/1M tokens
    'gpt-5.1-codex-mini': 5e-07,         # ~$0.50/1M tokens (code-focused)
    'o1': 1.5e-05,                       # $15/1M tokens
    'o1-mini': 3e-06,                    # $3/1M tokens
    'o3': 1.5e-05,                       # $15/1M tokens (estimated)
    'o3-mini': 1.1e-06,                  # $1.10/1M tokens

    # DeepSeek models
    'deepseek/deepseek-chat': 2.8e-07,   # $0.28/1M tokens (cache miss)
    'deepseek-chat': 2.8e-07,            # $0.28/1M tokens
    'deepseek/deepseek-reasoner': 5.5e-07,  # $0.55/1M tokens
    'deepseek-reasoner': 5.5e-07,        # $0.55/1M tokens

    # Kimi/Moonshot models
    'moonshot/kimi-k2': 6e-07,           # $0.60/1M tokens
    'kimi-k2': 6e-07,                    # $0.60/1M tokens
    'anthropic/kimi-for-coding': 6e-07,  # $0.60/1M tokens

    # MiniMax models
    'openai/MiniMax-M2': 3e-07,          # $0.30/1M tokens
    'MiniMax-M2': 3e-07,                 # $0.30/1M tokens

    # GLM/ZhipuAI models
    'anthropic/glm-4.6': 4.5e-07,        # $0.45/1M tokens
    'glm-4.6': 4.5e-07,                  # $0.45/1M tokens

    # Anthropic Claude models
    'claude-3-5-sonnet': 3e-06,          # $3/1M tokens
    'claude-3-5-sonnet-20241022': 3e-06, # $3/1M tokens
    'claude-3-opus': 1.5e-05,            # $15/1M tokens
    'claude-3-haiku': 2.5e-07,           # $0.25/1M tokens
    'claude-3-5-haiku': 8e-07,           # $0.80/1M tokens
    'claude-3-5-haiku-20241022': 8e-07,  # $0.80/1M tokens
    'claude-haiku-4-5': 1e-06,           # $1/1M tokens
    'claude-sonnet-4': 3e-06,            # $3/1M tokens
    'claude-opus-4': 1.5e-05,            # $15/1M tokens
    'claude-opus-4-5': 1.5e-05,          # $15/1M tokens

    # Google Gemini models
    'gemini-2.0-flash': 1e-07,           # $0.10/1M tokens
    'gemini-2.0-flash-lite': 7.5e-08,    # $0.075/1M tokens
    'gemini-2.5-pro': 1.25e-06,          # $1.25/1M tokens
    'gemini-2.5-flash': 1.5e-07,         # $0.15/1M tokens
    'gemini-3.0-pro': 1.25e-06,          # $1.25/1M tokens (estimated)

    # Qwen Cloud API models
    'qwen-max': 1.6e-06,                 # $1.60/1M tokens
    'qwen2.5-max': 1.6e-06,              # $1.60/1M tokens
    'qwen-plus': 4.2e-07,                # $0.42/1M tokens
    'qwen-turbo': 5.25e-08,              # $0.0525/1M tokens
    'qwen3-235b-a22b': 2.415e-07,        # $0.2415/1M tokens

    # Local/Ollama models (free)
    'ollama/gpt-oss:20b': 0.0,           # Local model
    'ollama/qwen3-coder:30b': 0.0,       # Local model
    'ollama/qwen3:4b-instruct': 0.0,     # Local model
}

# cost per token for each model output (USD, updated December 2025)
MODEL_COST_PER_OUTPUT = {
    # OpenAI models
    'gpt-4': 6e-05,                      # $60/1M tokens
    'gpt-4o': 1e-05,                     # $10/1M tokens
    'gpt-4o-mini': 6e-07,                # $0.60/1M tokens
    'gpt-4o-2024-05-13': 1e-05,          # $10/1M tokens
    'gpt-4o-2024-08-06': 1e-05,          # $10/1M tokens
    'gpt-4-turbo': 3e-05,                # $30/1M tokens
    'gpt-4-turbo-2024-04-09': 3e-05,     # $30/1M tokens
    'gpt-4-1106-preview': 3e-05,         # $30/1M tokens (GPT-4 Turbo preview)
    'openai/gpt-4-1106-preview': 3e-05,  # $30/1M tokens (prefixed)
    'gpt-4-1106-vision-preview': 3e-05,  # $30/1M tokens
    'gpt-4.1': 8e-06,                    # $8/1M tokens
    'gpt-3.5-turbo': 1.5e-06,            # $1.50/1M tokens
    'gpt-3.5-turbo-0125': 1.5e-06,       # $1.50/1M tokens
    'gpt-5': 1e-05,                      # $10/1M tokens
    'gpt-5-mini': 2e-06,                 # $2/1M tokens
    'gpt-5-nano': 4e-07,                 # $0.40/1M tokens
    'gpt-5.1': 1e-05,                    # $10/1M tokens
    'gpt-5.1-mini': 2e-06,               # $2/1M tokens
    'gpt-5.1-codex-mini': 2e-06,         # ~$2/1M tokens (code-focused)
    'o1': 6e-05,                         # $60/1M tokens
    'o1-mini': 1.2e-05,                  # $12/1M tokens
    'o3': 6e-05,                         # $60/1M tokens (estimated)
    'o3-mini': 4.4e-06,                  # $4.40/1M tokens

    # DeepSeek models
    'deepseek/deepseek-chat': 4.2e-07,   # $0.42/1M tokens
    'deepseek-chat': 4.2e-07,            # $0.42/1M tokens
    'deepseek/deepseek-reasoner': 2.19e-06,  # $2.19/1M tokens
    'deepseek-reasoner': 2.19e-06,       # $2.19/1M tokens

    # Kimi/Moonshot models
    'moonshot/kimi-k2': 2.5e-06,         # $2.50/1M tokens
    'kimi-k2': 2.5e-06,                  # $2.50/1M tokens
    'anthropic/kimi-for-coding': 2.5e-06,  # $2.50/1M tokens

    # MiniMax models
    'openai/MiniMax-M2': 1.2e-06,        # $1.20/1M tokens
    'MiniMax-M2': 1.2e-06,               # $1.20/1M tokens

    # GLM/ZhipuAI models
    'anthropic/glm-4.6': 1.9e-06,        # $1.90/1M tokens
    'glm-4.6': 1.9e-06,                  # $1.90/1M tokens

    # Anthropic Claude models
    'claude-3-5-sonnet': 1.5e-05,        # $15/1M tokens
    'claude-3-5-sonnet-20241022': 1.5e-05,  # $15/1M tokens
    'claude-3-opus': 7.5e-05,            # $75/1M tokens
    'claude-3-haiku': 1.25e-06,          # $1.25/1M tokens
    'claude-3-5-haiku': 4e-06,           # $4/1M tokens
    'claude-3-5-haiku-20241022': 4e-06,  # $4/1M tokens
    'claude-haiku-4-5': 5e-06,           # $5/1M tokens
    'claude-sonnet-4': 1.5e-05,          # $15/1M tokens
    'claude-opus-4': 7.5e-05,            # $75/1M tokens
    'claude-opus-4-5': 7.5e-05,          # $75/1M tokens

    # Google Gemini models
    'gemini-2.0-flash': 4e-07,           # $0.40/1M tokens
    'gemini-2.0-flash-lite': 3e-07,      # $0.30/1M tokens
    'gemini-2.5-pro': 1e-05,             # $10/1M tokens
    'gemini-2.5-flash': 6e-07,           # $0.60/1M tokens
    'gemini-3.0-pro': 1e-05,             # $10/1M tokens (estimated)

    # Qwen Cloud API models
    'qwen-max': 6.4e-06,                 # $6.40/1M tokens
    'qwen2.5-max': 6.4e-06,              # $6.40/1M tokens
    'qwen-plus': 1.26e-06,               # $1.26/1M tokens
    'qwen-turbo': 2.1e-07,               # $0.21/1M tokens
    'qwen3-235b-a22b': 2.415e-06,        # $2.415/1M tokens

    # Local/Ollama models (free)
    'ollama/gpt-oss:20b': 0.0,           # Local model
    'ollama/qwen3-coder:30b': 0.0,       # Local model
    'ollama/qwen3:4b-instruct': 0.0,     # Local model
}

class BaseAgent(ABC):
    """
    Base agent class.
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str, 
    ) -> None:
        """Initializes a chat model used for code or feedback models.

        Args:
            llm: The LLM Chat model. Currently either a CRFM or Azure Chat Model.
            model_id: The unique identifier of the model
        """
        self.llm = llm
        self.model_id = model_id
    
    def calc_cost(
        self, 
        response
    ) -> float:
        """
        Calculates the cost of a response from the openai API. Taken from https://github.com/princeton-nlp/SWE-bench/blob/main/inference/run_api.py

        Args:
        response (openai.ChatCompletion): The response from the API.

        Returns:
        float: The cost of the response.
        """
        model_name = getattr(response, "model", None)
        usage = getattr(response, "usage", None)

        input_tokens = 0
        output_tokens = 0

        # ChatCompletion-style usage
        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
            # Responses API usage schema
            elif hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens"):
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
            elif isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0

        # instead of crashing on KeyError, fall back to zero-cost accounting for unknown models
        input_rate = MODEL_COST_PER_INPUT.get(model_name, 0.0)
        output_rate = MODEL_COST_PER_OUTPUT.get(model_name, 0.0)
        cost = input_rate * input_tokens + output_rate * output_tokens
        return cost
        
    @abstractmethod
    def get_prompt(self) -> List[Dict[str, str]]:
        """
        Get the prompt fed into the model. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(self) -> str:
        """
        Get the response from the model. 
        """
        raise NotImplementedError

    @abstractmethod 
    def run(self) -> Dict[str, Any]:
        """
        Run the agent.
        """
        raise NotImplementedError