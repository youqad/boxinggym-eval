from typing import (
    Any,
    Dict,
    List,
)

from abc import ABC, abstractmethod

# import unified pricing from pricing.py (single source of truth)
from boxing_gym.agents.pricing import MODEL_COST_PER_INPUT, MODEL_COST_PER_OUTPUT

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