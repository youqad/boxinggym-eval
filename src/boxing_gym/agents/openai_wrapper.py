from typing import List, Dict
import os

import openai
from openai import AsyncOpenAI, OpenAI


from anthropic import AsyncAnthropic, Anthropic

class CustomUsage:
    """
    Class to map Anthropic's usage structure to what's expected by BaseAgent
    """
    def __init__(self, input_tokens, output_tokens):
        self.prompt_tokens = input_tokens  # Map input_tokens to prompt_tokens
        self.completion_tokens = output_tokens  # Map output_tokens to completion_tokens
        self.total_tokens = input_tokens + output_tokens

class SimpleChoice:
    def __init__(self, content):
        self.message = type('Message', (), {'content': content})

class AsyncClaudeSonnet:
    """
    Simple wrapper for Claude 3.7 Sonnet.
    """
    def __init__(
        self, 
        max_tokens=1024,
        ):
        """
        Initializes AsyncAnthropic client.
        """
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-3-7-sonnet-20250219"
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "Claude 3.7 Sonnet"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        system_msg = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered_messages.append(msg)
        
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=filtered_messages,
            system=system_msg,
            **{k: v for k, v in kwargs.items() if k in ['temperature', 'top_p', 'stream']}
        )
        
        content = response.content[0].text if response.content else ""
        
        response.choices = [SimpleChoice(content)]
        
        response.model = "claude"
        
        response.usage = CustomUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        return response


class AsyncQwen25:
    """
    Wrapper for Qwen 2.5 3B model using Fireworks OpenAI-compatible API.
    """
    def __init__(
        self, 
        max_tokens=512,
        model_name="qwen2-7b-instruct",  # Update to correct Fireworks model name
        api_base="http://localhost:8000/v1",
    ):
        """
        Initializes Qwen client using Fireworks OpenAI-compatible API.
        
        Args:
            max_tokens: Maximum tokens for generation
            model_name: Name of the Qwen model on Fireworks
            api_base: Base URL for the Fireworks API
        """
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.api_base = api_base
        
        # Initialize OpenAI client with Fireworks API key
        self.client = AsyncOpenAI(
            api_key=os.environ.get("VLLM_API_KEY"),  # Get the API key from environment
            base_url=self.api_base
        )

    @property
    def llm_type(self):
        return "Qwen-2.5-7b"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call to Qwen model.
        
        Args:
            messages: List of message dictionaries with role and content
            **kwargs: Additional parameters for the API call
        
        Returns:
            Response in a format similar to other LLM wrappers
        """
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        temperature = kwargs.pop('temperature', 0.0)
        top_p = kwargs.pop('top_p', 0.8)
        
        response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **{k: v for k, v in kwargs.items() if k in ['stream', 'stop']}
            )
        
        if hasattr(response, 'model') and response.model == self.model_name:
            response.model = "qwen"
        
        return response
# client = AsyncOpenAI()
class AsyncOpenAIGPT4V:
    """
    Simple wrapper for an GPT4-V.
    """
    def __init__(
        self, 
        max_tokens=512,
        ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = AsyncOpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4 V"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        print(kwargs)
        return await self.client.chat.completions.create(
        # model="gpt-4-vision-preview",
        # model="gpt-4-turbo-2024-04-09",
        model="gpt-4o",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )

class OpenAIGPT4:
    """
    Simple wrapper for an GPT4.
    """
    def __init__(
        self, 
        max_tokens=768,
        ):
        """
        Initializes OpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4"

    def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        return self.client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )

class AsyncOpenAIGPT3_5_Turbo:
    """
    Simple wrapper for an GPT3.
    """
    def __init__(
        self, 
        max_tokens=768,
        ):
        """
        Initializes OpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-3.5"

    

class AsyncOpenAIGPT3_Turbo:
    """
    Simple wrapper for an GPT4-V.
    """
    def __init__(
        self, 
        max_tokens=512,
        ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = AsyncOpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4 V"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        print(kwargs)
        return await self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )