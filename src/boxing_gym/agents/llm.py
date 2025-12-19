from typing import (
    Any,
    Dict,
    List, 
)

import asyncio
import time

from boxing_gym.agents.base_agent import BaseAgent

# TODO: inherit from this class so we can support both stan and gp stuff
# since the prompts for those are a bit different
class LLMAgent(BaseAgent):
    """
    Simple (Chat) LLM Agent class supporting async API calls.
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str, 
        use_vision: bool,
        batch_mode: bool=True,
        warm_start_examples: List = [],
        logger=None,
        vision_only=False,
        # budget: int,
        **model_args,
    ) -> None:
        super().__init__(llm=llm, model_id=model_id)
        self.model_args = model_args
        self.all_responses = []
        self.total_inference_cost = 0

        # tracking for W&B / benchmarking.
        self._usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
            "latencies_ms": [],
            "retry_count": 0,
            "error_count": 0,
        }

        self.batch_mode = batch_mode

        self.warm_start_examples = warm_start_examples

        self.use_vision = use_vision 

        self.vision_only = vision_only 

        self.logger = logger
        self.messages_all = []

    def get_prompt(
        self, 
        system_message: str, 
        user_message: str, 
        data_image: str, 
        incontext_info: List):
        raise NotImplementedError()

    async def get_response(
        self, 
        messages: List[Dict[str, str]],
        n: int = 1, 
        temperature: float = 0.7,
    ) -> Any:
        """
        Get the response from the model.
        """
        self.model_args['temperature'] = temperature
        self.model_args['n'] = n

        # print(f"messages: {messages}")
        self.logger.info(f"running with temperature: {self.model_args['temperature']}, n = {self.model_args['n']}")
        return await self.llm(messages=messages, **self.model_args)
    
    async def run(
        self, 
        expertise: str,
        user_message: str,
        data_image: str, 
        incontext_info: List,
        n: int = 1, 
        temperature: float = 0.7,
    ) -> Dict[str, Any]:

        
        """Runs the Code Agent

        Args:
            expertise (str): The system message to use
            message (str): The user message to use

        Returns:
            A dictionary containing the code model's response and the cost of the performed API call
        """

        # Get the prompt
        messages = self.get_prompt(
            system_message=expertise, 
            user_message=user_message, 
            data_image=data_image, 
            n=n,
            incontext_info=incontext_info)

        # self.logger.info(f"messages: {messages}")

        self.messages_all.append(messages)

        # get the response
        call_start = time.time()
        try:
            response = await self.get_response(messages=messages, n=n, temperature=temperature)
        except Exception as e:
            # Be robust to transient provider / rate-limit failures so Box's loop sweeps don't crash.
            try:
                if self.logger:
                    self.logger.warning(f"LLM call failed in Box loop agent: {e}")
            except Exception:
                pass
            # Return empty content in the expected shape.
            return "" if self.batch_mode else [""]
        call_end = time.time()

        # track latency and token usage.
        try:
            self._usage_stats["call_count"] += 1
            self._usage_stats["latencies_ms"].append((call_end - call_start) * 1000)
        except Exception:
            pass

        usage = getattr(response, "usage", None)
        if usage is None and hasattr(response, "__dict__"):
            usage = response.__dict__.get("usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        if usage:
            prompt_tokens = 0
            completion_tokens = 0
            reasoning_tokens = 0
            try:
                if hasattr(usage, "prompt_tokens"):
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
                elif hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens"):
                    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                    completion_tokens = getattr(usage, "output_tokens", 0) or 0
                    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
                elif isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                    reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
            except Exception:
                pass

            try:
                self._usage_stats["prompt_tokens"] += prompt_tokens
                self._usage_stats["completion_tokens"] += completion_tokens
                self._usage_stats["reasoning_tokens"] += reasoning_tokens
                self._usage_stats["total_tokens"] += prompt_tokens + completion_tokens + reasoning_tokens
            except Exception:
                pass

        # Extract content regardless of Chat or Responses schema
        def _extract_contents(resp):
            if hasattr(resp, "choices") and getattr(resp, "choices"):
                contents = []
                for c in resp.choices:
                    msg = getattr(c, "message", None)
                    if msg is None and isinstance(c, dict):
                        msg = c.get("message")
                    content = None
                    reasoning = None
                    try:
                        if msg is not None:
                            content = getattr(msg, "content", None)
                            reasoning = getattr(msg, "reasoning_content", None)
                            if content is None and isinstance(msg, dict):
                                content = msg.get("content")
                                reasoning = msg.get("reasoning_content")
                    except Exception:
                        content = None
                        reasoning = None

                    if content and reasoning:
                        contents.append(f"{content}\n\n{reasoning}")
                    elif content:
                        contents.append(content)
                    elif reasoning:
                        contents.append(reasoning)
                if contents:
                    return contents
            output = getattr(resp, "output", None)
            texts = []
            if output:
                for item in output:
                    item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                    if item_type == "message":
                        msg_content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                        if msg_content:
                            for c in msg_content:
                                text = getattr(c, "text", None)
                                if text is None and isinstance(c, dict):
                                    text = c.get("text")
                                if text:
                                    texts.append(text)
            return texts or [""]

        contents = _extract_contents(response)

        # get cost
        cost = self.calc_cost(response=response)
        print(f"Cost for running: {cost}")
        self.total_inference_cost += cost
        try:
            self._usage_stats["total_cost_usd"] = float(self.total_inference_cost)
        except Exception:
            pass

        if self.batch_mode:
            # Store response including cost 
            full_response = {
                'response': response,
                'response_str': contents[0] if contents else "",
                'cost': cost
            }
            # Update total cost and store response
            
            self.all_responses.append(full_response)
            # Return response_string
            return full_response['response_str']
        else:
            return contents

    def get_usage_stats(self) -> dict:
        """Return accumulated usage statistics for logging."""
        stats = dict(self._usage_stats)
        latencies = stats.pop("latencies_ms", [])
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            stats["latency_mean_ms"] = sum(latencies) / n
            stats["latency_p50_ms"] = latencies_sorted[n // 2]
            stats["latency_p95_ms"] = latencies_sorted[int(n * 0.95)] if n >= 20 else latencies_sorted[-1]
            stats["latency_min_ms"] = latencies_sorted[0]
            stats["latency_max_ms"] = latencies_sorted[-1]
        else:
            stats["latency_mean_ms"] = 0.0
            stats["latency_p50_ms"] = 0.0
            stats["latency_p95_ms"] = 0.0
            stats["latency_min_ms"] = 0.0
            stats["latency_max_ms"] = 0.0
        return stats

    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
            "latencies_ms": [],
            "retry_count": 0,
            "error_count": 0,
        }
    
    async def batch_prompt_sync(
        self, 
        expertise: str, 
        messages: List[str],
        incontext_info: List,
        data_image: str,
        n: int = 1,
        temperature: float = 0.7,
    ) -> List[str]:
        """Handles async API calls for batch prompting.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the code model for each message
        """
        responses = [self.run(expertise=expertise, 
                              user_message=message, 
                              incontext_info=incontext_info,
                              data_image=data_image,
                              n=n,
                              temperature=temperature) for message in messages]
        return await asyncio.gather(*responses)

    def batch_prompt(
        self, 
        expertise: str, 
        messages: List[str], 
        incontext_info: List,
        data_image: str,
        temperature: float = 0.7,
        n: int = 1,
    ) -> List[str]:
        """=
        Synchronous wrapper for batch_prompt.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages
            temperature (str): The temperature to use for the API call

        Returns:
            A list of responses from the code model for each message
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(
            expertise=expertise, 
            messages=messages, 
            temperature=temperature, 
            data_image=data_image,
            n=n,
            incontext_info=incontext_info
            ))

class ModelCriticismLLMAgent(LLMAgent):
    def add_incontext_examples(self, messages: List[str], incontext_info: List):
      for i, results in enumerate(incontext_info):
        messages.append(
            {
                "role": "user", 
                "content": [
                {"type": "text", "text": f"""
                    1. Program {i}: \n {results["test_fn"]}  \n 
                    2. Bayesian p val: \n {results["bayesian_pval_rescaled"]}. \n
                    2. Explanation: \n {results["t_data"]['test_statistic_explanation']}. \n
                """
                },
                ]
            }

        )

        self.logger.info(messages[-1])
        if "fig" in results:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                    {"type": "text", "text": f"""
                        1. Program {i}: \n {results["discrepancy_fn"]}  \n 
                        2. Image: \n 
                    """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{results['fig']}"}},
                    ]
                }
            )

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
        data_image: str, 
        n: int,
        incontext_info: List = [],
    ) -> List[Dict[str, str]]:
        """
        Get the prompt for the (chat) model.
        """
        messages = []
        if data_image is None:
            system_msg_dict = {
                "role": "system", 
                "content": system_message
            }
            
        # else:
        #     system_msg_dict = {
        #         "role": "system", 
        #         "content": 
        #         [
        #             {"type": "text", "text": system_message},
        #             {"type": "text", "text": "Here is a plot of the data itself."},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data_image}"}}
        #         ]
        #     }

        messages.append(system_msg_dict)

        messages.append({"role": "user", "content": user_message})

        self.logger.info(f"data_image: {data_image}")
        
        if len(incontext_info) > 0: 
          self.add_incontext_examples(messages=messages, incontext_info=incontext_info)
        return messages

class StanProposalLLMAgent(LLMAgent):

    def add_incontext_examples(self, messages: List[str], incontext_info: List):

      # i know you don't need this
      if len(self.warm_start_examples) > 0: 
        for results in self.warm_start_examples:
          if self.vision_only:
            raise Exception("Vision only is not supported for StanProposalLLMAgent if warm-starting")
          else:
            messages.append(
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "text", "text": f"""
                        Here is the previous program: \n
                        1. Program: {results['str_prob_prog']}\n 
                        2. The LOO score is: {results['loo']}. \n
                        2. The summary stats are: {results['summary_stats_df']}. \n
                        """
                        },
                    ]
                }
            )
        # 3. The summary stats from posterior predictive: \n {results["summary_stats"]} \n
          self.logger.info(f"Warm start: {messages[-1]['content']}")
           

      if len(incontext_info) > 0:
        messages.append(
            {
                "role": "user", 
                "content": """
                  Here's a list of the best programs in descending order.
                """
            }
        )
        for i, results in enumerate(incontext_info):
          if self.vision_only:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                    {"type": "text", "text": f"""
                        1. Program {i}: \n {results["str_prob_prog"]}  \n 
                        2. The LOO score: \n {results["loo"]}. \n
                        3. A plot of the posterior predictive against the true data: \n
                    """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{results['posterior_plot']}"}}
                    ]
                }
            )
            self.logger.info(f"Exemplar: {messages[-1]['content'][0]}")
          else:
            messages.append(
                {
                    "role": "user", 
                    "content": f"""
                        1. Program {i}: \n {results["str_prob_prog"]}  \n 
                        2. The LOO score: \n {results["loo"]}. \n
                    """
                        # 3. The summary stats from posterior predictive: \n {results["summary_stats"]} \n
                }
            )
            self.logger.info(f"Exemplar: {messages[-1]['content']}")

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
        data_image: str, 
        n: int,
        incontext_info: List = [],
    ) -> List[Dict[str, str]]:
        """
        Get the prompt for the (chat) model.
        """
        messages = []
        if data_image is None:
            system_msg_dict = {
                "role": "system", 
                "content": system_message
            }
            
        else:
            system_msg_dict = {
                "role": "system", 
                "content": 
                [
                    {"type": "text", "text": system_message},
                    {"type": "text", "text": "Here is a plot of the data itself. Use the properties in the plot to inform your modeling choices."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data_image}"}}
                ]
            }
            print("data image is not None!")

        messages.append(system_msg_dict)
        self.logger.info(f"system_msg_dict: {messages[-1]['content']}")
        messages.append({"role": "user", "content": user_message})
        if (len(incontext_info) > 0) or len(self.warm_start_examples) > 0: 
          self.add_incontext_examples(messages=messages, incontext_info=incontext_info)
            
        return messages

class StanCriticLLMAgent(LLMAgent):
    def add_incontext_examples(self, messages: List[str], incontext_info: List):
      for i, results in enumerate(incontext_info):
        if self.vision_only:
          messages.append(
              {
                  "role": "user", 
                  "content": [
                  {"type": "text", "text": f"""
                      1. Program {i}: \n {results["str_prob_prog"]}  \n 
                      2. The LOO score: \n {results["loo"]}. \n
                      3. A plot of the posterior predictive against the true data: \n
                  """
                  },
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{results['posterior_plot']}"}}
                  ]
              }
          )
        else:
          messages.append(
              {
                  "role": "user", 
                  "content": f"""
                      1. Program: \n {results["str_prob_prog"]}
                      2. The LOO score: {results["loo"]}. \n
                      3. The summary stats from posterior predictive: {results["summary_stats"]} \n
                  """
              }
          )
        self.logger.info(f"Exemplar: {messages[-1]['content']}")

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
        data_image: str, 
        n: int,
        incontext_info: List = [],
    ) -> List[Dict[str, str]]:
        """
        Get the prompt for the (chat) model.
        """
        messages = []
        if data_image is None:
            system_msg_dict = {
                "role": "system", 
                "content": system_message
            }
        else:
            system_msg_dict = {
                "role": "system", 
                "content": 
                [
                    {"type": "text", "text": system_message},
                    {"type": "text", "text": "Here is a plot of the data itself. Use the properties in the plot to inform your criticism."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data_image}"}}
                ]
            }

        messages.append(system_msg_dict)
        messages.append({"role": "user", "content": user_message})

        if len(incontext_info) > 0: 
          self.add_incontext_examples(messages=messages, incontext_info=incontext_info)
        return messages