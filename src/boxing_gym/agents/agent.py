import anthropic
import re
from openai import AsyncOpenAI, OpenAI
import openai
import os

class LMExperimenter:
    def __init__(self, model_name, temperature=0.0, max_tokens=256):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = None
        self.messages = []
        self.all_messages = []
        if "gpt-4o" in model_name:
            self.llm = openai.OpenAI()
        elif "claude" in model_name:
            self.llm = anthropic.Anthropic()
        elif 'deepseek' in model_name.lower():
            self.llm = OpenAI(
                api_key=os.environ.get("VLLM_API_KEY", "token-abc123"),
                base_url=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            )
            os.environ["VLLM_MODEL_NAME"] = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        elif 'qwen' in model_name.lower():
            self.llm = OpenAI(
                api_key=os.environ.get("VLLM_API_KEY", "token-abc123"),
                base_url=os.environ.get("VLLM_API_BASE", "http://cocoflops1:8000/v1")
            )
            os.environ["VLLM_MODEL_NAME"] = "Qwen/Qwen2.5-32B-Instruct"
        elif 's1' in model_name.lower():
            self.llm = OpenAI(
                api_key=os.environ.get("VLLM_API_KEY", "token-abc123"),
                base_url=os.environ.get("VLLM_API_BASE", "http://cocoflops1:8000/v1")
            )
            os.environ["VLLM_MODEL_NAME"] = "simplescaling/s1-32B"
        elif 'openthinker' in model_name.lower():
            self.llm = OpenAI(
                api_key=os.environ.get("VLLM_API_KEY", "token-abc123"),
                base_url=os.environ.get("VLLM_API_BASE", "http://cocoflops1:8000/v1")
            )
            os.environ["VLLM_MODEL_NAME"] = "open-thoughts/OpenThinker-7B"

    def set_system_message(self, message):
        self.all_messages.append(f"role:system, messaage:{message}")
        if "gpt-4o" in self.model_name:
            self.system = message
            self.messages.append({"role": "system", "content": [{"type": "text", "text": message}]})
        elif "claude" in self.model_name:
            self.system = message
        elif "qwen" or "deepseek" or "s1" or "openthinker" in self.model_name.lower():
            # Qwen uses the same message format as OpenAI
            self.system = message
            self.messages.append({"role": "system", "content": message})
        
    def add_message(self, message, role='user'):
        # if role == "assistant":
        #     # remove everything in thought tags
        #     message = re.sub(r'<thought>.*?</thought>', '', message)

        self.all_messages.append(f"role:{role}, messaage:{message}")
        if "gpt-4o" in self.model_name:
            self.messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                })
        elif "claude" in self.model_name:
            self.messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": message 
                        }
                    ]
                })
        elif "qwen" or "deepseek" or "openthinker" in self.model_name.lower():
            # Qwen uses standard OpenAI message format
            self.messages.append(
                {
                    "role": role,
                    "content": message
                })

    def prompt_llm(self, request_prompt):
        self.add_message(request_prompt)
        if "gpt-4o" in self.model_name:
            full_response = self.llm.chat.completions.create(model=self.model_name, messages=self.messages, max_tokens=self.max_tokens, temperature=self.temperature)#.content[0].text
            full_response = full_response.choices[0].message.content
        elif "claude" in self.model_name:
            full_response = self.llm.messages.create(model=self.model_name, system=self.system, messages=self.messages, max_tokens=self.max_tokens, temperature=self.temperature).content[0].text
        elif "deepseek" in self.model_name.lower():
            vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            # vLLM supports most of the OpenAI parameters, but check compatibility
            # Extra parameters can be provided if your vLLM server supports them
            response = self.llm.chat.completions.create(
                model=vllm_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.8
                # Note: repetition_penalty may not be directly supported by vLLM
                # If needed, check vLLM docs for equivalent parameters
            )
            full_response = response.choices[0].message.content
        elif "qwen" in self.model_name.lower():
            print(self.model_name.lower())
            print(os.environ.get("VLLM_MODEL_NAME"))
            vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct")
            # vLLM supports most of the OpenAI parameters, but check compatibility
            # Extra parameters can be provided if your vLLM server supports them
            response = self.llm.chat.completions.create(
                model=vllm_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.8
                # Note: repetition_penalty may not be directly supported by vLLM
                # If needed, check vLLM docs for equivalent parameters
            )
            full_response = response.choices[0].message.content
        elif "s1" in self.model_name.lower():
            print(self.model_name.lower())
            print(os.environ.get("VLLM_MODEL_NAME"))
            vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "simplescaling/s1-32B")
            # vLLM supports most of the OpenAI parameters, but check compatibility
            # Extra parameters can be provided if your vLLM server supports them
            response = self.llm.chat.completions.create(
                model=vllm_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.8
                # Note: repetition_penalty may not be directly supported by vLLM
                # If needed, check vLLM docs for equivalent parameters
            )
            full_response = response.choices[0].message.content
        elif "openthinker" in self.model_name.lower():
            vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "open-thoughts/OpenThinker-7B")
            # vLLM supports most of the OpenAI parameters, but check compatibility
            # Extra parameters can be provided if your vLLM server supports them
            response = self.llm.chat.completions.create(
                model=vllm_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.8
                # Note: repetition_penalty may not be directly supported by vLLM
                # If needed, check vLLM docs for equivalent parameters
            )
            full_response = response.choices[0].message.content
        self.add_message(full_response, 'assistant')
        return full_response

    def parse_response(self, response, is_observation):
        if is_observation:
            pattern = r'<observe>(.*?)</observe>'   
        else:
            pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

        
    def prompt_llm_and_parse(self, request_prompt, is_observation, max_tries=10):
        used_retries = 0
        for i in range(max_tries):
            full_response = self.prompt_llm(request_prompt)
            print(full_response)
            response = self.parse_response(full_response, is_observation)
            # print(f"parsed response: {response}")
            if response is not None:
                # check if response has numbers
                numbers = re.findall(r'[0-9]+', response)
                if len(numbers) == 0:
                    response = None
            if response == None or "done" in response:
                # TODO: make this better
                if is_observation:
                    request_prompt = "Please stick to the specified format and respond using <observe> tags. Continue making observations even if you think you have an accurate estimate. Your previous response was not valid.Remember what the exact observation format was and just exactly follow that. Give a very short response and give your observation directly. This is your last chance to output the correct format for observation and do so concisely."
                else:
                    request_prompt = "Please stick to the specified format and respond using <answer> tags. Make assumptions and provide your best guess. Remember what the exact answer format was and just exactly follow that. Give a very short response and give your answer directly. This is your last chance to output the correct format for the answer and do so concisely. You have very few tokens to respond with."
                used_retries += 1
            else:
                break
        if used_retries == max_tries:
            ValueError("Failed to get valid response")            
        return response, used_retries
    
    def generate_predictions(self, request_prompt):
        #request_prompt += "\nAnswer in the following format:\n<answer>ONLY a number with no explanations or text</answer>."
        request_prompt += "\nAnswer in the following format required from above :\n<answer>your answer</answer>."
        prediction, used_retries = self.prompt_llm_and_parse(request_prompt, False)
        self.messages = self.messages[:-2 * (used_retries+1)]  # Remove the last 2 messages
        return prediction
    
    def generate_actions(self, experiment_results=None):
        if experiment_results is None:
            follow_up_prompt = f"Think about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n<observe> your observation</observe>\nMake an observation now."
        else:
            follow_up_prompt = f"Result: {experiment_results}\nThink about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\nThought: <thought>\n<observe> your observation (remember the type of inputs accepted)</observe>"
        observe, used_retries = self.prompt_llm_and_parse(follow_up_prompt, True)
        return observe 
    
    def print_log(self):
        for entry in self.messages:
            print("Message Type:", type(entry).__name__)
            print("Content:", entry.content)
            print("------")
