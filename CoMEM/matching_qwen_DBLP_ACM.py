import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from jinja2 import Template
from diskcache import Cache
from rich import print
import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

class TokenUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
    def __str__(self):
        return f"Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}"

def local_chat_complete(prompt, model, tokenizer, device, max_tokens=8, token_usage=None):
    """Direct inference via Hugging Face Transformers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=False,        # Use Greedy Decoding (stabler)
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    if token_usage is not None:
        token_usage.prompt_tokens += inputs.input_ids.shape[1]
        token_usage.completion_tokens += (output_tokens.shape[1] - inputs.input_ids.shape[1])
        
    return generated_text

class MatchingSQ:
    template = Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
Answer: """
    )

    def __init__(self, model, tokenizer, device, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.token_usage = TokenUsage()
        cache = Cache(f"./CoMEM/results/diskcache/matching_{model_name.replace('/', '_')}")
        self._cached_chat = cache.memoize(name="chat_complete")(self._direct_query)

    def _direct_query(self, prompt):
        return local_chat_complete(
            prompt, self.model, self.tokenizer, self.device, max_tokens=8, token_usage=self.token_usage
        )

    def score(self, instance) -> list[float]:
        scores = []
        for candidate in instance["candidates"]:
            prompt = self.template.render(record_left=instance["anchor"], record_right=candidate)
            target = self._cached_chat(prompt).strip().lower()
            scores.append(1 if "yes" in target else -1 if "no" in target else 0)
        return scores

    def pointwise_rank(self, instance) -> list:
        scores = self.score(instance)
        indexes = list(range(len(instance["candidates"])))
        sorted_indexes = [x for _, x in sorted(zip(scores, indexes), reverse=True)]
        
        # Return the actual alphanumeric candidate IDs if provided
        if "candidate_ids" in instance:
            return [instance["candidate_ids"][idx] for idx in sorted_indexes]
        return sorted_indexes