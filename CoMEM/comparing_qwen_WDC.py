from typing import Literal
from jinja2 import Template
from matching_qwen_WDC import MatchingSQ, local_chat_complete, TokenUsage

class ComparingSQ:
    template = Template(
        """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer with the corresponding record identifier "Record A" or "Record B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
Answer: """
    )

    def __init__(self, model, tokenizer, device, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.template = self.template
        self.matcher = MatchingSQ(model, tokenizer, device, model_name)
        self.token_usage = TokenUsage()

    def cmp(self, instance) -> int:
        def _query(anchor, cpair) -> str:
            prompt = self.template.render(anchor=anchor, cpair=cpair)
            return local_chat_complete(
                prompt, self.model, self.tokenizer, self.device, max_tokens=16, token_usage=self.token_usage
            )

        target1 = _query(instance["anchor"], instance["cpair"]).strip()
        target2 = _query(instance["anchor"], instance["cpair"][::-1]).strip()

        score = 0
        if "Record A" in target1: score += 1
        elif "Record B" in target1: score -= -1
        if "Record B" in target2: score += 1
        elif "Record A" in target2: score -= 1
        return score

    def pairwise_rank(self, instance, mode: Literal["bubble"] = "bubble", topK: int = 1) -> list[int]:
        indexes = list(range(len(instance["candidates"])))
        n = len(indexes)
        if mode == "bubble":
            for i in range(topK):
                for j in range(n - 1, i, -1):
                    greater = self.cmp({"anchor": instance["anchor"], "cpair": [instance["candidates"][indexes[j]], instance["candidates"][indexes[j - 1]]]})
                    if greater >= 0: indexes[j], indexes[j - 1] = indexes[j - 1], indexes[j]
        return indexes