import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from utils import TokenUsage, gemini_chat_complete

import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

class Selecting:
    template = Template(
        """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
"""
    )

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        template: Template = template,
    ):
        self.model = model_name
        self.template = template
        self.token_usage = TokenUsage()

        cache = Cache(f"./CoMEM/results/diskcache/selecting_{model_name}")
        self._cached_chat = cache.memoize(name="chat_complete")(gemini_chat_complete)

    def __call__(self, instance) -> list[bool]:
        response = gemini_chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": self.template.render(
                        anchor=instance["anchor"],
                        candidates=instance["candidates"],
                    ),
                }
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=512,
            token_usage=self.token_usage,
        )

        content = (response.choices[0].message.content or "").strip()
        # Extract the first bracketed number
        match = re.search(r"\[(\d+)\]", content)
        if match:
            idx = int(match.group(1))
        else:
            idx = 0

        # Build binary predictions: exactly one True if a valid candidate index is given
        preds = [False] * len(instance["candidates"])
        if 1 <= idx <= len(instance["candidates"]):
            preds[idx - 1] = True
        return preds

    @property
    def cost(self):
        return 0.0


def load_instances(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath)
    groupby = list(
        df.groupby("id_left")[["record_left", "record_right", "label"]]
        .apply(lambda x: x.to_dict("list"))
        .to_dict()
        .items()
    )
    return [
        {
            "anchor": v["record_left"][0],
            "candidates": v["record_right"],
            "labels": [bool(l) for l in v["label"]],
        }
        for _, v in groupby
    ]


if __name__ == "__main__":
    dataset_files = sorted(Path("./CoMEM/data/sample_DBLP_ACM_top5.csv"))
    results = {}
    selector = Selecting()

    for file in dataset_files:
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        instances = load_instances(str(file))
        preds_lst = thread_map(selector, instances, max_workers=4)
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4, labels=[False, True]))
        print(confusion_matrix(labels[: len(preds)], preds))

        report = classification_report(
            labels[: len(preds)], preds, output_dict=True, labels=[False, True]
        )["True"]
        report.pop("support")
        results[dataset] = {k: v * 100 for k, v in report.items()}

    if len(results) > 1:
        results["mean"] = {
            "precision": sum(v["precision"] for v in results.values()) / len(results),
            "recall":    sum(v["recall"]    for v in results.values()) / len(results),
            "f1-score":  sum(v["f1-score"]  for v in results.values()) / len(results),
        }

    df_results = pd.DataFrame.from_dict(results, orient="index")
    print(df_results)
    print(df_results.to_csv(float_format="%.2f", index=True))
    print(f"Selecting token usage: {selector.token_usage}")