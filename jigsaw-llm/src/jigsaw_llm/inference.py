import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob

from jigsaw_llm.prompt.prompt import PromptVer4, PromptVer8, PromptVer9

os.environ["VLLM_USE_V1"] = "0"


def to_logits(logprobs: list[Logprob]) -> np.ndarray:
    logit = np.zeros(2)
    for logprob in logprobs:
        if logprob.decoded_token == "No":  # noqa
            logit[0] = logprob.logprob
        elif logprob.decoded_token == "Yes":  # noqa
            logit[1] = logprob.logprob
        else:
            print("Unexpected token:", logprob.decoded_token)
    return logit


def main(
    csv_path: Annotated[Path, typer.Argument()],
    save_dir: Annotated[Path, typer.Argument()],
    vllm_param_path: Annotated[Path, typer.Argument()],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    with vllm_param_path.open() as f:
        vllm_params = json.load(f)

    print("Loading vLLM parameters from:", vllm_params)

    fold_idx = vllm_params.pop("fold_idx")
    prompt_version = vllm_params.pop("prompt_version")
    model_dir = vllm_params["model"]

    enable_lora = vllm_params.get("enable_lora", False)
    lora_path = vllm_params.pop("lora_path", None)
    if enable_lora and lora_path is None:
        raise ValueError("lora_path must be specified when enable_lora is True")

    if enable_lora:
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if prompt_version == 4:  # noqa
        prompt_template = PromptVer4()
    elif prompt_version == 8:  # noqa
        prompt_template = PromptVer8()
    elif prompt_version == 9:
        prompt_template = PromptVer9()
    else:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")

    val_df = pd.read_csv(csv_path)
    if "test.csv" not in str(csv_path):
        val_df = val_df[val_df["fold"] == fold_idx].reset_index(drop=True)

    # if prompt_version == 7:
    #     val_df["rule_context"] = val_df["rule"].map(load_rule_context(Path("input/jigsaw-fold/aug_rule_context.json")))


    val_df["prompt"] = val_df.apply(
        lambda row: prompt_template.to_prompt(row),
        axis=1,
    )

    llm = LLM(trust_remote_code=True, **vllm_params)

    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=2)

    batch_inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        for p in val_df["prompt"]
    ]

    vllm_outputs = llm.generate(
        batch_inputs,
        sp,
        use_tqdm=True,
        lora_request=LoRARequest("adapter", 1, lora_path) if enable_lora else None,
    )

    fold_logits = []
    for output in vllm_outputs:
        _logprobs = list(output.outputs[0].logprobs[-1].values())  # pyright: ignore
        fold_logits.append(to_logits(_logprobs))

    fold_logits = np.array(fold_logits)

    torch.save(fold_logits, save_dir.joinpath(f"fold{fold_idx}_logits.bin"))
    torch.save(val_df, save_dir.joinpath(f"fold{fold_idx}_val_df.bin"))


if __name__ == "__main__":
    typer.run(main)
