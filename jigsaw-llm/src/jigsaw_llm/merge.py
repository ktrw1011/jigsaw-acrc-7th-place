from pathlib import Path
from typing import Annotated

import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from transformers.modeling_utils import PreTrainedModel


def merge(
    base_model_path: Annotated[Path, typer.Argument(..., help="Path to the base model")],
    adapter_path: Annotated[Path, typer.Argument(..., help="Path to the adapter model")],
    output_dir: Annotated[Path | None, typer.Option("-o", help="Output directory for the merged model")] = None,
    low_mem: Annotated[bool, typer.Option("-l", is_flag=True, help="Use low memory mode")] = False,
    shard_size: Annotated[str, typer.Option("-s", help="Shard size in MB for saving the model")] = "5GB",
    device: Annotated[str, typer.Option("-d", help="Device to load the model onto")] = "auto",
) -> None | PreTrainedModel:
    """Merge a PEFT adapter with a base model and save the merged model."""
    if "gemma-3" in str(base_model_path).lower():
        model_class = Gemma3ForCausalLM
        print(model_class)
    else:
        model_class = AutoModelForCausalLM  # type: ignore

    model = model_class.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype="auto",
        low_cpu_mem_usage=low_mem,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = model.merge_and_unload()  # pyright: ignore

    if output_dir is not None:
        model.save_pretrained(output_dir, max_shard_size=shard_size)
        tokenizer.save_pretrained(output_dir)
        return None
    return model


if __name__ == "__main__":
    typer.run(merge)