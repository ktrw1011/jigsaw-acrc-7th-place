from functools import partial
from pathlib import Path
from typing import Annotated

import datasets
import pandas as pd
import typer
import unsloth
import wandb
from jigsaw.data import drop_ambiguous_label
from jigsaw_llm.config import BaseConfig
from jigsaw_llm.model import init_model
from jigsaw_llm.prompt.prompt import PromptVer9
from jigsaw_llm.train import prepare_dataset
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

print("unsloth version: ", unsloth.__version__)


class Config(BaseConfig): ...


INPUT_DIR = Path("input")
EXP_VERSION = "1.0.1"


def main(
    cfg_path: Annotated[Path, typer.Argument()],
    fold: Annotated[int, typer.Argument()],
    disable_tracking: Annotated[bool, typer.Option("--disable-tracking", is_flag=True)] = False,
) -> None:
    cfg = Config.from_yaml(cfg_path)

    update_dict = {"fold": fold, "tracking": not disable_tracking}
    cfg = cfg.update_from_dict(update_dict)

    typer.echo(f"Using config: {cfg.model_dump_json(indent=2)}")
    output_dir = Path(f"output/{cfg.exp_name}-{EXP_VERSION}-fold{cfg.fold}")

    trainer_args_partial = partial(
        SFTConfig,
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        eval_strategy="no",
        prediction_loss_only=True,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        eval_delay=None,
        torch_empty_cache_steps=None,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        log_level="error",
        logging_dir=None,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_total_limit=1,
        seed=42,
        # bf16=True,
        # fp16=False,
        bf16_full_eval=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        # metric_for_best_model="loss",
        # greater_is_better=False,
        label_smoothing_factor=0.0,
        optim="paged_adamw_8bit",
        group_by_length=False,
        report_to="wandb" if cfg.tracking else "none",
        push_to_hub=False,
        resume_from_checkpoint=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        batch_eval_metrics=True,
        max_length=None,
        packing=False,
        padding_free=False,
        eval_packing=False,
        completion_only_loss=False,  # 内部の処理を外だししたのでここではFalse
        activation_offloading=False,
    )

    prompt_template = PromptVer9()

    # drop subreddit duplicates
    train_df = (
        pd.read_csv(INPUT_DIR.joinpath("jigsaw-fold/skf_wo_example_4folds.csv"))
        .drop_duplicates(subset=["rule", "body"], keep="first")
        .reset_index(drop=True)
    )
    train_df = drop_ambiguous_label(train_df)

    trn_df = train_df[train_df["fold"] != cfg.fold].copy()
    trn_df["prompt"] = trn_df.apply(
        lambda row: prompt_template.to_prompt(row),
        axis=1,
    )
    trn_df["completion"] = trn_df["rule_violation"].map(lambda x: "Yes" if bool(x) else "No")
    trn_df = trn_df.sample(frac=1).reset_index(drop=True)  # shuffle

    trn_ds = datasets.Dataset.from_pandas(
        trn_df[["prompt", "completion"]],
    )

    trn_ds = trn_ds.map(
        lambda x: {
            "prompt": [{"role": "user", "content": x["prompt"]}],
            "completion": [{"role": "assistant", "content": x["completion"]}],
        },
    )

    model, tokenizer = init_model(
        cfg.model_name_or_path,
        max_seq_length=1024,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
    )
    trn_ds = prepare_dataset(trn_ds, tokenizer)

    total_steps_per_epoch = (len(trn_ds) // cfg.train_batch_size) // cfg.gradient_accumulation_steps
    max_steps = int(total_steps_per_epoch * cfg.num_train_epochs)

    trainer_args = trainer_args_partial(
        max_steps=max_steps,
        save_steps=int(max_steps * cfg.save_steps_ratio),
        eval_steps=int(max_steps * cfg.eval_steps_ratio),
    )

    exp_name = f"{cfg.exp_name}-{EXP_VERSION}-fold{cfg.fold}"
    if cfg.tracking:
        wandb.init(project="kaggle_jigsaw_acrc", name=exp_name)

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=trn_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,  # pyright: ignore
            completion_only_loss=True,
        ),
    )

    trainer.train()
    cfg.save(output_dir)


if __name__ == "__main__":
    typer.run(main)
