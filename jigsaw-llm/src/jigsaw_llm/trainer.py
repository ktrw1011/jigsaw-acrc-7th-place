from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.data.data_collator import DataCollatorWithPadding, pad_without_fast_tokenizer_warning
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from trl import SFTConfig, SFTTrainer
from torch.utils.data import Dataset, SequentialSampler


@dataclass
class DistilSFTConfig(SFTConfig):
    token_ids: list[int] = field(default_factory=list)
    target_position: int = field(default=-1, metadata={"help": "Position of the target token in the sequence."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for knowledge distillation."})


class DistilSFTTrainer(SFTTrainer):
    args: DistilSFTConfig

    def _compute_kd_loss_full_vocab(
        self,
        attention_mask: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        全語彙でsoftmaxを計算してから特定トークンを抽出する版
        """
        batch_indices = torch.arange(student_logits.size(0), device=student_logits.device)
        seq_length = attention_mask.sum(dim=-1)
        logit_positions = seq_length + (self.args.target_position - 1)
        token_ids = torch.tensor(self.args.token_ids, device=student_logits.device)

        # 全語彙でlog_softmaxを計算
        full_student_logits = student_logits[batch_indices, logit_positions]  # [batch_size, vocab_size]
        full_student_log_probs = F.log_softmax(full_student_logits / self.args.temperature, dim=-1)

        # 特定トークンのlog_probsを抽出
        target_student_log_probs = full_student_log_probs[:, token_ids]  # [batch_size, num_target_tokens]

        # KD lossの計算
        kd_loss = -torch.sum(teacher_logits * target_student_log_probs, dim=-1).mean()

        return kd_loss

    def compute_loss(  # noqa
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculationg might be slightly inacurate when performing gradient accumulation.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            # TODO
            labels = inputs.pop("logits")
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()  # noqa
            else:
                model_name = unwrapped_model._get_name()  # noqa
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                # kd loss
                loss = self._compute_kd_loss_full_vocab(inputs["attention_mask"], outputs.logits, labels)
            else:
                loss = self.label_smoother(outputs, labels)  # pyright: ignore
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.",
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


class DistilSFTForUnslothCollator(DataCollatorWithPadding):
    """
    unsloth用のDataCollatorWithPadding
    labelsを供給するとunslothがlossを計算するのでlabelsではなくlogitsに変える
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        self.tokenizer.padding_side = "left"
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "logit" in batch:
            batch["logits"] = batch["logit"]
            del batch["logit"]
        return batch


class SequentialSamplerSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset: Dataset) -> torch.utils.data.Sampler:
        # Build the sampler.
        if self.args.group_by_length:
            raise NotImplementedError("group_by_length is not implemented in SequentialSamplerSFTTrainer")
        return SequentialSampler(train_dataset)