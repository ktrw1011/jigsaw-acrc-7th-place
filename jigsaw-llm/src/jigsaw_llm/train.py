import warnings

import datasets
from transformers.tokenization_utils import PreTrainedTokenizer
from trl.data_utils import apply_chat_template


def prepare_dataset(dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer) -> datasets.Dataset:
    def tokenize(example, processing_class, dataset_text_field, add_special_tokens):  # noqa
        if "prompt" in example:  # prompt-completion case
            processed_prompt = processing_class(
                text=example["prompt"],
                add_special_tokens=add_special_tokens,
            )
            processed = processing_class(
                text=example["prompt"] + example["completion"],
                add_special_tokens=add_special_tokens,
            )

            # Check if the tokenized prompt starts with the tokenized prompt+completion
            prompt_ids = processed_prompt["input_ids"]
            prompt_completion_ids = processed["input_ids"]
            if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:  # noqa
                warnings.warn(  # noqa
                    "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                    "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                    "token handling. Verify that the tokenizer is processing text consistently.",
                )

            # Create a completion mask
            completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
            processed = {**processed, "completion_mask": completion_mask}

        else:  # language modeling case
            processed = processing_class(
                text=example[dataset_text_field],
                add_special_tokens=add_special_tokens,
            )
        return processed

    first_example = next(iter(dataset))
    column_names = first_example.keys()  # pyright: ignore
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=column_names,  # pyright: ignore
    )
    add_special_tokens = False

    dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "processing_class": tokenizer,
            "dataset_text_field": "text",
            "add_special_tokens": add_special_tokens,
        },
    )

    return dataset
