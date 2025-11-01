from transformers.models.gemma2 import Gemma2ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from unsloth import FastLanguageModel

VALID_MODEL = Gemma2ForCausalLM


def init_model(
    model_name_or_path: str,
    max_seq_length: int,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    ) -> tuple[VALID_MODEL, PreTrainedTokenizer]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=False,
        trust_remote_code=True,
        # dtype="float16",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        max_seq_length=max_seq_length,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer
