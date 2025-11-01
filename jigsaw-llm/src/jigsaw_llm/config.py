from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
    exp_name: str

    base_model_name_or_path: str
    base_tokenizer_name_or_path: str

    learning_rate: float
    train_batch_size: int
    eval_batch_size: int

    num_train_epochs: float
    eval_steps_ratio: float
    save_steps_ratio: float
    gradient_accumulation_steps: int

    fold: int = 0
    fold_path: str = ""
    tracking: bool = True
    load_in_4bit:bool = True
    load_in_8bit:bool = False
    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""

    def save(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with Path(output_dir).joinpath("exp_config.json").open("w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "Self":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def update_from_dict(self, update_dict: dict) -> "Self":
        """辞書から設定を更新して新しいインスタンスを返す"""
        current_config = self.model_dump()
        current_config.update(update_dict)
        return self.__class__(**current_config)

    def update_from_yaml(self, yaml_path: Path) -> "Self":
        """YAMLファイルから設定を更新して新しいインスタンスを返す"""
        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        return self.update_from_dict(yaml_config)
