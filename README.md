# kaggle-jigsaw-acrc 7th place solution
- [kaggle writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/7th-place-solution)
- [submission notebook]()

## setup
- using `uv`

```bash
mkdir -p input output

uv sync --all-packages --extra unsloth --extra build
uv sync --all-packages --extra unsloth --extra build --extra compile

cd ./input && \
uv run kaggle competitions download -c jigsaw-agile-community-rules && \
unzip jigsaw-agile-community-rules.zip -d jigsaw-agile-community-rules && \
rm jigsaw-agile-community-rules.zip
```

## Generate fold
```
uv run jigsaw/src/jigsaw/gen_fold.py extend-df-fold input/jigsaw-agile-community-rules/train.csv 4 input/jigsaw-fold  
```

## Run Experiments
```bash
uv run jigsaw-llm/exp/exp_101.py jigsaw-llm/configs/exp100/qwen3-14b.yaml fold_idx
uv run jigsaw-llm/exp/exp_101.py jigsaw-llm/configs/exp100/phi-4.yaml fold_idx
uv run jigsaw-llm/exp/exp_101.py jigsaw-llm/configs/exp100/gemma2-9b.yaml fold_idx
uv run jigsaw-llm/exp/exp_104.py jigsaw-llm/configs/exp100/shieldgemma-9b.yaml fold_idx
uv run jigsaw-llm/exp/exp_106.py jigsaw-llm/configs/exp100/qwen3-8b-guard.yaml fold_idx
```