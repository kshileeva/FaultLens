# Testing the trained FaultLens model

After training with Unsloth, save your adapter from the training script:

```python
# In your training script (e.g. after SFTTrainer.train())
model.save_pretrained("outputs/faultlens_lora")
tokenizer.save_pretrained("outputs/faultlens_lora")
```

## 1. Test on a JSONL file

Use a held-out slice of your data (or any JSONL with the same `messages` format):

```bash
# RunPod / GPU environment with Unsloth installed
python scripts/test_trained_model.py \
  --adapter_path /path/to/outputs/faultlens_lora \
  --test_file data/test.jsonl \
  --limit 50
```

With ground-truth comparison (line-range overlap):

```bash
python scripts/test_trained_model.py \
  --adapter_path /path/to/outputs/faultlens_lora \
  --test_file data/test.jsonl \
  --limit 100 \
  --compare_gold
```

## 2. Test on a single snippet (stdin)

```bash
echo 'def f():
    return 1 + "x"' | python scripts/test_trained_model.py --adapter_path /path/to/outputs/faultlens_lora
```

## 3. Use the model in the Gradio app

To drive the web UI with your trained model instead of the heuristic:

- Load the adapter in a separate process or in the app (see below).
- When the user clicks **Run**, build the same prompt (system + user with `<CODE>...</CODE>`), call the model, parse the JSON response, and pass `start_line` / `end_line` to `add_code_highlight()` and the analysis text to the analysis box.

Minimal integration pattern:

```python
# Optional: set FAULTLENS_ADAPTER to your saved adapter path
import os
from unsloth import FastLanguageModel

adapter_path = os.environ.get("FAULTLENS_ADAPTER")
if adapter_path:
    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_path, max_seq_length=4096, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
else:
    model, tokenizer = None, None

def run_with_model(code_text: str, error_text: str) -> tuple[str, int | None, int | None]:
    if model is None:
        return "Model not loaded.", None, None
    # Build messages, generate, parse JSON, return analysis + start_line + end_line
    # (Reuse logic from scripts/test_trained_model.py)
    ...
```

Then in `run_demo`, add a mode like `"Model (FaultLens)"` that calls `run_with_model` and uses the returned line range for highlighting.

## Requirements

- `unsloth` (and its dependencies) in the same environment where you run the test script or the app.
- GPU for inference (same as training; L40S 48GB is sufficient for 8B 4-bit).
