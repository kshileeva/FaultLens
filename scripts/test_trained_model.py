"""
Test a trained FaultLens model (Unsloth LoRA checkpoint).

Usage:
  # Test on a JSONL file (same format as train_sft.cleaned.jsonl); optional --limit
  python scripts/test_trained_model.py --adapter_path /path/to/saved/adapter --test_file data/test.jsonl [--limit 20]

  # Test on a single code snippet from stdin
  echo "def f(): return 1 + 'x'" | python scripts/test_trained_model.py --adapter_path /path/to/saved/adapter

  # With ground truth in test file, show simple overlap metric
  python scripts/test_trained_model.py --adapter_path /path/to/saved/adapter --test_file data/test.jsonl --compare_gold
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Unsloth is optional at import; we check when running.
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Reuse extraction from your existing utils if run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_RE = re.compile(r"<CODE>\n(.*?)\n</CODE>", re.DOTALL)


def extract_code(user_content: str) -> str | None:
    m = CODE_RE.search(user_content)
    if not m:
        return None
    return m.group(1)


def build_user_message(code: str) -> str:
    return (
        "<CODE>\n"
        + code
        + "\n</CODE>\n\nLocate the bug region (do not provide a fix). Output JSON only."
    )


SYSTEM_MSG = (
    "You are FaultLens, an assistant that localizes bugs in code but never provides code fixes. "
    "You must output ONLY valid JSON matching the expected bug_localization schema."
)


def parse_assistant_json(raw: str) -> dict | None:
    """Extract JSON from model output (may be wrapped in markdown or have trailing text)."""
    raw = raw.strip()
    # Try to find a JSON object
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


def load_model_and_tokenizer(adapter_path: str, max_seq_length: int = 4096, load_in_4bit: bool = True):
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is not installed. Install with: pip install unsloth")
    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    code: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
) -> dict | None:
    user_content = build_user_message(code)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the new part (assistant reply)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_assistant_json(generated)


def line_overlap(pred_start: int, pred_end: int, gold_start: int, gold_end: int) -> float:
    """Jaccard-like overlap on line ranges: |intersection| / |union|."""
    pred_set = set(range(pred_start, pred_end + 1))
    gold_set = set(range(gold_start, gold_end + 1))
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Test trained FaultLens model (Unsloth adapter)")
    ap.add_argument("--adapter_path", required=True, help="Path to saved Unsloth LoRA adapter (or merged model)")
    ap.add_argument("--test_file", type=Path, default=None, help="JSONL test file (same format as train_sft.cleaned.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of examples from test_file to run")
    ap.add_argument("--compare_gold", action="store_true", help="If test_file has gold assistant messages, compute line overlap")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    print("Loading model and tokenizer...", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(
        args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    if args.test_file and args.test_file.exists():
        # Run on test file
        examples = []
        with args.test_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        if args.limit:
            examples = examples[: args.limit]
        print(f"Running on {len(examples)} examples from {args.test_file}", file=sys.stderr)

        overlaps = []
        for i, ex in enumerate(examples):
            messages = ex.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            assistant_gold_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if not user_msg:
                print(f"  [{i+1}] Skip: no user message", file=sys.stderr)
                continue
            user_content = user_msg.get("content", "")
            code = extract_code(user_content)
            if not code:
                print(f"  [{i+1}] Skip: no <CODE> block", file=sys.stderr)
                continue

            result = run_inference(
                model, tokenizer, code,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            ex_id = ex.get("id", i + 1)

            if result and result.get("suspects"):
                s0 = result["suspects"][0]
                pred_start = s0.get("start_line")
                pred_end = s0.get("end_line")
                print(f"\n--- {ex_id} ---")
                print(f"Predicted: lines {pred_start}-{pred_end}  confidence={s0.get('confidence')}  category={s0.get('category')}")

                if args.compare_gold and assistant_gold_msg:
                    gold_content = assistant_gold_msg.get("content", "")
                    try:
                        gold_json = json.loads(gold_content)
                        gold_suspects = gold_json.get("suspects") or []
                        if gold_suspects:
                            g0 = gold_suspects[0]
                            gs, ge = g0.get("start_line"), g0.get("end_line")
                            ov = line_overlap(pred_start or 0, pred_end or 0, gs, ge)
                            overlaps.append(ov)
                            print(f"Gold:      lines {gs}-{ge}  overlap={ov:.3f}")
                    except json.JSONDecodeError:
                        pass
            else:
                print(f"\n--- {ex_id} --- (no valid JSON or empty suspects)")

        if overlaps:
            print(f"\nMean line-range overlap (n={len(overlaps)}): {sum(overlaps)/len(overlaps):.3f}", file=sys.stderr)

    else:
        # Single snippet from stdin
        code = sys.stdin.read()
        if not code.strip():
            print("No code on stdin and no --test_file given.", file=sys.stderr)
            sys.exit(1)
        result = run_inference(
            model, tokenizer, code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if result:
            print(json.dumps(result, indent=2))
            if result.get("suspects"):
                for s in result["suspects"]:
                    print(f"  -> Lines {s.get('start_line')}-{s.get('end_line')} (conf={s.get('confidence')})")
        else:
            print("Model did not return valid JSON.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
