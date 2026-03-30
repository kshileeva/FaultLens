import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel


def load_train_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FaultLens with Unsloth Qwen3 (QLoRA SFT).")
    ap.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-8B-Base",
        help="Base model name (Unsloth-compatible Qwen3).",
    )
    ap.add_argument(
        "--train_file",
        type=Path,
        default=Path("data/train_sft.cleaned.jsonl"),
        help="JSONL train file (chat messages format; default: cleaned SFT export).",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/faultlens_qwen3_8b"),
        help="Directory to save LoRA adapter and tokenizer.",
    )
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit (QLoRA).",
    )
    args = ap.parse_args()

    if not args.train_file.exists():
        raise SystemExit(f"Train file not found: {args.train_file}")

    print(f"Loading train data from {args.train_file} ...")
    rows = load_train_jsonl(args.train_file)
    print(f"Loaded {len(rows)} examples.")

    # Hugging Face Dataset with original messages kept intact
    dataset = Dataset.from_list(rows)

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    # Unsloth patches eos_token to "<EOS_TOKEN>" which TRL rejects; reset to Qwen3's actual EOS token.
    tokenizer.eos_token = "<|im_end|>"

    print("Wrapping with LoRA (QLoRA).")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    system_prompt = (
     "You are FaultLens, an assistant that localizes bugs in code but never provides code fixes."
     "You must output ONLY valid JSON matching the expected bug_localization schema."
    )

    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        """Convert one chat-style row into Qwen3 ChatML format."""
        msgs = list(example["messages"])
        has_system = any(m.get("role") == "system" for m in msgs)
        if not has_system:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        text = ""
        for msg in msgs:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(format_example)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=args.seed,
        packing=False,
        dataset_text_field="text",
    )

    print("Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()

    print(f"Saving LoRA adapter and tokenizer to {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Done")


if __name__ == "__main__":
    main()
