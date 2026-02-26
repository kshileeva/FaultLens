from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CODE_RE = re.compile(r"<CODE>\n(.*?)\n</CODE>", re.DOTALL)
ERROR_RE = re.compile(r"<ERROR>\n(.*?)\n</ERROR>", re.DOTALL)


def extract_code(user_content: str) -> Optional[str]:
    m = CODE_RE.search(user_content)
    if not m:
        return None
    return m.group(1)


def count_code_lines(code: str) -> int:
    # code extracted without the outer <CODE> tags; treat splitlines() count as line count
    lines = code.splitlines()
    # If code ends with trailing newline in the original, splitlines() is still correct for line numbers.
    return len(lines)


def parse_assistant_json(assistant_content: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(assistant_content)
    except Exception:
        return None


def is_flattened(code: str) -> bool:
    """
    Heuristic for broken synth entries: code is a single line but contains multiple constructs.
    """
    lines = code.splitlines()
    if len(lines) != 1:
        return False
    s = lines[0]
    # if it's long and has several statement separators/keywords, it's likely flattened.
    score = 0
    score += s.count(";")
    score += s.count("{")
    score += s.count("}")
    score += len(re.findall(r"\b(def|class|function|import|return|if|for|while)\b", s))
    return (len(s) >= 120 and score >= 3) or (score >= 6)


def validate_record(
    obj: Dict[str, Any],
    drop_negatives: bool,
    drop_flattened: bool,
) -> Tuple[bool, str]:
    """
    Returns (keep?, reason_if_dropped_or_ok_label).
    """
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, "bad_messages"

    user_msg = next((m for m in msgs if m.get("role") == "user"), None)
    assistant_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not user_msg or not assistant_msg:
        return False, "missing_user_or_assistant"

    user_content = user_msg.get("content", "")
    assistant_content = assistant_msg.get("content", "")

    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        return False, "nonstring_content"

    code = extract_code(user_content)
    if code is None:
        return False, "missing_code_block"

    if drop_flattened and is_flattened(code):
        return False, "flattened_code"

    nlines = count_code_lines(code)
    if nlines <= 0:
        return False, "empty_code"

    payload = parse_assistant_json(assistant_content)
    if payload is None:
        return False, "assistant_not_json"

    suspects = payload.get("suspects")
    if suspects is None:
        return False, "missing_suspects"
    if not isinstance(suspects, list):
        return False, "suspects_not_list"

    if drop_negatives and len(suspects) == 0:
        return False, "negative_dropped"

    # If suspects are empty and we keep negatives, accept.
    if len(suspects) == 0:
        return True, "ok_negative"

    # Validate each suspect line range
    for s in suspects:
        if not isinstance(s, dict):
            return False, "suspect_not_object"
        st = s.get("start_line")
        en = s.get("end_line")
        if not isinstance(st, int) or not isinstance(en, int):
            return False, "suspect_line_not_int"
        if st < 1 or en < 1:
            return False, "suspect_line_lt1"
        if st > en:
            return False, "suspect_start_gt_end"
        if en > nlines:
            return False, f"suspect_out_of_bounds(nlines={nlines},end={en})"

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output cleaned JSONL")
    ap.add_argument("--drop_negatives", action="store_true", help="Drop entries where suspects == []")
    ap.add_argument("--drop_flattened", action="store_true", help="Drop likely-flattened one-line code blocks")
    ap.add_argument(
        "--only_keep_prefix",
        nargs="*",
        default=[],
        help="Only keep ids starting with any of these prefixes (e.g. ex_ bugsinpy_ bugsjs_). If empty, keep all.",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    kept = 0
    dropped = 0
    total = 0

    prefixes = list(args.only_keep_prefix)

    with inp.open("r", encoding="utf-8", errors="replace") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                stats["blank_line"] = stats.get("blank_line", 0) + 1
                continue

            try:
                obj = json.loads(line)
            except Exception:
                dropped += 1
                stats["invalid_json"] = stats.get("invalid_json", 0) + 1
                continue

            ex_id = obj.get("id", "")
            if prefixes:
                if not isinstance(ex_id, str) or not any(ex_id.startswith(p) for p in prefixes):
                    dropped += 1
                    stats["id_prefix_filtered"] = stats.get("id_prefix_filtered", 0) + 1
                    continue

            ok, reason = validate_record(
                obj,
                drop_negatives=args.drop_negatives,
                drop_flattened=args.drop_flattened,
            )

            if ok:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                stats[reason] = stats.get(reason, 0) + 1
            else:
                dropped += 1
                stats[reason] = stats.get(reason, 0) + 1

    print(f"Input:  {inp}")
    print(f"Output: {out}")
    print(f"Total lines read: {total}")
    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")
    print("Breakdown:")
    for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
