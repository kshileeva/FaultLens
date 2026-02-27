from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are FaultLens, an assistant that localizes bugs in code but never provides code fixes. "
    "You must output ONLY valid JSON matching the expected bug_localization schema."
)
USER_SUFFIX = "\n\nLocate the bug region (do not provide a fix). Output JSON only."

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TEST_PATH_HINT_RE = re.compile(r"(^|/)(test|tests)(/|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
    """Run a command and return combined stdout/stderr as text.

    Some tools (notably `git diff --no-index`) may emit bytes that are not valid UTF-8.
    We decode using UTF-8 with replacement to avoid UnicodeDecodeError.
    """
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def parse_unified_diff(diff_text: str) -> Dict[str, List[Hunk]]:
    out: Dict[str, List[Hunk]] = {}
    cur_file: Optional[str] = None

    for line in diff_text.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            cur_file = m.group(2)  # b/path
            out.setdefault(cur_file, [])
            continue

        if line.startswith("Binary files "):
            cur_file = None
            continue

        hm = HUNK_RE.match(line)
        if hm and cur_file:
            old_start = int(hm.group(1))
            old_len = int(hm.group(2) or "1")
            new_start = int(hm.group(3))
            new_len = int(hm.group(4) or "1")
            out[cur_file].append(Hunk(old_start, old_len, new_start, new_len))

    return out


def extract_window(
    lines: List[str],
    start_1: int,
    end_1: int,
    pad: int = 30,
    rng: Optional[random.Random] = None,
    jitter: int = 0,
) -> Tuple[str, int]:
    n = len(lines)
    extra_left = rng.randint(0, jitter) if (rng and jitter > 0) else 0
    extra_right = rng.randint(0, jitter) if (rng and jitter > 0) else 0
    win_start = max(1, start_1 - pad - extra_left)
    win_end = min(n, end_1 + pad + extra_right)
    snippet = "\n".join(lines[win_start - 1 : win_end]) + "\n"
    return snippet, win_start


def clamp_region(start: int, end: int, nlines: int) -> Tuple[int, int]:
    start = max(1, min(start, nlines))
    end = max(1, min(end, nlines))
    if end < start:
        start, end = end, start
    return start, end


def merge_regions(regions: List[Tuple[int, int]], merge_gap: int = 15) -> List[Tuple[int, int]]:
    if not regions:
        return []
    regions = sorted(regions, key=lambda t: (t[0], t[1]))
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = regions[0]
    for s, e in regions[1:]:
        if s <= cur_e + merge_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def split_regions(regions: List[Tuple[int, int]], max_len: int) -> List[Tuple[int, int]]:
    if max_len <= 0:
        return regions
    out: List[Tuple[int, int]] = []
    for s, e in regions:
        if e < s:
            s, e = e, s
        cur = s
        while cur <= e:
            sub_end = min(e, cur + max_len - 1)
            out.append((cur, sub_end))
            cur = sub_end + 1
    return out


def normalize_noindex_path(path_from_diff: str, fixed_root: Path, buggy_root: Optional[Path] = None) -> str:
    s = path_from_diff.strip().replace("\\", "/")
    p = Path(s)

    for root in [fixed_root, buggy_root] if buggy_root else [fixed_root]:
        if not root:
            continue
        try:
            rel = p.relative_to(root)
            return rel.as_posix()
        except Exception:
            pass

    def tail_after_root(root: Path) -> Optional[str]:
        if not root:
            return None
        root_posix = Path(str(root)).as_posix()
        root_parts_a = tuple(root_posix.split("/"))
        root_parts_b = tuple(root_posix.lstrip("/").split("/"))

        s_posix = Path(s).as_posix()
        parts_a = tuple(s_posix.split("/"))
        parts_b = tuple(s_posix.lstrip("/").split("/"))

        def find_tail(parts: Tuple[str, ...], root_parts: Tuple[str, ...]) -> Optional[str]:
            if not root_parts:
                return None
            for i in range(0, len(parts) - len(root_parts) + 1):
                if parts[i : i + len(root_parts)] == root_parts:
                    tail = parts[i + len(root_parts) :]
                    return "/".join(tail)
            return None

        return (
            find_tail(parts_a, root_parts_a)
            or find_tail(parts_b, root_parts_a)
            or find_tail(parts_a, root_parts_b)
            or find_tail(parts_b, root_parts_b)
        )

    tail = tail_after_root(fixed_root)
    if tail:
        return tail
    if buggy_root:
        tail = tail_after_root(buggy_root)
        if tail:
            return tail

    return s


def is_test_file(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    return bool(TEST_PATH_HINT_RE.search(rel))


def make_record(
    ex_id: str,
    language: str,
    code_text: str,
    start_line: int,
    end_line: int,
    reason: str,
    metadata: Dict[str, Any],
    error_text: Optional[str] = None,
) -> Dict[str, Any]:
    user = f"<CODE>\n{code_text}</CODE>"
    inputs_used = ["code"]
    if error_text and error_text.strip():
        user += f"\n\n<ERROR>\n{error_text.strip()}\n</ERROR>"
        inputs_used.append("error_output")
    user += USER_SUFFIX

    assistant = {
        "task": "bug_localization",
        "language": language,
        "inputs_used": inputs_used,
        "suspects": [
            {
                "region_id": "R1",
                "start_line": start_line,
                "end_line": end_line,
                "confidence": 0.80,
                "category": "real_commit_hunk",
                "reason": reason,
            }
        ],
        "debug_next_steps": [
            "Run the triggering test(s) on the buggy revision and confirm failure location.",
            "Inspect values/bounds/types around the suspect region.",
            "Compare buggy vs fixed code in this region to identify the faulty assumption.",
        ],
        "no_fix_policy": {
            "provide_patch": False,
            "provide_exact_replacement": False,
            "explanation": "I can localize suspicious regions but will not provide code fixes.",
        },
    }

    return {
        "id": ex_id,
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }



def resolve_defects4j_exe(explicit: str = "") -> str:
    """Resolve the defects4j executable.

    - If `explicit` is provided, use it.
    - Else try $DEFECTS4J_BIN, then PATH (shutil.which).
    Raises a clear error if not found.
    """
    if explicit:
        return explicit
    env_bin = os.environ.get("DEFECTS4J_BIN", "").strip()
    if env_bin:
        return env_bin
    found = shutil.which("defects4j")
    if found:
        return found
    raise FileNotFoundError(
        "defects4j executable not found. "
        "Add <defects4j>/framework/bin to PATH, or set DEFECTS4J_BIN to the full path to the defects4j script."
    )


def get_next_global_index(out_jsonl: Path) -> int:
    """Return next global index for __g###### suffix when appending to an existing JSONL file."""
    if not out_jsonl.exists() or out_jsonl.stat().st_size == 0:
        return 1

    # Read the last non-empty line efficiently
    with out_jsonl.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0:
            pos -= 1
            f.seek(pos)
            b = f.read(1)
            if b == b"\n" and buf:
                break
            buf = b + buf

    last = buf.decode("utf-8", errors="replace").strip()
    if not last:
        return 1

    try:
        obj = json.loads(last)
        ex_id = obj.get("id", "")
        m = re.search(r"__g(\d+)$", ex_id)
        if m:
            return int(m.group(1)) + 1
    except Exception:
        pass
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (APPEND)")
    ap.add_argument("--project", required=True, help="Defects4J project id (e.g., Lang, Chart, Math)")
    ap.add_argument("--bug", default="", help="Single bug id (e.g., 1). If empty, process all active bug ids.")
    ap.add_argument("--pad_hunk", type=int, default=10)
    ap.add_argument("--pad_window", type=int, default=30)
    ap.add_argument("--window_jitter", type=int, default=30)
    ap.add_argument("--merge_gap", type=int, default=15)
    ap.add_argument("--max_region", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_test_files", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--defects4j", default="", help="Path to the defects4j executable (optional). If omitted, uses DEFECTS4J_BIN or PATH.")
    args = ap.parse_args()

    try:
        d4j = resolve_defects4j_exe(args.defects4j)
    except FileNotFoundError as e:
        # Print a friendlier message and re-raise
        raise SystemExit(str(e))

    out_jsonl = Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    global_idx = get_next_global_index(out_jsonl)

    # Bug list
    if args.bug:
        bug_ids = [args.bug]
    else:
        # defects4j bids -p <pid> prints active bug ids
        raw = run([d4j, "bids", "-p", args.project], check=True)
        bug_ids = [b.strip() for b in raw.split() if b.strip().isdigit()]

    total_written = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        with out_jsonl.open("a", encoding="utf-8") as out_f:
            for bid in bug_ids:
                buggy_dir = tmp_root / f"{args.project}_{bid}b"
                fixed_dir = tmp_root / f"{args.project}_{bid}f"

                for d in (buggy_dir, fixed_dir):
                    if d.exists():
                        shutil.rmtree(d, ignore_errors=True)
                    d.mkdir(parents=True, exist_ok=True)

                try:
                    run([d4j, "checkout", "-p", args.project, "-v", f"{bid}b", "-w", str(buggy_dir)], check=True)
                    run([d4j, "checkout", "-p", args.project, "-v", f"{bid}f", "-w", str(fixed_dir)], check=True)
                except Exception as e:
                    if args.verbose:
                        print(f"[WARN] checkout failed {args.project}-{bid}: {e}")
                    shutil.rmtree(buggy_dir, ignore_errors=True)
                    shutil.rmtree(fixed_dir, ignore_errors=True)
                    continue

                diff_text = run(
                    ["git", "diff", "--no-index", "--unified=0", str(buggy_dir), str(fixed_dir)],
                    check=False,
                )
                file_hunks = parse_unified_diff(diff_text)
                if args.verbose and not file_hunks:
                    print(f"[WARN] no diff hunks for {args.project}-{bid}")

                region_idx = 0
                for diff_b_path, hunks in file_hunks.items():
                    rel_path = normalize_noindex_path(diff_b_path, fixed_dir, buggy_root=buggy_dir)
                    rel_norm = rel_path.replace("\\", "/")
                    if rel_norm.startswith(".git/") or "/.git/" in rel_norm:
                        continue
                    if not rel_path.endswith(".java"):
                        continue
                    if args.skip_test_files and is_test_file(rel_path):
                        continue

                    buggy_file = buggy_dir / rel_path
                    if not buggy_file.exists() or buggy_file.is_dir():
                        continue

                    content = buggy_file.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    if not lines:
                        continue

                    raw_regions: List[Tuple[int, int]] = []
                    for h in hunks:
                        if h.old_len == 0:
                            continue
                        old_start = h.old_start
                        old_end = h.old_start + max(1, h.old_len) - 1
                        gold_start = max(1, old_start - args.pad_hunk)
                        gold_end = min(len(lines), old_end + args.pad_hunk)
                        if gold_end - gold_start + 1 > args.max_region:
                            gold_end = gold_start + args.max_region - 1
                        raw_regions.append((gold_start, gold_end))

                    if not raw_regions:
                        continue

                    merged = merge_regions(raw_regions, merge_gap=args.merge_gap)
                    merged = split_regions(merged, max_len=args.max_region)

                    for gold_start, gold_end in merged:
                        gold_start = max(1, min(gold_start, len(lines)))
                        gold_end = max(1, min(gold_end, len(lines)))
                        if gold_end < gold_start:
                            gold_start, gold_end = gold_end, gold_start

                        region_idx += 1
                        snippet, win_start = extract_window(
                            lines,
                            gold_start,
                            gold_end,
                            pad=args.pad_window,
                            rng=rng,
                            jitter=args.window_jitter,
                        )
                        snippet_lines = snippet.splitlines()
                        local_start = gold_start - win_start + 1
                        local_end = gold_end - win_start + 1
                        local_start, local_end = clamp_region(local_start, local_end, len(snippet_lines))

                        ex_id = f"defects4j_{args.project}_{bid}_{region_idx:03d}__g{global_idx:06d}"
                        global_idx += 1

                        reason = f"Defects4J real bug. Diff buggy vs fixed modifies this region in {rel_path}."
                        metadata = {
                            "dataset": "Defects4J",
                            "project": args.project,
                            "bug_id": bid,
                            "file_path": rel_path,
                            "checkout_versions": [f"{bid}b", f"{bid}f"],
                        }

                        rec = make_record(
                            ex_id=ex_id,
                            language="java",
                            code_text=snippet,
                            start_line=local_start,
                            end_line=local_end,
                            reason=reason,
                            metadata=metadata,
                            error_text=None,  # optional: later parse defects4j test output
                        )
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_written += 1

                shutil.rmtree(buggy_dir, ignore_errors=True)
                shutil.rmtree(fixed_dir, ignore_errors=True)

    print(f"Done. Wrote {total_written} examples to {out_jsonl}")


if __name__ == "__main__":
    main()
