from __future__ import annotations

import argparse
import csv
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

# diff --git a/... b/...  (works for both git diff and git diff --no-index)
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

TEST_PATH_HINT_RE = re.compile(r"(^|/)(test|tests|__tests__)(/|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def parse_unified_diff(diff_text: str) -> Dict[str, List[Hunk]]:
    """
    Returns: file_path -> hunks (OLD ranges are in buggy version)
    Works on `git diff --no-index` output too.
    """
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
    """
    In --no-index diffs, paths can be absolute or relative and typically include the temp roots:
      a/<buggy_root>/some/file.js
      b/<fixed_root>/some/file.js
    We want a path relative to the project root (so we can open buggy_root/rel_path).

    Strategy:
    - If path is absolute/relative and is under fixed_root, return relative_to(fixed_root).
    - Else if under buggy_root, return relative_to(buggy_root).
    - Else, try substring match on path parts: find the tail after the fixed_root (or buggy_root) parts.
    - Else, return the path as-is (normalized slashes).
    """
    s = path_from_diff.strip().replace("\\", "/")
    p = Path(s)

    # 1) Direct relative_to checks
    for root in [fixed_root, buggy_root] if buggy_root else [fixed_root]:
        if not root:
            continue
        try:
            rel = p.relative_to(root)
            return rel.as_posix()
        except Exception:
            pass

    # 2) Try to locate root path parts inside p.parts
    def tail_after_root(root: Path) -> Optional[str]:
        if not root:
            return None
        # Try matching both with and without leading '/'
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
                "confidence": 0.90 if error_text else 0.78,
                "category": "real_commit_hunk",
                "reason": reason,
            }
        ],
        "debug_next_steps": [
            "Run the failing test(s) and confirm the failure location in the stack trace.",
            "Inspect values/types/bounds around the suspect region.",
            "Compare with the fix to understand what assumption was violated.",
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


def get_next_global_index(out_jsonl: Path) -> int:
    if not out_jsonl.exists() or out_jsonl.stat().st_size == 0:
        return 1
    last = None
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
    try:
        obj = json.loads(last)
        ex_id = obj.get("id", "")
        m = re.search(r"__g(\d+)$", ex_id)
        if m:
            return int(m.group(1)) + 1
    except Exception:
        pass
    return 1



def is_test_file(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    return bool(TEST_PATH_HINT_RE.search(rel))


# Helper to locate the actual checked-out project root inside the BugsJS output directory.
def find_checkout_root(out_dir: Path) -> Path:
    """BugsJS checkout output often contains a single nested project folder (e.g., out_dir/bower/...).
    Return the most likely project root to diff/read from.

    Heuristics:
    - If out_dir itself contains a .git directory, return out_dir.
    - If out_dir has exactly one child directory, and that child (or a shallow descendant) contains .git, return that child.
    - Otherwise, search up to depth 3 for a directory containing .git and return the shallowest match.
    - Fallback: return out_dir.
    """
    if (out_dir / ".git").exists():
        return out_dir

    children = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        cand = children[0]
        if (cand / ".git").exists():
            return cand
        # shallow search under the single child
        for p in cand.rglob(".git"):
            return p.parent
        return cand

    # general shallow search
    best: Optional[Path] = None
    best_depth = 10**9
    for p in out_dir.rglob(".git"):
        root = p.parent
        try:
            rel = root.relative_to(out_dir)
            depth = len(rel.parts)
        except Exception:
            depth = 10**8
        if depth < best_depth:
            best = root
            best_depth = depth
    return best if best else out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugsjs_root", required=True, help="Path to BugsJS repo root (bug-dataset)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (APPEND)")
    ap.add_argument("--project", default="", help="Optional: only process one project (e.g., Bower)")
    ap.add_argument("--bug", default="", help="Optional: only one bug id (e.g., 1)")
    ap.add_argument("--pad_hunk", type=int, default=10)
    ap.add_argument("--pad_window", type=int, default=30)
    ap.add_argument("--window_jitter", type=int, default=30)
    ap.add_argument("--merge_gap", type=int, default=15)
    ap.add_argument("--max_region", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_test_files", action="store_true")
    ap.add_argument("--rewrite", action="store_true", help="Delete output file before writing")
    ap.add_argument("--verbose", action="store_true", help="Print checkout/diff diagnostics")
    args = ap.parse_args()

    bugsjs_root = Path(args.bugsjs_root).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.rewrite and out_jsonl.exists():
        out_jsonl.unlink()

    rng = random.Random(args.seed)
    global_idx = get_next_global_index(out_jsonl)

    # BugsJS layout: Projects/<ProjectName>/
    projects_dir = bugsjs_root / "Projects"
    if not projects_dir.exists():
        raise RuntimeError(f"Expected Projects/ under {bugsjs_root}")

    projects = sorted([p for p in projects_dir.iterdir() if p.is_dir()])
    if args.project:
        projects = [p for p in projects if p.name == args.project]
        if not projects:
            raise RuntimeError(f"Project not found: {args.project}")

    total_written = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        with out_jsonl.open("a", encoding="utf-8") as out_f:
            for proj in projects:
                project = proj.name
                # Determine bug ids: easiest is to call `main.py -t info` in a loop,
                # but BugsJS also has project_bugs.csv. We'll discover bug ids from that file if present.
                bugs_csv = proj / f"{project}_bugs.csv"
                bug_ids: List[str] = []
                if bugs_csv.exists():
                    # BugsJS CSVs are often ';' delimited (e.g., header starts with 'ID;...').
                    # Use csv.reader with delimiter detection from the header line.
                    raw = bugs_csv.read_text(encoding="utf-8", errors="replace").splitlines()
                    if not raw:
                        bug_ids = []
                    else:
                        header = raw[0]
                        delim = ";" if (";" in header and "," not in header) else ","
                        reader = csv.reader(raw, delimiter=delim)
                        rows = list(reader)
                        # Find the bug id column index (prefer 'ID')
                        id_idx = 0
                        if rows:
                            cols = [c.strip().strip("\ufeff") for c in rows[0]]
                            for j, c in enumerate(cols):
                                if c.strip().lower() in {"id", "bugid", "bug_id"}:
                                    id_idx = j
                                    break
                        for r in rows[1:]:
                            if not r or id_idx >= len(r):
                                continue
                            bid = r[id_idx].strip()
                            if bid.isdigit():
                                bug_ids.append(bid)
                else:
                    # fallback: try 1..200 and keep those that checkout succeeds (not ideal)
                    bug_ids = [str(i) for i in range(1, 201)]

                if args.bug:
                    bug_ids = [b for b in bug_ids if b == args.bug]
                    if not bug_ids:
                        raise RuntimeError(f"Bug id {args.bug} not found for {project}")

                if not bug_ids:
                    print(f"[WARN] {project}: no bug ids discovered from {bugs_csv.name}")

                for bug_id in bug_ids:
                    # Checkout buggy and fixed into temp dirs
                    buggy_dir = tmp_root / f"{project}_b{bug_id}_buggy"
                    fixed_dir = tmp_root / f"{project}_b{bug_id}_fixed"

                    for d in (buggy_dir, fixed_dir):
                        if d.exists():
                            shutil.rmtree(d, ignore_errors=True)
                        d.mkdir(parents=True, exist_ok=True)

                    # BugsJS CLI (from your README)
                    # python3 main.py -p Bower -b 1 -t checkout -v fixed -o output/
                    try:
                        run(["python3", "main.py", "-p", project, "-b", str(bug_id), "-t", "checkout", "-v", "buggy", "-o", str(buggy_dir)],
                            cwd=str(bugsjs_root),
                            check=True)
                        run(["python3", "main.py", "-p", project, "-b", str(bug_id), "-t", "checkout", "-v", "fixed", "-o", str(fixed_dir)],
                            cwd=str(bugsjs_root),
                            check=True)
                    except Exception as e:
                        if args.verbose:
                            print(f"[WARN] checkout failed for {project} bug {bug_id}: {e}")
                        continue

                    buggy_root = find_checkout_root(buggy_dir)
                    fixed_root = find_checkout_root(fixed_dir)
                    if args.verbose:
                        print(f"[DEBUG] buggy_root: {buggy_root}")
                        print(f"[DEBUG] fixed_root: {fixed_root}")
                    # diff buggy vs fixed without needing git history
                    diff_text = run(
                        ["git", "diff", "--no-index", "--unified=0", str(buggy_root), str(fixed_root)],
                        cwd=str(bugsjs_root),
                        check=False,
                    )
                    file_hunks = parse_unified_diff(diff_text)
                    if args.verbose and not file_hunks:
                        print(f"[WARN] no diff hunks for {project} bug {bug_id} (maybe checkout folders identical or diff format unexpected)")

                    # one sample per (merged+split) region per file
                    region_idx = 0
                    for diff_b_path, hunks in file_hunks.items():
                        rel_path = normalize_noindex_path(diff_b_path, fixed_root, buggy_root=buggy_root)
                        rel_norm = rel_path.replace("\\", "/")
                        if rel_norm.startswith(".git/") or "/.git/" in rel_norm:
                            continue
                        if args.verbose and region_idx == 0:
                            # show a couple mappings early
                            print(f"[DEBUG] diff path: {diff_b_path}")
                            print(f"[DEBUG] rel_path:  {rel_path}")
                            print(f"[DEBUG] buggy exists: {(buggy_root / rel_path).exists()}")
                        if not rel_path.endswith((".js", ".ts")):
                            continue
                        if args.skip_test_files and is_test_file(rel_path):
                            continue

                        buggy_file = buggy_root / rel_path
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

                            lang = "javascript" if rel_path.endswith(".js") else "typescript"
                            ex_id = f"bugsjs_{project}_{bug_id}_{region_idx:03d}__g{global_idx:06d}"
                            global_idx += 1

                            reason = f"BugsJS real bug. Diff buggy vs fixed modifies this region in {rel_path}."
                            metadata = {
                                "dataset": "BugsJS",
                                "project": project,
                                "bug_id": bug_id,
                                "file_path": rel_path,
                                "checkout_versions": ["buggy", "fixed"],
                            }

                            rec = make_record(
                                ex_id=ex_id,
                                language=lang,
                                code_text=snippet,
                                start_line=local_start,
                                end_line=local_end,
                                reason=reason,
                                metadata=metadata,
                                error_text=None,  # optional: add later by running BugsJS test command
                            )
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            total_written += 1

                    # cleanup per bug to keep disk low
                    shutil.rmtree(buggy_dir, ignore_errors=True)
                    shutil.rmtree(fixed_dir, ignore_errors=True)

    print(f"Done. Wrote {total_written} examples to {out_jsonl}")


if __name__ == "__main__":
    main()
