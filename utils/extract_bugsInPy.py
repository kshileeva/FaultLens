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

# -----------------------------
# Parsers (bug.info + patch hunks)
# -----------------------------

INFO_RE = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*;?\s*$')
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

GIT_URL_RE = re.compile(
    r"(https?://[^\s\"']+\.git|git@[^:\s]+:[^\s\"']+\.git|https?://github\.com/[^\s\"']+|https?://gitlab\.com/[^\s\"']+)"
)

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


def parse_bug_info(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = INFO_RE.match(line)
        if m:
            d[m.group(1)] = m.group(2)
    return d


def parse_bug_patch(patch_text: str) -> Dict[str, List[Hunk]]:
    """
    Returns: file_path -> list of hunks
    For buggy-region labeling we use OLD ranges (-old_start,old_len).
    """
    out: Dict[str, List[Hunk]] = {}
    cur_file: Optional[str] = None

    for line in patch_text.splitlines():
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


# -----------------------------
# Repo URL discovery (project.info)
# -----------------------------

def extract_repo_url(project_info_path: Path) -> Optional[str]:
    txt = project_info_path.read_text(encoding="utf-8", errors="replace")
    m = GIT_URL_RE.search(txt)
    if not m:
        return None
    url = m.group(1)

    # Normalize GitHub/GitLab web URLs to .git if needed (still works without .git for most hosts)
    if url.startswith("http") and url.endswith("/"):
        url = url[:-1]
    return url


# -----------------------------
# Git plumbing
# -----------------------------

def ensure_project_repo(repo_url: str, dest_dir: Path) -> None:
    """
    Clone if missing; otherwise fetch.
    """
    if dest_dir.exists() and (dest_dir / ".git").exists():
        run(["git", "fetch", "--all", "--prune"], cwd=str(dest_dir), check=True)
        return
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--no-tags", repo_url, str(dest_dir)], check=True)
    # fetch full history (some repos shallow by default; clone above isn't shallow, but just in case)
    run(["git", "fetch", "--unshallow"], cwd=str(dest_dir), check=False)


def git_show_file(repo_dir: Path, commit: str, file_path: str) -> Optional[str]:
    p = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        return None
    return p.stdout


# -----------------------------
# Snippet extraction + mapping
# -----------------------------

def extract_window(
    lines: List[str],
    start_1: int,
    end_1: int,
    pad: int = 30,
    rng: Optional[random.Random] = None,
    jitter: int = 0,
) -> Tuple[str, int]:
    """
    Extract a snippet window around [start_1, end_1] with optional random jitter.

    Without jitter: window is [start_1-pad, end_1+pad].
    With jitter: window start/end are expanded by an additional random amount in [0, jitter].
    This avoids constant snippet-local start lines (e.g., always 31 when pad=30).
    """
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
    """Merge overlapping or near-adjacent regions. Two regions merge if next.start <= cur.end + merge_gap."""
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


# -----------------------------
# JSONL output
# -----------------------------

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
            "Run the relevant test(s) and confirm the failure location in the stack trace.",
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
    """
    If appending, avoid duplicate ids by scanning last line.
    IDs we emit are: bugsinpy_<project>_<bug>_<hunkidx>__g<global>
    This returns the next global integer.
    """
    if not out_jsonl.exists() or out_jsonl.stat().st_size == 0:
        return 1
    last = None
    with out_jsonl.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        # read backwards until newline
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


# -----------------------------
# Main extraction
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugsinpy_root", required=True, help="Path to FaultLens/BugsInPy")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (will APPEND)")
    ap.add_argument("--project", default="", help="Optional: only process one project (e.g., youtube-dl)")
    ap.add_argument("--pad_hunk", type=int, default=10)
    ap.add_argument("--pad_window", type=int, default=30)
    ap.add_argument("--window_jitter", type=int, default=30, help="Extra random context (0..window_jitter) added to snippet window on both sides")
    ap.add_argument("--merge_gap", type=int, default=15, help="Merge hunks in same file if within this many lines")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for window jitter")
    ap.add_argument("--skip_test_files", action="store_true", help="Skip files under tests/ or test/")
    ap.add_argument("--max_region", type=int, default=60)
    ap.add_argument("--include_tests", action="store_true", help="(Optional) later: capture error_text by running run_test.sh (not implemented here)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    bugsinpy_root = Path(args.bugsinpy_root).resolve()
    projects_root = bugsinpy_root / "projects"
    out_jsonl = Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    global_idx = get_next_global_index(out_jsonl)

    # Where we clone actual upstream repos temporarily
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        projects = sorted([p for p in projects_root.iterdir() if p.is_dir()])
        if args.project:
            projects = [p for p in projects if p.name == args.project]
            if not projects:
                raise RuntimeError(f"Project not found: {args.project}")

        total_written = 0

        with out_jsonl.open("a", encoding="utf-8") as out_f:
            for proj_dir in projects:
                project = proj_dir.name
                project_info = proj_dir / "project.info"
                bugs_dir = proj_dir / "bugs"

                if not project_info.exists() or not bugs_dir.exists():
                    continue

                repo_url = extract_repo_url(project_info)
                if not repo_url:
                    print(f"[SKIP] {project}: no repo URL found in project.info")
                    continue

                repo_dir = tmp_root / f"{project}_repo"
                print(f"[CLONE/FETCH] {project} from {repo_url}")
                ensure_project_repo(repo_url, repo_dir)

                bug_ids = sorted([b for b in bugs_dir.iterdir() if b.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else x.name)

                for bug_folder in bug_ids:
                    bug_id = bug_folder.name
                    info_path = bug_folder / "bug.info"
                    patch_path = bug_folder / "bug_patch.txt"
                    if not info_path.exists() or not patch_path.exists():
                        continue

                    info = parse_bug_info(info_path)
                    buggy_commit = info.get("buggy_commit_id")
                    fixed_commit = info.get("fixed_commit_id")
                    test_file = info.get("test_file", "")
                    py_ver = info.get("python_version", "")

                    if not buggy_commit or not fixed_commit:
                        continue

                    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
                    file_hunks = parse_bug_patch(patch_text)

                    # One sample per (merged) region (snippet-first)
                    region_idx = 0
                    for file_path, hunks in file_hunks.items():
                        if not file_path.endswith(".py"):
                            continue

                        if args.skip_test_files:
                            norm = file_path.replace("\\", "/")
                            if norm.startswith("tests/") or norm.startswith("test/") or "/tests/" in norm or "/test/" in norm:
                                continue

                        content = git_show_file(repo_dir, buggy_commit, file_path)
                        if content is None:
                            continue
                        lines = content.splitlines()
                        if not lines:
                            continue

                        # Build expanded regions from hunks in buggy file (OLD ranges)
                        raw_regions: List[Tuple[int, int]] = []
                        for h in hunks:
                            if h.old_len == 0:
                                continue  # pure insertion in fixed version
                            old_start = h.old_start
                            old_end = h.old_start + max(1, h.old_len) - 1

                            gold_start = max(1, old_start - args.pad_hunk)
                            gold_end = min(len(lines), old_end + args.pad_hunk)

                            # cap region length
                            if gold_end - gold_start + 1 > args.max_region:
                                gold_end = gold_start + args.max_region - 1

                            raw_regions.append((gold_start, gold_end))

                        if not raw_regions:
                            continue

                        merged = merge_regions(raw_regions, merge_gap=args.merge_gap)

                        for gold_start, gold_end in merged:
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

                            ex_id = f"bugsinpy_{project}_{bug_id}_{region_idx:03d}__g{global_idx:06d}"
                            global_idx += 1

                            reason = (
                                f"BugsInPy real bug. Fix modifies this region in {file_path} "
                                f"(buggy {buggy_commit[:7]} -> fixed {fixed_commit[:7]})."
                            )
                            metadata = {
                                "dataset": "BugsInPy",
                                "project": project,
                                "bug_id": bug_id,
                                "file_path": file_path,
                                "buggy_commit_id": buggy_commit,
                                "fixed_commit_id": fixed_commit,
                                "test_file": test_file,
                                "python_version": py_ver,
                            }

                            rec = make_record(
                                ex_id=ex_id,
                                language="python",
                                code_text=snippet,
                                start_line=local_start,
                                end_line=local_end,
                                reason=reason,
                                metadata=metadata,
                                error_text=None,  # optional: add later via run_test.sh
                            )
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            total_written += 1

                # delete cloned repo after project done
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)

        print(f"Done. Wrote {total_written} JSONL examples to {out_jsonl}")


if __name__ == "__main__":
    main()
