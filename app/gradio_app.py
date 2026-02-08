# Demo UI FaultLens: loads examples/gold.jsonl and renders outputs
import json
import re
from pathlib import Path
from typing import Any, Optional
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes app/ folder)
EXAMPLES_PATH = ROOT / "examples" / "gold.jsonl"

def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not path.exists():
        return examples

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


EXAMPLES = load_examples(EXAMPLES_PATH)
EXAMPLE_BY_ID = {ex["id"]: ex for ex in EXAMPLES}
DROPDOWN_CHOICES = [(f'{ex["id"]} — {ex.get("title","")}'.strip(" —"), ex["id"]) for ex in EXAMPLES]

# Guardrails (very simple)

_LEAK_PATTERNS = [
    r"```",
    r"diff\s+--git",
    r"^\s*[+-]\s",
    r"\breplace\b",
    r"\bchange\b.*\bto\b",
    r"\bedit\b.*\bline\b",
]

def looks_like_a_fix(text: str) -> bool:
    t = text.lower()
    for p in _LEAK_PATTERNS:
        if re.search(p, t, flags=re.MULTILINE):
            return True
    return False


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))


def add_code_highlight(code_text: str, start_line: Optional[int], end_line: Optional[int]) -> str:
    lines = code_text.splitlines()
    # 1-based inclusive
    s = start_line or -1
    e = end_line or -1
    # basic styling
    css = """
    <style>
      .codewrap { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                  font-size: 13px; line-height: 1.45; background: #0b0f14; color: #e6edf3;
                  padding: 12px; border-radius: 10px; overflow-x: auto; }
      .row { display: grid; grid-template-columns: 56px 1fr; gap: 12px; padding: 1px 0; }
      .ln { color: #7d8590; text-align: right; user-select: none; }
      .hl { background: rgba(255, 255, 255, 0.10); border-radius: 6px; padding: 0 4px; }
      .muted { color: #7d8590; }
    </style>
    """
    body = ['<div class="codewrap">']
    if not lines:
        body.append('<div class="muted">No code provided.</div></div>')
        return css + "\n".join(body)

    for i, raw in enumerate(lines, start=1):
        safe = html_escape(raw) if raw != "" else "&nbsp;"
        is_hl = (s <= i <= e) if (s != -1 and e != -1) else False
        code_cell = f'<span class="hl">{safe}</span>' if is_hl else safe
        body.append(f'<div class="row"><div class="ln">{i}</div><div>{code_cell}</div></div>')
    body.append("</div>")
    return css + "\n".join(body)


def format_analysis(gold: dict[str, Any]) -> str:
    suspects = gold.get("suspects", []) or []
    steps = gold.get("debug_next_steps", []) or []
    no_fix = (gold.get("no_fix_policy") or {}).get("explanation", "")

    if suspects:
        s0 = suspects[0]
        header = f"Suspect: lines {s0.get('start_line')}-{s0.get('end_line')} · conf {float(s0.get('confidence', 0.0)):.2f} · {s0.get('category','')}"
        why = f"Why: {s0.get('reason','')}"
    else:
        header = "Suspect: (none)"
        why = "Why: (no suspect region produced)"

    next_steps = "\n".join([f"- {s}" for s in steps[:3]])
    # policy = f"Policy: {no_fix}" if no_fix else ""
    policy = None

    parts = [header, why]
    if next_steps:
        parts.append("Next checks:\n" + next_steps)
    if policy:
        parts.append(policy)
    return "\n\n".join(parts).strip()


def load_example(example_id: str) -> tuple[str, str, str, str, str, str]:
    ex = EXAMPLE_BY_ID.get(example_id)
    if not ex:
        return "", "", "", add_code_highlight("", None, None), "", ""

    code_text = ex.get("code_text", "")
    error_text = ex.get("error_text", "") or ""
    gold = ex.get("gold", {})

    analysis = format_analysis(gold)

    # highlight first suspect
    s0 = (gold.get("suspects") or [{}])[0]
    start_line = s0.get("start_line")
    end_line = s0.get("end_line")

    code_html = add_code_highlight(code_text, start_line, end_line)
    conf = f"{float(s0.get('confidence', 0.0) or 0.0):.2f}"
    cat = str(s0.get("category", "") or "")
    return code_text, error_text, analysis, code_html, conf, cat


def run_demo(code_text: str, error_text: str, mode: str, example_id: str) -> tuple[str, str, str, str]:
    """
    Demo behaviors:
    - If mode == "Use gold (example)" and an example is selected, show its gold output.
    - Else, do a tiny heuristic: if error mentions a line number, highlight that ±3 lines.
    """
    analysis = ""

    start_line = None
    end_line = None
    conf = 0.55
    cat = "unknown"
    reason = "No strong signal found; using a generic suspicious-region guess."

    if mode == "Use gold (example)" and example_id in EXAMPLE_BY_ID:
        gold = EXAMPLE_BY_ID[example_id].get("gold", {})
        analysis = format_analysis(gold)
        s0 = (gold.get("suspects") or [{}])[0]
        start_line = s0.get("start_line")
        end_line = s0.get("end_line")
        conf = float(s0.get("confidence", conf) or conf)
        cat = str(s0.get("category", cat) or cat)
        code_html = add_code_highlight(code_text, start_line, end_line)
        conf_text = f"{conf:.2f}"
        return analysis, code_html, conf_text, cat

    # Heuristic mode (for arbitrary pasted snippets)
    # Try to find "line X" in error output
    m = re.search(r"\bline\s+(\d+)\b", error_text.lower())
    if m:
        ln = int(m.group(1))
        start_line = max(1, ln - 3)
        end_line = ln + 3
        conf = 0.65
        cat = "compilation_or_syntax" if ("syntax" in error_text.lower() or "indent" in error_text.lower()) else "unknown"
        reason = "Error output references a line number; focusing on a small window around it."
    else:
        # Fallback: highlight nothing, but still show guidance
        reason = "No line reference in the error output; consider providing a stack trace with line numbers."

    # Guardrail: never output fixes (this demo only outputs reasoning, so OK)
    if looks_like_a_fix(reason):
        reason = "I can indicate where to look and how to debug, but I will not provide the code change."

    if start_line:
        analysis = f"Suspect: lines {start_line}-{end_line} · conf {conf:.2f} · {cat}\n\nWhy: {reason}\n\nNext checks:\n- Re-run and capture a stack trace with file/line info.\n- Reduce to a minimal failing input.\n- Add assertions around the suspicious block.\n\nPolicy: I can indicate where to look and how to debug, but I will not provide the code change."
    else:
        analysis = f"Suspect: (no line reference found)\n\nWhy: {reason}\n\nNext checks:\n- Re-run and capture a stack trace with file/line info.\n- Reduce to a minimal failing input.\n- Add assertions around the suspicious block.\n\nPolicy: I can indicate where to look and how to debug, but I will not provide the code change."

    code_html = add_code_highlight(code_text, start_line, end_line)
    return analysis, code_html, f"{conf:.2f}", cat


def build_app() -> gr.Blocks:
    with gr.Blocks(title="FaultLens Demo", css="footer{display:none !important;}") as demo:
        gr.Markdown("## FaultLens \nPaste a snippet + optional error output, or load a gold example.")

        with gr.Row():
            # Left: controls + analysis
            with gr.Column(scale=6):
                with gr.Row():
                    example_dd: Any = gr.Dropdown(
                        choices=DROPDOWN_CHOICES,
                        label="Gold example",
                        value=DROPDOWN_CHOICES[0][1] if DROPDOWN_CHOICES else None,
                    )

                mode = gr.Radio(
                    ["Use gold (example)", "Heuristic (from error output)"],
                    value="Use gold (example)",
                    label="Run mode",
                )

                code_in = gr.Textbox(label="Code snippet / single file", lines=14, placeholder="Paste code here…")
                err_in = gr.Textbox(label="Error output (optional)", lines=6, placeholder="Paste stack trace / compiler error here…")

                run_btn: Any = gr.Button("Run", variant="primary")

            # Right: code viewer + metadata
            with gr.Column(scale=6):
                conf_out = gr.Textbox(label="Confidence", value="", interactive=False)
                cat_out = gr.Textbox(label="Category", value="", interactive=False)
                code_view = gr.HTML(label="Highlighted code")
                analysis_out = gr.Textbox(label="Analysis", lines=10, interactive=False)

        # Load example wiring
        gr.on(
            triggers=example_dd.change,
            fn=load_example,
            inputs=[example_dd],
            outputs=[code_in, err_in, analysis_out, code_view, conf_out, cat_out],
        )

        # Run wiring
        gr.on(
            triggers=run_btn.click,
            fn=run_demo,
            inputs=[code_in, err_in, mode, example_dd],
            outputs=[analysis_out, code_view, conf_out, cat_out],
        )

        # Auto-load first example on start (if exists)
        if DROPDOWN_CHOICES:
            gr.on(
                triggers=demo.load,
                fn=load_example,
                inputs=[example_dd],
                outputs=[code_in, err_in, analysis_out, code_view, conf_out, cat_out],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
