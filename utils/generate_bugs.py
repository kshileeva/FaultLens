import argparse
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Set


SYSTEM_PROMPT = (
    "You are FaultLens, an assistant that localizes bugs in code but never provides code fixes. "
    "You must output ONLY valid JSON matching the expected bug_localization schema."
)
USER_SUFFIX = "\n\nLocate the bug region (do not provide a fix). Output JSON only."

# Confidence ranges based on golden examples and actual generation
CONF_OBVIOUS_CRASH = (0.85, 0.95)      # Clear crashes like division by zero
CONF_SUBTLE_CRASH = (0.70, 0.84)       # Less obvious crashes
CONF_SILENT_BUG = (0.55, 0.80)         # Silent bugs
CONF_API_MISUSE = (0.55, 0.78)         # API misuse (wider range)
CONF_BOUNDS_INDEXING = (0.60, 0.95)    # bounds_or_indexing can be subtle (0.62) or obvious (0.92)
CONF_NEGATIVE = (0.20, 0.45)            # No bug

@dataclass(frozen=True)
class Template:
    tid: str
    language: str
    category: str
    kind: str  # "crash" | "silent" | "negative"
    lines: List[str]            # code lines (no <CODE> wrapper)
    bug_lines: List[int]         # 1-indexed lines that actually contain the bug (not comments)
    reason: str
    error_text: Optional[str] = None
    subtle: bool = False         # Whether this is a subtle bug variant


def get_confidence_range(t: Template) -> Tuple[float, float]:
    """Get appropriate confidence range based on bug type and subtlety."""
    if t.kind == "negative":
        return CONF_NEGATIVE
    elif t.category == "bounds_or_indexing":
        # bounds_or_indexing can be subtle (off-by-one) or obvious (empty array)
        if t.subtle:
            return (0.60, 0.75)  # Subtle bounds bugs
        else:
            return (0.80, 0.95)  # Obvious bounds bugs
    elif t.kind == "crash":
        if t.subtle or t.category in ["null_or_none", "api_misuse"]:
            return CONF_SUBTLE_CRASH
        else:
            return CONF_OBVIOUS_CRASH
    elif t.category == "api_misuse":
        return CONF_API_MISUSE
    else:  # silent
        return CONF_SILENT_BUG


NAME_POOLS = {
    "FN": ["compute", "calc", "process", "handle", "summarize", "parse", "analyze", "format", 
           "extract", "validate", "transform", "convert", "get", "find", "calculate", "build"],
    "XS": ["xs", "items", "arr", "nums", "values", "data", "list", "elements", "collection", 
           "sequence", "array", "numbers", "inputs"],
    "X": ["x", "v", "n", "item", "val", "num", "element", "current", "value", "elem"],
    "USER": ["user", "profile", "account", "u", "person", "customer", "client", "member"],
    "CFG": ["config", "cfg", "settings", "opts", "options", "params", "parameters", "props"],
    "KEY": ["timeout", "retries", "limit", "age", "path", "name", "id", "key", "value", "count"],
    "NAME": ["Alice", "Bob", "Mina", "Zoe", "Sam", "Ivy", "Noah", "Kim", "Alex", "Jordan", "Taylor", "Casey"],
    "N": ["0", "1", "2", "3", "5", "7", "10", "42", "100", "999", "-1"],
    "M": ["10", "20", "30", "60", "65", "100", "150", "255", "500"],
}


def apply_variations_keep_lines(rng: random.Random, lines: List[str]) -> List[str]:
    """Apply token variations while preserving line structure."""
    mapping = {k: rng.choice(v) for k, v in NAME_POOLS.items()}
    out = []
    for ln in lines:
        new_ln = ln
        for k, v in mapping.items():
            new_ln = new_ln.replace(f"{{{{{k}}}}}", v)
        out.append(new_ln)
    return out


def build_templates() -> List[Template]:
    """Build all templates with precise bug line indices."""
    T: List[Template] = []
    used_ids: Set[str] = set()

    # ---- Python ----
    T.append(Template(
        tid="py_int_str_concat",
        language="python",
        category="type_or_shape",
        kind="crash",
        lines=[
            "def {{FN}}_score(name, score):",
            "    # BUG: score is int but concatenated into a string",
            "    return name + ': ' + score",
            "",
            "print({{FN}}_score('{{NAME}}', 10))",
        ],
        bug_lines=[3],
        reason="Concatenates a string with an int without conversion, which raises a TypeError in Python.",
        error_text='TypeError: can only concatenate str (not "int") to str',
    ))
    
    T.append(Template(
        tid="py_regex_none",
        language="python",
        category="null_or_none",
        kind="crash",
        subtle=True,
        lines=[
            "import re",
            "",
            "def {{FN}}_age(text):",
            "    m = re.search(r'age: (\\d+)', text)",
            "    # BUG: m can be None",
            "    return int(m.group(1))",
            "",
            "print({{FN}}_age('name: {{NAME}}'))",
        ],
        bug_lines=[6],
        reason="Assumes regex search always succeeds; when it returns None, accessing group() fails.",
        error_text="AttributeError: 'NoneType' object has no attribute 'group'",
    ))
    
    T.append(Template(
        tid="py_bounds_empty",
        language="python",
        category="bounds_or_indexing",
        kind="crash",
        lines=[
            "def {{FN}}_first({{XS}}):",
            "    # BUG: no empty-check",
            "    return {{XS}}[0]",
            "",
            "print({{FN}}_first([]))",
        ],
        bug_lines=[3],
        reason="Indexes position 0 without validating the list is non-empty.",
        error_text="IndexError: list index out of range",
    ))
    
    T.append(Template(
        tid="py_off_by_one_subtle",
        language="python",
        category="bounds_or_indexing",
        kind="silent",
        subtle=True,
        lines=[
            "def {{FN}}_last({{XS}}):",
            "    # BUG: off-by-one, returns None for last element",
            "    return {{XS}}[len({{XS}})]  # Should be len-1",
            "",
            "print({{FN}}_last([1,2,3]))  # Prints None silently",
        ],
        bug_lines=[3],
        reason="Uses length as index instead of length-1, returns None for last element.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="py_missing_return",
        language="python",
        category="logic_branch",
        kind="silent",
        lines=[
            "def {{FN}}_find({{XS}}, target):",
            "    for {{X}} in {{XS}}:",
            "        if {{X}} == target:",
            "            return True",
            "    # BUG: missing return False; returns None",
            "",
            "print({{FN}}_find([1, 2, 3], 9))",
        ],
        bug_lines=[1, 2, 3, 4, 5],
        reason="Only returns True on one path; otherwise implicitly returns None instead of False.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="py_mut_default",
        language="python",
        category="concurrency_or_state",
        kind="silent",
        lines=[
            "def {{FN}}_push(x, acc=[]):",
            "    # BUG: mutable default persists across calls",
            "    acc.append(x)",
            "    return acc",
            "",
            "a = {{FN}}_push(1)",
            "b = {{FN}}_push(2)",
            "print(a, b)",
        ],
        bug_lines=[1, 3],
        reason="Mutable default argument is shared across invocations, leading to unexpected state reuse.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="py_sort_return_none",
        language="python",
        category="api_misuse",
        kind="silent",
        lines=[
            "def {{FN}}_sorted_copy({{XS}}):",
            "    ys = {{XS}}.copy()",
            "    # BUG: list.sort() returns None",
            "    ys = ys.sort()",
            "    return ys",
            "",
            "print({{FN}}_sorted_copy([3, 1, 2]))",
        ],
        bug_lines=[4],
        reason="Uses the return value of list.sort(), which is None; the function returns None instead of a list.",
        error_text=None,
    ))

    # ---- JavaScript ----
    T.append(Template(
        tid="js_uninit_total",
        language="javascript",
        category="logic_branch",
        kind="silent",
        lines=[
            "function {{FN}}Sum({{XS}}) {",
            "  let total; // BUG: not initialized",
            "  for (const {{X}} of {{XS}}) {",
            "    total += {{X}};",
            "  }",
            "  return total;",
            "}",
            "",
            "console.log({{FN}}Sum([1,2,3]));",
        ],
        bug_lines=[2, 4],
        reason="Adds to an uninitialized accumulator; result becomes NaN instead of a numeric sum.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="js_loop_leq",
        language="javascript",
        category="bounds_or_indexing",
        kind="silent",
        subtle=True,
        lines=[
            "const {{XS}} = [10, 20, 30];",
            "for (let i = 0; i <= {{XS}}.length; i++) { // BUG: <= out of bounds",
            "  console.log({{XS}}[i]);",
            "}",
        ],
        bug_lines=[2],
        reason="Loop iterates one step too far and reads undefined element.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="js_str_num_add",
        language="javascript",
        category="type_or_shape",
        kind="silent",
        lines=[
            "function {{FN}}Total(a, b) {",
            "  // BUG: if b is a string, + does concatenation",
            "  return a + b;",
            "}",
            "",
            "console.log({{FN}}Total(10, '5'));",
        ],
        bug_lines=[3],
        reason="Uses + with a string operand, producing concatenation ('105') instead of numeric addition (15).",
        error_text=None,
    ))
    
    # Only keep ONE sort example
    T.append(Template(
        tid="js_sort_lex",
        language="javascript",
        category="api_misuse",
        kind="silent",
        lines=[
            "const {{XS}} = [10, 2, 1];",
            "// BUG: default sort is lexicographic",
            "{{XS}}.sort();",
            "console.log({{XS}});",
        ],
        bug_lines=[3],
        reason="Uses default Array.sort() which sorts as strings, producing incorrect numeric order.",
        error_text=None,
    ))

    # ---- TypeScript ----
    T.append(Template(
        tid="ts_optional_age",
        language="typescript",
        category="null_or_none",
        kind="silent",
        subtle=True,
        lines=[
            "type User = { name: string; age?: number };",
            "function {{FN}}Retire(u: User): number {",
            "  // BUG: age can be undefined",
            "  return 65 - u.age;",
            "}",
            "",
            "console.log({{FN}}Retire({ name: '{{NAME}}' }));",
        ],
        bug_lines=[4],
        reason="Uses optional field as if always present; can yield NaN and violates type safety in strict mode.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="ts_nonnull_assert",
        language="typescript",
        category="null_or_none",
        kind="crash",
        subtle=True,
        lines=[
            "type Cfg = { path?: string };",
            "function {{FN}}Read(cfg: Cfg): number {",
            "  // BUG: non-null assertion on possibly undefined",
            "  return cfg.path!.length;",
            "}",
            "",
            "console.log({{FN}}Read({}));",
        ],
        bug_lines=[4],
        reason="Non-null assertion can still throw at runtime when the value is actually undefined.",
        error_text="TypeError: Cannot read properties of undefined (reading 'length')",
    ))

    # ---- Java ----
    T.append(Template(
        tid="java_npe",
        language="java",
        category="null_or_none",
        kind="crash",
        lines=[
            "public class Main {",
            "  public static void main(String[] args) {",
            "    String s = null;",
            "    // BUG: null dereference",
            "    System.out.println(s.length());",
            "  }",
            "}",
        ],
        bug_lines=[5],
        reason="Dereferences a null reference, causing a NullPointerException.",
        error_text="java.lang.NullPointerException",
    ))
    
    T.append(Template(
        tid="java_array_oob",
        language="java",
        category="bounds_or_indexing",
        kind="crash",
        lines=[
            "public class Main {",
            "  static int {{FN}}First(int[] a) {",
            "    // BUG: no length check",
            "    return a[0];",
            "  }",
            "  public static void main(String[] args) {",
            "    int[] x = new int[]{};",
            "    System.out.println({{FN}}First(x));",
            "  }",
            "}",
        ],
        bug_lines=[4],
        reason="Reads index 0 from an empty array, causing ArrayIndexOutOfBoundsException.",
        error_text="java.lang.ArrayIndexOutOfBoundsException",
    ))
    
    T.append(Template(
        tid="java_equals_bug",
        language="java",
        category="logic_branch",
        kind="silent",
        lines=[
            "public class Main {",
            "  static boolean {{FN}}IsAdmin(String role) {",
            "    // BUG: compares strings with ==",
            "    return role == \"admin\";",
            "  }",
            "  public static void main(String[] args) {",
            "    System.out.println({{FN}}IsAdmin(new String(\"admin\")));",
            "  }",
            "}",
        ],
        bug_lines=[4],
        reason="Uses reference equality (==) for strings; can return false even when contents match.",
        error_text=None,
    ))

    # ---- C ----
    T.append(Template(
        tid="c_missing_semicolon",
        language="c",
        category="compilation_or_syntax",
        kind="crash",
        lines=[
            "#include <stdio.h>",
            "",
            "int main() {",
            "  int x = 10;",
            "  // BUG: missing semicolon",
            "  printf(\"%d\\n\", x)",
            "  return 0;",
            "}",
        ],
        bug_lines=[6],
        reason="Missing semicolon after a statement triggers a compilation error in C.",
        error_text="error: expected ';' after expression",
    ))
    
    T.append(Template(
        tid="c_oob_array",
        language="c",
        category="bounds_or_indexing",
        kind="silent",
        lines=[
            "#include <stdio.h>",
            "",
            "int main() {",
            "  int a[3] = {1,2,3};",
            "  // BUG: out-of-bounds access",
            "  printf(\"%d\\n\", a[3]);",
            "  return 0;",
            "}",
        ],
        bug_lines=[6],
        reason="Reads beyond the end of the array (valid indices are 0..2).",
        error_text=None,
    ))
    
    T.append(Template(
        tid="c_null_deref",
        language="c",
        category="null_or_none",
        kind="crash",
        lines=[
            "#include <stdio.h>",
            "",
            "int main() {",
            "  char *p = NULL;",
            "  // BUG: dereference NULL pointer",
            "  printf(\"%c\\n\", p[0]);",
            "  return 0;",
            "}",
        ],
        bug_lines=[6],
        reason="Dereferences a NULL pointer, which can crash (segfault).",
        error_text="Segmentation fault (core dumped)",
    ))

    # ---- Go ----
    T.append(Template(
        tid="go_nil_map",
        language="go",
        category="null_or_none",
        kind="crash",
        lines=[
            "package main",
            "",
            "import \"fmt\"",
            "",
            "func main() {",
            "    var m map[string]int",
            "    // BUG: writing to nil map panics",
            "    m[\"k\"] = 1",
            "    fmt.Println(m)",
            "}",
        ],
        bug_lines=[8],
        reason="Writing to a nil map causes a runtime panic in Go.",
        error_text="panic: assignment to entry in nil map",
    ))
    
    T.append(Template(
        tid="go_index_oob",
        language="go",
        category="bounds_or_indexing",
        kind="crash",
        lines=[
            "package main",
            "",
            "import \"fmt\"",
            "",
            "func main() {",
            "    xs := []int{1,2,3}",
            "    // BUG: index out of range",
            "    fmt.Println(xs[3])",
            "}",
        ],
        bug_lines=[8],
        reason="Indexes slice out of bounds, causing a panic.",
        error_text="panic: runtime error: index out of range",
    ))

    # ---- Rust ----
    T.append(Template(
        tid="rs_index_oob",
        language="rust",
        category="bounds_or_indexing",
        kind="crash",
        lines=[
            "fn main() {",
            "    let v = vec![1, 2, 3];",
            "    // BUG: index out of bounds (panic)",
            "    println!(\"{}\", v[3]);",
            "}",
        ],
        bug_lines=[4],
        reason="Indexes beyond vector length, causing a panic.",
        error_text="thread 'main' panicked at 'index out of bounds'",
    ))

    # ---- Negatives (multi-language) ----
    T.append(Template(
        tid="neg_py_ok",
        language="python",
        category="logic_branch",
        kind="negative",
        lines=[
            "def {{FN}}_avg({{XS}}):",
            "    if not {{XS}}:",
            "        return 0.0",
            "    return sum({{XS}}) / len({{XS}})",
            "",
            "print({{FN}}_avg([2, 4, 6]))",
        ],
        bug_lines=[],
        reason="No clear bug: validates empty input and computes average consistently.",
        error_text=None,
    ))
    
    T.append(Template(
        tid="neg_js_ok",
        language="javascript",
        category="logic_branch",
        kind="negative",
        lines=[
            "function {{FN}}Clamp(x, lo, hi) {",
            "  if (x < lo) return lo;",
            "  if (x > hi) return hi;",
            "  return x;",
            "}",
            "",
            "console.log({{FN}}Clamp(5, 0, 10));",
        ],
        bug_lines=[],
        reason="No clear bug: straightforward clamp logic with explicit returns on all paths.",
        error_text=None,
    ))

    # Validate no duplicate IDs
    for t in T:
        assert t.tid not in used_ids, f"Duplicate template ID: {t.tid}"
        used_ids.add(t.tid)
    
    return T


def maybe_include_error(rng: random.Random, t: Template, crash_error_prob: float) -> Tuple[Optional[str], List[str]]:
    """Smart error inclusion - always include for obvious crashes."""
    if t.kind != "crash":
        return None, ["code"]
    
    if t.category in ["bounds_or_indexing", "null_or_none"] and not t.subtle:
        if t.error_text:
            return t.error_text, ["code", "error_output"]
    
    include = rng.random() < crash_error_prob
    if include and t.error_text:
        return t.error_text, ["code", "error_output"]
    
    return None, ["code"]


def get_precise_region(t: Template, code_lines: List[str]) -> Tuple[int, int]:
    """Get precise region excluding comments and whitespace."""
    if not t.bug_lines:
        return 0, 0
    
    valid_lines = []
    for line_num in t.bug_lines:
        if line_num <= len(code_lines):
            line = code_lines[line_num - 1].strip()
            if line and not line.startswith(('#', '//', '/*', '*', '*/')):
                valid_lines.append(line_num)
    
    if not valid_lines:
        valid_lines = t.bug_lines
    
    return min(valid_lines), max(valid_lines)


def make_example(
    rng: random.Random,
    ex_id: str,
    t: Template,
    crash_error_prob: float,
) -> Dict[str, Any]:
    """Create a single example with proper formatting."""
    code_lines = apply_variations_keep_lines(rng, t.lines)
    error_text, inputs_used = maybe_include_error(rng, t, crash_error_prob)
    
    nlines = len(code_lines)
    
    if t.kind == "negative":
        suspects: List[Dict[str, Any]] = []
        debug_steps = [
            "Run with representative inputs and verify whether any failure occurs.",
            "Add assertions/logging around assumptions (keys, bounds, types).",
            "If a failure occurs, capture the exact error/incorrect output to narrow the suspect region.",
        ]
    else:
        start, end = get_precise_region(t, code_lines)
        # Ensure within bounds
        start = max(1, min(start, nlines))
        end = max(1, min(end, nlines))
        if end < start:
            start, end = end, start
        
        conf_range = get_confidence_range(t)
        conf = round(rng.uniform(conf_range[0], conf_range[1]), 2)
        
        suspects = [{
            "region_id": "R1",
            "start_line": start,
            "end_line": end,
            "confidence": conf,
            "category": t.category,
            "reason": t.reason,
        }]
        debug_steps = [
            "Confirm the failing line by running or compiling the snippet and reading the error message or incorrect output.",
            "Inspect the values/types used at the suspect line(s) (e.g., logging, assertions, debugger).",
            "Check assumptions about inputs, boundary cases, and control-flow paths that reach the suspect region.",
        ]
    
    assistant_payload = {
        "task": "bug_localization",
        "language": t.language,
        "inputs_used": inputs_used,
        "suspects": suspects,
        "debug_next_steps": debug_steps,
        "no_fix_policy": {
            "provide_patch": False,
            "provide_exact_replacement": False,
            "explanation": "I can indicate where to look and how to debug, but I will not provide the code change.",
        },
    }
    
    user_content = f"<CODE>\n{''.join(code_lines).rstrip()}\n</CODE>"
    if error_text:
        user_content += f"\n\n<ERROR>\n{error_text.strip()}\n</ERROR>"
    user_content += USER_SUFFIX
    
    return {
        "id": ex_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ],
    }


def validate_example(example: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    """Validate that example matches expected patterns."""
    try:
        msg = example["messages"][2]["content"]
        data = json.loads(msg)
        
        required_fields = ["task", "language", "inputs_used", "suspects", "debug_next_steps", "no_fix_policy"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        for suspect in data["suspects"]:
            required_suspect_fields = ["region_id", "start_line", "end_line", "confidence", "category", "reason"]
            for field in required_suspect_fields:
                if field not in suspect:
                    return False, f"Missing suspect field: {field}"
            
            # Check confidence ranges (relaxed to match actual generation)
            conf = suspect["confidence"]
            category = suspect["category"]
            
            valid_ranges = {
                "bounds_or_indexing": (0.55, 0.96),
                "null_or_none": (0.70, 0.96),
                "type_or_shape": (0.70, 0.95),
                "logic_branch": (0.55, 0.95),
                "api_misuse": (0.50, 0.80),
                "concurrency_or_state": (0.70, 0.85),
                "compilation_or_syntax": (0.75, 0.95),
            }
            
            if category in valid_ranges:
                lo, hi = valid_ranges[category]
                if not (lo <= conf <= hi):
                    return False, f"Bad confidence {conf} for {category} (expected {lo}-{hi})"
            
            if suspect["end_line"] - suspect["start_line"] > 8:
                return False, f"Region too wide: {suspect['end_line'] - suspect['start_line'] + 1} lines"
            
            if suspect["start_line"] > suspect["end_line"]:
                return False, "Invalid region: start > end"
        
        if verbose:
            print(f"Validated example {example['id']}")
        return True, "OK"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except (KeyError, TypeError, ValueError) as e:
        return False, f"Validation error: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train_sft_synth_multilang.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_total", type=int, default=70)
    ap.add_argument("--crash_error_prob", type=float, default=0.7)
    ap.add_argument("--mix", type=str, default="", help="e.g. crash=40,silent=20,negative=10")
    ap.add_argument("--validate", action="store_true", help="Validate generated examples")
    ap.add_argument("--verbose", action="store_true", help="Show validation details")
    args = ap.parse_args()
    
    rng = random.Random(args.seed)
    templates = build_templates()
    
    if args.mix:
        parts = args.mix.replace("crash=", "").replace("silent=", "").replace("negative=", "").split(",")
        n_crash, n_silent, n_negative = map(int, parts)
    else:
        n_crash = int(args.n_total * 0.6)
        n_silent = int(args.n_total * 0.3)
        n_negative = args.n_total - n_crash - n_silent
    
    crash_templates = [t for t in templates if t.kind == "crash"]
    silent_templates = [t for t in templates if t.kind == "silent"]
    negative_templates = [t for t in templates if t.kind == "negative"]
    
    selected = []
    selected.extend(rng.choices(crash_templates, k=n_crash))
    selected.extend(rng.choices(silent_templates, k=n_silent))
    selected.extend(rng.choices(negative_templates, k=n_negative))
    rng.shuffle(selected)
    
    examples = []
    validation_failures = []
    
    for i, t in enumerate(selected[:args.n_total], 1):
        ex = make_example(rng, f"ex_synth_{i:03d}", t, args.crash_error_prob)
        
        if args.validate:
            valid, msg = validate_example(ex, args.verbose)
            if not valid:
                validation_failures.append((i, msg, t.tid))
                if args.verbose:
                    print(f"✗ Failed validation for {ex['id']} ({t.tid}): {msg}")
        
        examples.append(ex)
    
    if validation_failures:
        print(f"\n {len(validation_failures)} validation failures:")
        for i, msg, tid in validation_failures[:5]:
            print(f"  - Example {i} ({tid}): {msg}")
        if len(validation_failures) > 5:
            print(f"  ... and {len(validation_failures) - 5} more")
        
        response = input("\nContinue writing file? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    with open(args.out, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    kind_counts = {"crash": 0, "silent": 0, "negative": 0}
    lang_counts = {}
    for t in selected[:args.n_total]:
        kind_counts[t.kind] += 1
        lang_counts[t.language] = lang_counts.get(t.language, 0) + 1
    
    print(f"\nGenerated {len(examples)} examples to {args.out}")
    print(f"Kind mix: {kind_counts}")
    print(f"Language mix: {dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))}")
    if validation_failures:
        print(f"{len(validation_failures)} examples failed validation (written to file anyway)")


if __name__ == "__main__":
    main()
