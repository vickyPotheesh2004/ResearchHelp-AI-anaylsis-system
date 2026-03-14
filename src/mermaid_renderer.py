"""
Mermaid Renderer Module

Provides robust Mermaid diagram cleaning, validation, and rendering utilities.
Fixes common LLM-generated Mermaid syntax errors before they reach the UI.

Classes:
    MermaidCleaner: Strips/fixes bad syntax before rendering
    MermaidValidator: Catches errors before they reach the UI

Functions:
    extract_mermaid_blocks(): Finds all Mermaid code blocks in LLM response

Usage:
    from mermaid_renderer import MermaidCleaner, MermaidValidator, extract_mermaid_blocks
    
    # Extract and clean
    blocks = extract_mermaid_blocks(llm_response)
    for block in blocks:
        cleaned = MermaidCleaner.clean(block)
        is_valid, error = MermaidValidator.validate(cleaned)
        if is_valid:
            render(cleaned)
        else:
            show_error(error)
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


# ── 1A. KNOWN VALID DIAGRAM TYPES ────────────────────────────────────────────

VALID_DIAGRAM_TYPES = {
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "timeline",
    "quadrantChart",
    "xychart-beta",
    "block-beta",
    "journey",
    "C4Context",
}

# ── 1B. MERMAID CLEANER ───────────────────────────────────────────────────────

class MermaidCleaner:
    """
    Cleans LLM-generated Mermaid syntax before it reaches the renderer.
    Fixes every known category of syntax error without altering the diagram meaning.
    """

    @staticmethod
    def clean(raw: str) -> str:
        """
        Master cleaning pipeline. Returns cleaned Mermaid code string.
        Call this on any LLM output before passing to the renderer.
        """
        code = raw.strip()

        # Step 1 — Strip markdown fences (```mermaid ... ``` or ``` ... ```)
        code = MermaidCleaner._strip_fences(code)

        # Step 2 — Strip leading/trailing whitespace again after fence removal
        code = code.strip()

        # Step 3 — Fix diagram type keyword on first line
        code = MermaidCleaner._fix_diagram_type(code)

        # Step 4 — Fix node label special characters
        code = MermaidCleaner._fix_node_labels(code)

        # Step 5 — Fix arrow syntax (LLMs often use wrong arrow types)
        code = MermaidCleaner._fix_arrows(code)

        # Step 6 — Fix indentation (tabs → spaces, normalize)
        code = MermaidCleaner._fix_indentation(code)

        # Step 7 — Remove blank lines inside node/edge definitions
        code = MermaidCleaner._remove_consecutive_blanks(code)

        # Step 8 — Fix subgraph syntax
        code = MermaidCleaner._fix_subgraphs(code)

        # Step 9 — Ensure diagram ends with a newline
        code = code.rstrip() + "\n"

        return code

    # ── Private cleaners ─────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(code: str) -> str:
        """Remove ```mermaid ... ``` and ``` ... ``` wrappers."""
        # Remove opening fence: ```mermaid or ```
        code = re.sub(r"^```(?:mermaid)?\s*\n?", "", code, flags=re.IGNORECASE)
        # Remove closing fence: ```
        code = re.sub(r"\n?```\s*$", "", code)
        return code

    @staticmethod
    def _fix_diagram_type(code: str) -> str:
        """
        Ensures the first non-empty line is a valid Mermaid diagram type.
        Fixes common LLM mistakes:
          - 'flowChart' → 'flowchart'
          - 'sequence'  → 'sequenceDiagram'
          - 'graph TB\n' missing space → kept as-is (already valid)
          - 'graph' alone → 'graph TD' (add default direction)
        """
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            first_word = stripped.split()[0].lower()

            # Fix case variations
            case_fixes = {
                "flowchart": "flowchart",
                "flowChart": "flowchart",
                "Flowchart": "flowchart",
                "sequence": "sequenceDiagram",
                "sequencediagram": "sequenceDiagram",
                "classdiagram": "classDiagram",
                "statediagram": "stateDiagram-v2",
                "erdiagram": "erDiagram",
                "gitgraph": "gitGraph",
                "mindmap": "mindmap",
                "timeline": "timeline",
            }

            for wrong, correct in case_fixes.items():
                if first_word == wrong.lower() and stripped.lower().startswith(wrong.lower()):
                    lines[i] = line.replace(stripped.split()[0], correct, 1)
                    stripped = lines[i].strip()
                    break

            # "graph" alone without direction → add TD
            if stripped == "graph":
                lines[i] = line.replace("graph", "graph TD", 1)

            # "flowchart" alone without direction → add TD
            if stripped == "flowchart":
                lines[i] = line.replace("flowchart", "flowchart TD", 1)

            break  # only process first non-empty line

        return "\n".join(lines)

    @staticmethod
    def _fix_node_labels(code: str) -> str:
        """
        Fixes special characters inside node labels that break the parser.

        Mermaid node label rules:
          - Square bracket labels: A[label]   — safe chars: letters, spaces, hyphens
          - Parenthesis labels:    A(label)   — same rules
          - Curly labels:          A{label}   — decision nodes
          - Labels must NOT contain: unescaped quotes, colons, angle brackets inside []

        Fixes applied:
          - Quotes inside [] labels → removed
          - Colons inside [] labels → replaced with " —"
          - Unmatched brackets → removed
          - HTML entities inside labels → decoded
        """
        def fix_label_content(match):
            open_bracket = match.group(1)   # [, (, or {
            content = match.group(2)
            close_bracket = match.group(3)  # ], ), or }

            # Replace colons (breaks Mermaid sequence diagrams outside of ->:)
            content = content.replace(":", " —")

            # Replace raw double quotes → remove (Mermaid uses quotes for classDef)
            content = content.replace('"', "")
            content = content.replace("'", "")

            # Remove angle brackets that aren't part of arrows
            content = re.sub(r"<(?!br)(?!/br)([^>]{0,30})>", r"\1", content)

            # Decode common HTML entities
            content = content.replace("&", "&")
            content = content.replace("<", "less than")
            content = content.replace(">", "greater than")
            content = content.replace("&nbsp;", " ")

            return f"{open_bracket}{content}{close_bracket}"

        # Match content inside [], (), {} node labels
        code = re.sub(
            r"(\[)([^\[\]]*?)(\])",
            fix_label_content,
            code
        )
        code = re.sub(
            r"(\()([^()]*?)(\))",
            fix_label_content,
            code
        )
        code = re.sub(
            r"(\{)([^{}]*?)(\})",
            fix_label_content,
            code
        )

        return code

    @staticmethod
    def _fix_arrows(code: str) -> str:
        """
        Fixes common arrow syntax mistakes LLMs make.

        Valid Mermaid arrows:
          -->   (normal)
          ---   (no arrowhead)
          --->  (dotted, longer)
          ==>   (thick)
          -.->  (dotted with arrow)
          -->>  (async in sequence)
          ->>   (solid line in sequence)

        Common LLM mistakes:
          ->    (missing one dash)  → -->
          =>    (wrong thick arrow) → ==>
          ~>    (invalid)           → -->
          ──>   (unicode dash)      → -->
          —>    (em dash)           → -->
        """
        # em dash or unicode long dash → double hyphen
        code = code.replace("——>", "-->")
        code = code.replace("—>",  "-->")
        code = code.replace("–>",  "-->")

        # Single dash arrow (outside of node labels)
        code = re.sub(r"(?<![>\-=])->(?!\>)", "-->", code)

        # => not preceded by = → ==>
        code = re.sub(r"(?<!=)=>(?!=)", "==>", code)

        # ~> → -->
        code = code.replace("~>", "-->")

        # Unicode arrow characters
        code = code.replace("→", "-->")
        code = code.replace("⟶", "-->")
        code = code.replace("⇒", "==>")

        return code

    @staticmethod
    def _fix_indentation(code: str) -> str:
        """Convert tabs to 4 spaces. Strip trailing whitespace per line."""
        lines = []
        for line in code.split("\n"):
            line = line.replace("\t", "    ")
            line = line.rstrip()
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _remove_consecutive_blanks(code: str) -> str:
        """Collapse 2+ consecutive blank lines into one."""
        return re.sub(r"\n{3,}", "\n\n", code)

    @staticmethod
    def _fix_subgraphs(code: str) -> str:
        """
        Ensures subgraph blocks are correctly closed.
        LLMs often forget the 'end' keyword.
        """
        lines = code.split("\n")
        open_subgraphs = 0
        fixed = []

        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith("subgraph"):
                open_subgraphs += 1
            elif stripped == "end":
                if open_subgraphs > 0:
                    open_subgraphs -= 1
            fixed.append(line)

        # Close any unclosed subgraphs
        for _ in range(open_subgraphs):
            fixed.append("end")

        return "\n".join(fixed)


# ── 1C. MERMAID VALIDATOR ─────────────────────────────────────────────────────

class MermaidValidator:
    """
    Lightweight pre-render validation.
    Catches the most common errors before they produce a red error box in the UI.
    Returns (is_valid: bool, error_message: str | None).
    """

    @staticmethod
    def validate(code: str) -> tuple[bool, str | None]:
        """
        Returns (True, None) if code looks valid.
        Returns (False, "reason") if a structural problem is detected.
        """
        lines = [l for l in code.split("\n") if l.strip()]

        if not lines:
            return False, "Empty diagram — no content to render."

        # Check first line is a known diagram type
        first_word = lines[0].strip().split()[0]
        if first_word not in VALID_DIAGRAM_TYPES:
            return False, (
                f"Unknown diagram type '{first_word}'. "
                f"Must be one of: {', '.join(sorted(VALID_DIAGRAM_TYPES))}"
            )

        # Must have at least 2 lines (type + at least one node/step)
        if len(lines) < 2:
            return False, "Diagram has no content after the type declaration."

        # Check for unmatched brackets in node labels
        bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        for line in lines[1:]:
            # Skip comment lines
            if line.strip().startswith("%%"):
                continue
            for open_b, close_b in bracket_pairs:
                opens = line.count(open_b)
                closes = line.count(close_b)
                if abs(opens - closes) > 1:
                    return False, (
                        f"Unmatched '{open_b}' bracket on line: {line.strip()[:60]}"
                    )

        # flowchart / graph must have a direction
        if first_word in ("flowchart", "graph"):
            parts = lines[0].strip().split()
            if len(parts) < 2:
                return False, (
                    f"'{first_word}' requires a direction: TD, LR, BT, or RL. "
                    f"E.g. 'flowchart TD'"
                )
            direction = parts[1].upper()
            if direction not in ("TD", "TB", "LR", "RL", "BT", "DT"):
                return False, (
                    f"Invalid direction '{parts[1]}' for {first_word}. "
                    f"Use TD, LR, BT, or RL."
                )

        return True, None


# ── 1D. EXTRACT MERMAID BLOCKS FROM LLM RESPONSE ─────────────────────────────

def extract_mermaid_blocks(text: str) -> list[str]:
    """
    Finds all ```mermaid ... ``` blocks in an LLM response string.
    Returns a list of raw Mermaid code strings (fences not included).
    Also detects bare diagram blocks that start with a known diagram type
    but were not wrapped in fences.
    """
    blocks = []

    # Pattern 1 — fenced blocks: ```mermaid ... ```
    fenced = re.findall(
        r"```(?:mermaid)?\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    blocks.extend(fenced)

    # Pattern 2 — bare blocks: lines starting with a known diagram type
    # (LLMs sometimes omit the fences)
    if not blocks:
        type_pattern = "|".join(re.escape(t) for t in VALID_DIAGRAM_TYPES)
        bare = re.findall(
            rf"^({type_pattern})\s.*?(?=\n\n|\Z)",
            text,
            flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        blocks.extend(bare)

    return blocks


# ════════════════════════════════════════════════════════════════════════════
#
#  PART 2 — MERMAID RENDERING WITH FIXED ASYNC API
#
# ════════════════════════════════════════════════════════════════════════════

# Updated MERMAID_HTML_TEMPLATE using async/await for Mermaid v10+
MERMAID_HTML_TEMPLATE = """
<div id="{div_id}" style="width:100%;min-height:300px;display:flex;align-items:center;justify-content:center;background:#f8f9fa;border-radius:8px;">
    <div id="{div_id}-loading" style="color:#6c757d;">Loading diagram...</div>
</div>
<script>
(function() {{
    var code = {escaped_code};
    var divId = "{div_id}";

    async function renderMermaid() {{
        try {{
            // Check if mermaid is loaded, if not load it
            if (typeof mermaid === "undefined") {{
                var script = document.createElement('script');
                script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
                document.head.appendChild(script);
                await new Promise(function(resolve) {{
                    script.onload = resolve;
                    script.onerror = resolve;
                }});
            }}
            
            // Initialize mermaid
            mermaid.initialize({{
                startOnLoad: false,
                theme: "default",
                themeVariables: {{
                    fontSize: "14px",
                    fontFamily: "sans-serif",
                }},
                flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: "basis" }},
                sequence:  {{ useMaxWidth: true }},
                gantt:     {{ useMaxWidth: true }},
                securityLevel: "loose",
            }});
            
            // Generate unique ID for this diagram
            var id = "mermaid-" + Math.random().toString(36).substr(2,9);
            
            // Use async render for Mermaid v10+
            var result = await mermaid.render(id, code);
            
            // Insert the SVG into the container
            var container = document.getElementById(divId);
            container.innerHTML = result.svg;
            
            // Make SVG responsive
            var svg = container.querySelector('svg');
            if (svg) {{
                svg.style.maxWidth = "100%";
                svg.style.height = "auto";
            }}
        }} catch(e) {{
            document.getElementById(divId).innerHTML =
                '<div style="color:#b91c1c;background:#fef2f2;padding:15px;border-radius:8px;font-size:14px;">'
                + '<strong>Mermaid Render Error:</strong><br>'
                + e.message
                + '<br><br><strong>Diagram code:</strong><pre style="background:#f5f5f5;padding:10px;overflow-x:auto;">'
                + code.replace(/</g,"<")
                + '</pre></div>';
        }}
    }}
    renderMermaid();
}})();
</script>
"""

MERMAID_CDN = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
"""


def render_content_with_mermaid(content: str) -> None:
    """
    DROP-IN REPLACEMENT for the existing render_content_with_mermaid() in app.py.

    Finds all Mermaid code blocks in the LLM response, cleans and validates
    each one, then renders them inline using Mermaid.js via Streamlit components.
    Non-Mermaid text is rendered as standard Markdown.

    Parameters
    ----------
    content : str
        The full LLM response string, which may contain ```mermaid ... ``` blocks
        mixed with regular Markdown text.
    """
    import streamlit as st
    import streamlit.components.v1 as components
    import json

    # Split content on ```mermaid ... ``` boundaries
    # Produces alternating [text, mermaid_code, text, mermaid_code, ...] segments
    parts = re.split(r"(```(?:mermaid)?\s*\n.*?```)", content, flags=re.DOTALL | re.IGNORECASE)

    # Inject Mermaid CDN once per session
    if "mermaid_cdn_injected" not in st.session_state:
        components.html(MERMAID_CDN, height=0)
        st.session_state["mermaid_cdn_injected"] = True

    import uuid

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Detect if this segment is a mermaid block
        is_mermaid = bool(re.match(r"^```(?:mermaid)?", part, re.IGNORECASE))

        if is_mermaid:
            # Extract raw code (strip fences)
            raw_code = re.sub(r"^```(?:mermaid)?\s*\n?", "", part, flags=re.IGNORECASE)
            raw_code = re.sub(r"\n?```\s*$", "", raw_code)

            # Clean the code
            cleaned = MermaidCleaner.clean(raw_code)

            # Validate
            is_valid, error_msg = MermaidValidator.validate(cleaned)

            if not is_valid:
                # Show clean error with the fixed code so the user can see what went wrong
                st.warning(f"⚠️ Diagram syntax issue: {error_msg}")
                with st.expander("Show diagram code"):
                    st.code(cleaned, language="text")
            else:
                # Render the diagram
                div_id = f"mermaid-div-{uuid.uuid4().hex[:8]}"
                escaped = json.dumps(cleaned)   # safe JS string escaping

                html = MERMAID_HTML_TEMPLATE.format(
                    div_id=div_id,
                    escaped_code=escaped,
                )
                # height auto-sizes; 400 is a safe default for most diagrams
                components.html(html, height=450, scrolling=False)

                # Also show code in expander for copy/debug
                with st.expander("View diagram code", expanded=False):
                    st.code(cleaned, language="text")

        else:
            # Regular Markdown text — render as normal
            st.markdown(part)
