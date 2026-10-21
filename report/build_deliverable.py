"""Build the printed deliverable: the one-page report followed by its annexes.

Annex A: annotated prompt log (context/prompt_log.md).
Annex B: output of main.py.
Annex C: code excerpts, the functions behind the report's claims (EXCERPTS).
Run report_render.compile_report first so report/report.pdf is current.
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
# Printed code: the audited estimator and the ARCH-LM test (Block A), the news impact curve (Block B).
EXCERPTS = [
    ("src/garch_mle.py", "fit_garch11"),
    ("src/garch_mle.py", "hessian_standard_errors"),
    ("src/tutorial1.py", "arch_lm"),
    ("src/tutorial1.py", "news_impact_curve"),
]


def function_source(path, name):
    """Source of one top-level function and its first line number in the file."""
    text = (ROOT / path).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno]), node.lineno


def tex_escape(text):
    out = []
    for c in text:
        out.append({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}.get(c, c))
    return "".join(out)


def inline(text):
    """Markdown inline (code, bold, $math$) to LaTeX."""
    parts = re.split(r"(`[^`]*`|\$[^$]*\$)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            # Break points after underscores and spaces: long names in narrow columns.
            code = tex_escape(p[1:-1]).replace(r"\_", r"\_\allowbreak{}")
            out.append(r"\texttt{" + code + "}")
        elif p.startswith("$") and p.endswith("$") and len(p) > 1:
            out.append(p)
        else:
            p = tex_escape(p)
            out.append(re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p))
    return "".join(out)


def prompt_log_tex(markdown):
    lines, out, in_table = markdown.splitlines(), [], False
    for line in lines:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                out.append(r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.26\linewidth}>{\raggedright\arraybackslash}p{0.38\linewidth}>{\raggedright\arraybackslash}p{0.30\linewidth}@{}}\toprule")
                out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in cells) + r" \\ \midrule \endhead")
                in_table, first_row = True, True
                continue
            # Rules between rows only, so the table ends on a single bottom rule.
            out.append(("" if first_row else r"\midrule ") + " & ".join(inline(c) for c in cells) + r" \\")
            first_row = False
            continue
        if in_table:
            out.append(r"\bottomrule\end{longtable}")
            in_table = False
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            out.append(r"\subsection*{" + inline(line[3:]) + "}")
        elif line.strip():
            out.append(inline(line) + "\n")
    if in_table:
        out.append(r"\bottomrule\end{longtable}")
    return "\n".join(out)


def main_output():
    run = subprocess.run([sys.executable, "main.py"], cwd=ROOT, capture_output=True, text=True, check=True)
    return run.stdout.replace(str(ROOT), ".").rstrip()


def annexes_tex():
    log = prompt_log_tex((ROOT / "context/prompt_log.md").read_text(encoding="utf-8"))
    blocks = []
    for path, name in EXCERPTS:
        source, first = function_source(path, name)
        blocks.append(
            r"\subsection*{\texttt{" + tex_escape(f"{path} : {name}") + "}}\n"
            + f"\\begin{{lstlisting}}[firstnumber={first}]\n{source}\n\\end{{lstlisting}}"
        )
    code = "\n".join(blocks)
    return rf"""\documentclass[9pt,a4paper]{{extarticle}}
\usepackage{{fontspec}}
\usepackage[english]{{babel}}
\usepackage{{amsmath,amssymb,array,booktabs,longtable,listings,xcolor,fancyhdr}}
\usepackage[margin=14mm,top=18mm]{{geometry}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\pagestyle{{fancy}}\fancyhf{{}}
\fancyhead[L]{{\small GARCH Audit, appendices}}
\fancyhead[R]{{\small Yosri BEN HALIMA, Mathieu CHABIRAND}}
\fancyfoot[C]{{\small \thepage}}
\setcounter{{page}}{{2}}
\lstset{{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,numbers=left,
  numberstyle=\tiny\color{{gray}},keywordstyle=\color{{blue!60!black}},commentstyle=\color{{green!40!black}},
  language=Python,frame=single,rulecolor=\color{{gray!50}}}}
\begin{{document}}
\section*{{Appendix A: annotated prompt log}}
{log}
\section*{{Appendix B: output of \texttt{{main.py}}}}
Every number labelled \emph{{computed}} in the report is produced by this script (and \texttt{{report/results.json}}).\\
Data: \texttt{{data/NLNPE\_TD\_data.xlsx}}, sheet \texttt{{weekly\_spx\_ust10}}.
\begin{{lstlisting}}[language={{}},numbers=none]
{main_output()}
\end{{lstlisting}}
\section*{{Appendix C: code excerpts}}
The audited estimator and the ARCH-LM test (Block A), the news impact curve (Block B), with their line numbers in the repository.
{code}
\end{{document}}
"""


def build():
    source = REPORT / "annexes.tex"
    source.write_text(annexes_tex(), encoding="utf-8")
    compiler = shutil.which("tectonic") or str(Path.home() / "AppData/Local/Programs/tectonic/tectonic.exe")
    subprocess.run([compiler, "--outdir", str(REPORT), str(source)], check=True, cwd=REPORT)
    merged = pymupdf.open(REPORT / "report.pdf")
    merged.insert_pdf(pymupdf.open(REPORT / "annexes.pdf"))
    merged.save(REPORT / "deliverable.pdf")
    print(f"{REPORT / 'deliverable.pdf'}: {merged.page_count} pages")


if __name__ == "__main__":
    build()
