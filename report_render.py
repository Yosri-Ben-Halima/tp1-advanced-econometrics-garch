"""Compile the separately authored LaTeX report without generating its prose."""

import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from tutorial1 import news_impact_curve


def report_evidence(result):
    return {key: result[key] for key in
            ["n", "start", "end", "statistics", "custom", "arch", "gjr", "leverage", "nic",
             "arch_lm", "post_garch_arch_lm"]}


def same_evidence(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_evidence(left[k], right[k]) for k in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-7)
    return left == right


def compile_report(result, output):
    output = Path(output).resolve()
    source = output/"report.tex"
    evidence = output/"report_evidence.json"
    if not source.exists() or not evidence.exists():
        raise FileNotFoundError("The authored report.tex and its report_evidence.json are required.")
    if not same_evidence(report_evidence(result), json.loads(evidence.read_text(encoding="utf-8"))):
        raise ValueError("Analysis results changed. Review report.tex and update report_evidence.json before compiling.")
    z = np.linspace(-4, 4, 401)
    garch, _ = news_impact_curve(result["custom"]["estimates"], z)
    gjr, _ = news_impact_curve(result["gjr"]["estimates"], z, gjr=True)
    curves = np.column_stack([z, garch, gjr])
    with plt.rc_context({"font.family":"serif", "font.size":10}):
        fig, ax = plt.subplots(figsize=(6.8,2.4))
        ax.plot(curves[:,0],curves[:,1],color="#235AA6",lw=1.6,label="GARCH")
        ax.plot(curves[:,0],curves[:,2],color="#B86E18",lw=1.6,ls="--",label="GJR-GARCH")
        ax.set(xlim=(-4,4),ylim=(0,np.max(curves[:,1:])*1.12),
               xlabel=r"Choc standardisé $z$",ylabel=r"Variance suivante ($\%^{2}$)")
        ax.set_yticks([0,10,20,30]); ax.grid(axis="y",color="#E4E7EB",lw=.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=9,loc="upper right",ncol=2)
        fig.tight_layout(pad=.2)
        fig.savefig(output/"news_impact.pdf")
        plt.close(fig)
    compiler = shutil.which("tectonic")
    if compiler is None:
        existing = Path(__file__).resolve().parent.parent/"garch_course/tools/tectonic.exe"
        if existing.exists():
            compiler = str(existing)
    if compiler is None:
        raise RuntimeError("No existing LaTeX compiler is available. Open report.tex in the Codex LaTeX editor to compile it.")
    subprocess.run([compiler,"--outdir",str(output),str(source)],check=True,cwd=output)
