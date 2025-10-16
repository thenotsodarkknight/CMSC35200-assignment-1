#!/usr/bin/env python3
"""
Compare Sophia baseline gene-analysis results with local Ollama runs.

Produces a markdown report with agree/disagree/unsure counts per disease
for each local model directory under outputs/gene_analysis_local/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "outputs" / "gene_analysis"
LOCAL_DIR = PROJECT_ROOT / "outputs" / "gene_analysis_local"
REPORT = LOCAL_DIR / "comparison.md"


def load_results(path: Path) -> Dict[str, Dict[str, bool]]:
    data = json.loads((path / "results.json").read_text())
    out: Dict[str, Dict[str, bool]] = {}
    for r in data:
        out[r["gene"]] = {
            "cancer": bool(r["has_cancer_link"]),
            "heart_disease": bool(r["has_heart_disease_link"]),
            "diabetes": bool(r["has_diabetes_link"]),
            "dementia": bool(r["has_dementia_link"]),
        }
    return out


def load_explanations(path: Path) -> Dict[str, str]:
    data = json.loads((path / "results.json").read_text())
    return {r["gene"]: r["explanation"] for r in data}


def compare_models(
    baseline: Dict[str, Dict[str, bool]],
    local: Dict[str, Dict[str, bool]],
    local_expl: Dict[str, str],
) -> Dict[str, Tuple[int, int, int]]:
    # returns per disease: (agree, disagree, unsure)
    diseases = ["cancer", "heart_disease", "diabetes", "dementia"]
    counts = {d: [0, 0, 0] for d in diseases}
    for gene, bvals in baseline.items():
        if gene not in local:
            continue
        lvals = local[gene]
        for d in diseases:
            if lvals[d] == bvals[d]:
                counts[d][0] += 1
            else:
                # mark unsure if local says False and uses cautious language
                if not lvals[d] and "no known association" in local_expl.get(gene, "").lower():
                    counts[d][2] += 1
                else:
                    counts[d][1] += 1
    return {k: tuple(v) for k, v in counts.items()}


def main() -> None:
    baseline = load_results(BASELINE_DIR)
    lines: List[str] = ["# Local vs Sophia Comparison", ""]
    for model_dir in sorted(p for p in LOCAL_DIR.iterdir() if p.is_dir()):
        local = load_results(model_dir)
        expl = load_explanations(model_dir)
        stats = compare_models(baseline, local, expl)
        lines.append(f"## {model_dir.name}")
        lines.append("| Disease | Agree | Disagree | Unsure |")
        lines.append("| --- | ---:| ---:| ---:|")
        for d, (ag, di, un) in stats.items():
            lines.append(f"| {d.replace('_',' ')} | {ag} | {di} | {un} |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {REPORT}")


if __name__ == "__main__":
    main()

