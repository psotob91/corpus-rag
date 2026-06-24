"""Write a small but realistic outputs/_sample_/ product for testing build+search.

Generates an OUTPUT_LAYOUT product (document.md, document.meta.json,
chunks.jsonl, and tables/figures/formulas JSON twins) for a synthetic
survival-analysis / epidemiology paper, so the index and retriever can be
exercised end-to-end without running real ingest.

Run:  uv run python scripts/make_sample_corpus.py
ASCII-only output.
"""
from __future__ import annotations

import json
from pathlib import Path

SLUG = "_sample_"


def _repo_root() -> Path:
    # scripts/ lives at the repo root; outputs/ is its sibling.
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Content: 6 survival-analysis / epidemiology text chunks + 1 table chunk.
# ---------------------------------------------------------------------------

TEXT_CHUNKS = [
    {
        "section_path": ["Introduction"],
        "page": 1,
        "text": (
            "Survival analysis studies the time until an event of interest, such as "
            "death or disease recurrence, occurs. A defining feature is right "
            "censoring, where the event has not been observed by the end of follow-up. "
            "These methods are central to epidemiology and clinical trials."
        ),
    },
    {
        "section_path": ["Methods", "Kaplan-Meier estimator"],
        "page": 2,
        "text": (
            "The Kaplan-Meier estimator provides a non-parametric estimate of the "
            "survival function from observed and censored event times. The survival "
            "curve steps down at each observed event and accounts for individuals lost "
            "to follow-up. The log-rank test compares survival curves between groups."
        ),
    },
    {
        "section_path": ["Methods", "Cox proportional hazards model"],
        "page": 2,
        "text": (
            "The Cox proportional hazards model estimates the effect of covariates on "
            "the hazard rate without assuming a baseline hazard shape. The hazard ratio "
            "quantifies the relative risk between groups; a hazard ratio above one "
            "indicates increased risk. The proportional hazards assumption can be "
            "checked with Schoenfeld residuals."
        ),
    },
    {
        "section_path": ["Results", "Cohort characteristics"],
        "page": 3,
        "text": (
            "We followed a cohort of 1,248 patients over a median follow-up of 4.6 "
            "years. Incidence rates were estimated per 1,000 person-years. Baseline "
            "covariates included age, sex, smoking status, and comorbidity burden."
        ),
    },
    {
        "section_path": ["Results", "Hazard ratio estimates"],
        "page": 3,
        "text": (
            "The adjusted Cox model yielded a hazard ratio of 1.74 (95% confidence "
            "interval 1.31 to 2.31) for the high-exposure group relative to the "
            "reference group. The association remained significant after adjusting for "
            "confounders, and the log-rank test gave a p-value below 0.001."
        ),
    },
    {
        "section_path": ["Discussion"],
        "page": 4,
        "text": (
            "Our findings suggest a strong dose-response relationship between exposure "
            "and event hazard. Competing risks and time-varying covariates may bias "
            "naive estimates; a Fine-Gray subdistribution hazard model offers a "
            "complementary view for competing-risks settings in epidemiology."
        ),
    },
]

TABLE_CHUNK = {
    "section_path": ["Results", "Hazard ratio estimates"],
    "page": 3,
    "kind": "table",
    "caption": (
        "Table 1. Adjusted hazard ratios from the Cox proportional hazards model, "
        "with 95% confidence intervals and p-values for each covariate."
    ),
}

# A figure and a formula to round out the artifact directories.
FIGURE = {
    "caption": "Figure 1. Kaplan-Meier survival curves by exposure group with 95% CIs.",
    "page": 2,
}

FORMULA = {
    "latex": r"h(t \mid x) = h_0(t)\,\exp(\beta^\top x)",
    "source": "ocr",
    "valid": True,
    "coherence": 0.97,
}


def build_intermediate(slug: str = SLUG) -> dict:
    """Assemble the in-memory product (mirrors what ingest would emit)."""
    table_id = f"{slug}-t1"
    figure_id = f"{slug}-f1"
    formula_id = f"{slug}-e1"

    chunks: list[dict] = []
    for i, c in enumerate(TEXT_CHUNKS, 1):
        chunks.append(
            {
                "chunk_id": f"{slug}-c{i}",
                "doc_id": slug,
                "text": c["text"],
                "kind": "text",
                "section_path": c["section_path"],
                "parents": [],
                "page": c["page"],
            }
        )
    # One table chunk (caption is the indexable text; artifact_id links the twin).
    chunks.append(
        {
            "chunk_id": f"{slug}-c{len(TEXT_CHUNKS) + 1}",
            "doc_id": slug,
            "text": TABLE_CHUNK["caption"],
            "kind": "table",
            "section_path": TABLE_CHUNK["section_path"],
            "parents": [],
            "page": TABLE_CHUNK["page"],
            "artifact_id": table_id,
        }
    )

    document_md = _document_md(slug)

    table_json = {
        "table_id": table_id,
        "doc_id": slug,
        "page": TABLE_CHUNK["page"],
        "caption": TABLE_CHUNK["caption"],
        "columns": ["Covariate", "Hazard ratio", "95% CI", "p-value"],
        "rows": [
            ["High exposure", "1.74", "1.31-2.31", "<0.001"],
            ["Age (per 10y)", "1.22", "1.10-1.36", "<0.001"],
            ["Current smoker", "1.41", "1.05-1.89", "0.02"],
        ],
    }
    figure_json = {
        "figure_id": figure_id,
        "doc_id": slug,
        "page": FIGURE["page"],
        "has_caption": True,
        "caption": FIGURE["caption"],
    }
    formula_json = {
        "formula_id": formula_id,
        "doc_id": slug,
        **FORMULA,
    }

    meta = {
        "id": slug,
        "source": "pdf-native",
        "route": "sample-generator",
        "doi": "10.0000/sample.survival",
        "status": "ok",
        "sha256_16": "0000000000000000",
        "n_chunks": len(chunks),
        "n_tables": 1,
        "n_figures": 1,
        "n_formulas": 1,
        "group_id": slug,
        "role": "main",
        "errors": [],
    }

    return {
        "slug": slug,
        "document_md": document_md,
        "meta": meta,
        "chunks": chunks,
        "tables": {table_id: table_json},
        "figures": {figure_id: figure_json},
        "formulas": {formula_id: formula_json},
    }


def _document_md(slug: str) -> str:
    parts = ["# Survival Analysis in a Cohort Study (sample)", ""]
    seen: list[str] = []
    for c in TEXT_CHUNKS:
        heading = " > ".join(c["section_path"])
        if heading not in seen:
            seen.append(heading)
            level = "#" * (1 + len(c["section_path"]))
            parts.append(f"{level} {c['section_path'][-1]}")
            parts.append("")
        parts.append(c["text"])
        parts.append("")
    parts.append("## Tables")
    parts.append("")
    parts.append(f"See table `{slug}-t1`: {TABLE_CHUNK['caption']}")
    parts.append("")
    return "\n".join(parts)


def write_product(product: dict, outputs_dir: Path) -> Path:
    """Serialize the product into outputs/<slug>/ per OUTPUT_LAYOUT."""
    slug = product["slug"]
    root = outputs_dir / slug
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "formulas").mkdir(parents=True, exist_ok=True)

    (root / "document.md").write_text(product["document_md"], encoding="utf-8")
    (root / "document.meta.json").write_text(
        json.dumps(product["meta"], indent=2), encoding="utf-8"
    )
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in product["chunks"]:
            fh.write(json.dumps(chunk) + "\n")

    for tid, tjson in product["tables"].items():
        (root / "tables" / f"{tid}.table.json").write_text(
            json.dumps(tjson, indent=2), encoding="utf-8"
        )
    for fid, fjson in product["figures"].items():
        (root / "figures" / f"{fid}.json").write_text(
            json.dumps(fjson, indent=2), encoding="utf-8"
        )
    for eid, ejson in product["formulas"].items():
        (root / "formulas" / f"{eid}.json").write_text(
            json.dumps(ejson, indent=2), encoding="utf-8"
        )
    return root


def main() -> int:
    outputs_dir = _repo_root() / "outputs"
    product = build_intermediate(SLUG)
    root = write_product(product, outputs_dir)
    print(f"wrote sample product to {root}")
    print(
        f"  {product['meta']['n_chunks']} chunks, "
        f"{product['meta']['n_tables']} table(s), "
        f"{product['meta']['n_figures']} figure(s), "
        f"{product['meta']['n_formulas']} formula(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
