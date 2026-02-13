#!/usr/bin/env python3
"""Display benchmark results with paper comparisons.

Scans a results/ tree for BoxingGym JSON outputs, prints a concise table
with raw and standardized errors. For Discovery@10 (include_prior=True) it
reports the paper reference and the delta to the paper.

Usage:
  uv run python scripts/show_benchmarks.py [--root results] [--csv out.csv] [--html out.html]

Examples:
  uv run python scripts/show_benchmarks.py                            # pretty table
  uv run python scripts/show_benchmarks.py --csv outputs/bench.csv   # also save CSV
  uv run python scripts/show_benchmarks.py --html outputs/bench.html # save HTML table
"""

import argparse
import csv
import html
import json
import os
from collections.abc import Iterable
from typing import Any

PAPER_DISCOVERY10_PRIOR = {
    # Discovery@10 (with prior) values from the paper's LaTeX tables.
    "dugongs": (-0.06, 0.04),
    "peregrines": (-0.65, 0.02),
    "lotka_volterra": (-0.01, 0.12),
}


def get_norm_factors(env_name: str) -> tuple[float | None, float | None]:
    # Static map to avoid importing envs (which may need API keys) during aggregation
    STATIC = {
        "dugongs": (0.9058681693402041, 9.234192516908691),
        "peregrines": (10991.5464, 15725.115658658306),
        "lotka_volterra": (8.327445247142364, 17.548285564117467),
        "hyperbolic_temporal_discount": (0.25, 4.3),
        "irt": (0.5, 0.5),
        "survival": (0.2604, 0.43885286828275377),
        "location_finding": (1.57, 1.15385),
        "death_process": (0.2902838787350395, 1.756991075450312),
        "emotion": (1.58508525, 0.7237143937677741),
        "morals": (0.424, 0.494190246767376),
        "moral_machines": (0.424, 0.494190246767376),
    }
    return STATIC.get(env_name, (None, None))


def iter_json_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                yield os.path.join(dirpath, fn)


def load_result(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def standardize(
    err_mean: float, err_std: float, mu0: float | None, sigma0: float | None
) -> tuple[float | None, float | None]:
    if sigma0 in (None, 0):
        return None, None
    return (err_mean - mu0) / sigma0, err_std / sigma0


def _standardize_with_norm_factors(
    err_mean: float,
    err_std: float,
    norm_factors: Any,
) -> tuple[float | None, float | None]:
    if not isinstance(norm_factors, dict):
        return None, None
    mu0 = norm_factors.get("mu")
    sigma0 = norm_factors.get("sigma")
    if mu0 is None or sigma0 is None:
        return None, None
    try:
        sigma = float(sigma0)
        if sigma <= 0:
            return None, None
        mu = float(mu0)
        return (float(err_mean) - mu) / sigma, float(err_std) / sigma
    except Exception:
        return None, None


def row_to_str(row: dict) -> str:
    budget = row.get("budget")
    budget_str = f"{budget:>3}" if budget is not None else "  -"
    paper = row.get("paper_discovery10_mean")
    paper_str = f" {paper}±{row.get('paper_discovery10_se')}" if paper else "   n/a"
    delta = row.get("delta_vs_paper") or ""
    return (
        f"{row['env']:<20} {row['model']:<28} b={budget_str} "
        f"raw={row['raw_mean']}±{row['raw_std']}  z={row['z_mean']}±{row['z_std']}  "
        f"paper={paper_str}  Δ={delta}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Show benchmark results with paper comparisons.")
    ap.add_argument(
        "root", nargs="?", default="results", help="Root folder to scan (default: results)"
    )
    ap.add_argument("--csv", dest="csv_out", default=None, help="Optional path to also write CSV")
    ap.add_argument(
        "--html", dest="html_out", default=None, help="Optional path to also write HTML"
    )
    return ap.parse_args()


def extract_z_by_budget(z_results_precomputed: list) -> dict[int, dict[str, Any]]:
    z_by_budget: dict[int, dict[str, Any]] = {}
    for zr in z_results_precomputed:
        if not isinstance(zr, dict):
            continue
        b = zr.get("budget")
        if b is None:
            continue
        try:
            z_by_budget[int(b)] = zr
        except Exception:
            continue
    return z_by_budget


def _coerce_budget(value: Any) -> int | float | None:
    """Convert budget-like values to numeric for sorting/formatting."""
    if value is None:
        return None
    try:
        b = float(value)
    except Exception:
        return None
    return int(b) if b.is_integer() else b


def compute_z_scores(
    err_mean: float | None,
    err_std: float | None,
    z_entry: dict | None,
    norm_factors: dict,
    mu0: float | None,
    sigma0: float | None,
) -> tuple[float | None, float | None]:
    """Compute z_mean and z_std from various sources."""
    z_mean = z_std = None

    if isinstance(z_entry, dict):
        z_mean = z_entry.get("z_mean")
        z_std = z_entry.get("z_std")

    if (z_mean is None or z_std is None) and err_mean is not None and err_std is not None:
        z_mean, z_std = _standardize_with_norm_factors(
            float(err_mean), float(err_std), norm_factors
        )

    if (z_mean is None or z_std is None) and err_mean is not None and err_std is not None:
        z_mean, z_std = standardize(float(err_mean), float(err_std), mu0, sigma0)

    return z_mean, z_std


def process_result_file(path: str, blob: dict) -> list[dict]:
    rows = []

    cfg = blob.get("config", {})
    envs_raw = cfg.get("envs")
    env_cfg = envs_raw if isinstance(envs_raw, dict) else {}
    exp_raw = cfg.get("exp")
    exp_cfg = exp_raw if isinstance(exp_raw, dict) else {}
    env_name = env_cfg.get("env_name")
    goal_name = env_cfg.get("goal_name")
    include_prior = bool(cfg.get("include_prior", False))
    llms_raw = cfg.get("llms")
    if isinstance(llms_raw, dict):
        model_name = llms_raw.get("model_name")
    elif isinstance(llms_raw, str):
        model_name = llms_raw
    else:
        model_name = None
    seed = cfg.get("seed")
    use_ppl = bool(cfg.get("use_ppl", False))
    experiment_type = exp_cfg.get("experiment_type")
    budgets = exp_cfg.get("num_experiments")

    if not env_name or not goal_name:
        return rows

    mu0, sigma0 = get_norm_factors(env_name)
    data = blob.get("data") or {}
    raw_results = data.get("results") or []
    z_results_precomputed = data.get("z_results") or []
    norm_factors = data.get("norm_factors") or {}

    z_by_budget = extract_z_by_budget(z_results_precomputed)

    if not raw_results:
        return rows

    for i, entry in enumerate(raw_results):
        if not entry or not isinstance(entry, list) or not entry[0]:
            continue

        try:
            err_mean, err_std = entry[0]
        except Exception:
            err_mean = err_std = None

        budget_raw = None
        if isinstance(budgets, list) and i < len(budgets):
            budget_raw = budgets[i]
        elif isinstance(budgets, (int, float)):
            budget_raw = budgets

        budget = _coerce_budget(budget_raw)

        # Find matching z_entry
        z_entry = None
        if budget is not None and z_by_budget:
            try:
                z_entry = z_by_budget.get(int(budget))
            except Exception:
                z_entry = None
        if (
            z_entry is None
            and (budget is None or not z_by_budget)
            and z_results_precomputed
            and i < len(z_results_precomputed)
        ):
            z_entry = z_results_precomputed[i]

        z_mean, z_std = compute_z_scores(err_mean, err_std, z_entry, norm_factors, mu0, sigma0)

        if isinstance(z_entry, dict) and budget is None:
            budget = _coerce_budget(z_entry.get("budget"))

        # Check for paper comparison
        paper_mean = paper_se = delta_vs_paper = None
        budget_is_10 = isinstance(budget, (int, float)) and int(budget) == 10
        if (
            goal_name == "direct_naive"
            and include_prior
            and budget_is_10
            and env_name in PAPER_DISCOVERY10_PRIOR
            and z_mean is not None
        ):
            paper_mean, paper_se = PAPER_DISCOVERY10_PRIOR[env_name]
            delta_vs_paper = z_mean - paper_mean

        rows.append(
            {
                "path": path,
                "env": env_name,
                "goal": goal_name,
                "experiment_type": experiment_type,
                "include_prior": include_prior,
                "use_ppl": use_ppl,
                "model": model_name,
                "seed": seed,
                "budget": budget,
                "raw_mean": f"{err_mean:.6g}" if err_mean is not None else "",
                "raw_std": f"{err_std:.6g}" if err_std is not None else "",
                "z_mean": f"{z_mean:.6g}" if z_mean is not None else "",
                "z_std": f"{z_std:.6g}" if z_std is not None else "",
                "paper_discovery10_mean": f"{paper_mean:.6g}" if paper_mean is not None else "",
                "paper_discovery10_se": f"{paper_se:.6g}" if paper_se is not None else "",
                "delta_vs_paper": f"{delta_vs_paper:.6g}" if delta_vs_paper is not None else "",
            }
        )

    return rows


def collect_results(root: str) -> list[dict]:
    rows = []
    for path in iter_json_files(root):
        blob = load_result(path)
        if not blob or not isinstance(blob, dict):
            continue
        rows.extend(process_result_file(path, blob))
    return rows


def print_results(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return

    print(
        "ENV                  MODEL                        BGT raw±std        z±std          paper      Δ"
    )
    print("-" * 110)

    def _sort_key(r: dict) -> tuple[str, float, str]:
        b_num = _coerce_budget(r.get("budget"))
        b_key = float(b_num) if b_num is not None else float("inf")
        return (r["env"], b_key, r.get("model") or "")

    for row in sorted(rows, key=_sort_key):
        print(row_to_str(row))


def write_csv(rows: list[dict], csv_out: str) -> None:
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    fieldnames = [
        "path",
        "env",
        "goal",
        "experiment_type",
        "include_prior",
        "use_ppl",
        "model",
        "seed",
        "budget",
        "raw_mean",
        "raw_std",
        "z_mean",
        "z_std",
        "paper_discovery10_mean",
        "paper_discovery10_se",
        "delta_vs_paper",
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {csv_out}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>BoxingGym Benchmarks</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22d3ee;
      --accent2: #a855f7;
      --border: #1f2937;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; background: radial-gradient(circle at 20% 20%, rgba(34,211,238,0.08), transparent 25%),
                                  radial-gradient(circle at 80% 10%, rgba(168,85,247,0.08), transparent 25%),
                                  linear-gradient(145deg, #0b1221, #0d1628);
            color: var(--text); font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
    h1 {{ margin: 0 0 12px; font-weight: 700; letter-spacing: -0.01em; }}
    .muted {{ color: var(--muted); font-size: 0.95rem; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.35); }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); align-items: end; }}
    label {{ display: block; margin-bottom: 6px; color: var(--muted); font-size: 0.85rem; }}
    select, input {{ width: 100%; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border); background: #0b1221; color: var(--text); }}
    input::placeholder {{ color: #64748b; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ position: sticky; top: 0; background: #0c1322; font-weight: 600; cursor: pointer; }}
    tr:hover td {{ background: rgba(34,211,238,0.06); }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; border: 1px solid var(--border); color: var(--muted); }}
    .pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; border: 1px solid var(--border); }}
    .pill.blue {{ color: #38bdf8; border-color: rgba(56,189,248,0.4); }}
    .pill.purple {{ color: #c084fc; border-color: rgba(192,132,252,0.4); }}
    .pill.green {{ color: #34d399; border-color: rgba(52,211,153,0.4); }}
    .delta-pos {{ color: #34d399; font-weight: 600; }}
    .delta-neg {{ color: #f87171; font-weight: 600; }}
    .paper {{ color: #a5b4fc; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>BoxingGym Benchmarks</h1>
    <div class="muted">Total rows: {total_rows} &nbsp;·&nbsp; Click headers to sort, use filters/search to narrow.</div>
    <div class="grid" style="margin-top:16px;">
      <div>
        <label>Env</label>
        <select id="f-env">{env_opts}</select>
      </div>
      <div>
        <label>Model</label>
        <select id="f-model">{model_opts}</select>
      </div>
      <div>
        <label>Budget</label>
        <select id="f-budget">{budget_opts}</select>
      </div>
      <div>
        <label>Search (path / goal / exp)</label>
        <input id="f-search" type="text" placeholder="substring match..." />
      </div>
      <div>
        <label>Include prior</label>
        <select id="f-prior">
          <option value=''>Any</option>
          <option value='True'>True</option>
          <option value='False'>False</option>
        </select>
      </div>
      <div>
        <label>Paper baseline</label>
        <select id="f-paper">
          <option value=''>All rows</option>
          <option value='1'>Only rows with paper ref</option>
          <option value='0'>Hide rows with paper ref</option>
        </select>
      </div>
    </div>
    <table id="bench-table">
      <thead><tr>{header_html}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
  <script>
    const tbl = document.getElementById('bench-table');
    const rows = Array.from(tbl.tBodies[0].rows);
    let sortDir = 1, sortIdx = 0;

    function applyFilters() {{
      const env = document.getElementById('f-env').value;
      const model = document.getElementById('f-model').value;
      const budget = document.getElementById('f-budget').value;
      const prior = document.getElementById('f-prior').value;
      const paper = document.getElementById('f-paper').value;
      const q = document.getElementById('f-search').value.toLowerCase();
      rows.forEach(r => {{
        const matchEnv = !env || r.dataset.env === env;
        const matchModel = !model || r.dataset.model === model;
        const matchBudget = !budget || r.dataset.budget === budget;
        const matchPrior = !prior || r.dataset.prior === prior;
        const matchPaper = !paper || r.dataset.paper === paper;
        const text = r.innerText.toLowerCase();
        const matchSearch = !q || text.includes(q);
        r.style.display = (matchEnv && matchModel && matchBudget && matchPrior && matchPaper && matchSearch) ? '' : 'none';
      }});
    }}

    function sortBy(idx) {{
      const body = tbl.tBodies[0];
      sortDir = (sortIdx === idx) ? -sortDir : 1;
      sortIdx = idx;
      const collator = new Intl.Collator(undefined, {{numeric:true, sensitivity:'base'}});
      const sorted = rows.slice().sort((a,b)=>{{
        return collator.compare(a.cells[idx].innerText.trim(), b.cells[idx].innerText.trim()) * sortDir;
      }});
      sorted.forEach(r => body.appendChild(r));
      applyFilters();
    }}

    document.querySelectorAll('#bench-table th').forEach((th, idx)=>{{
      th.addEventListener('click', ()=> sortBy(idx));
    }});

    ['f-env','f-model','f-budget','f-prior','f-paper','f-search'].forEach(id=>{{
      document.getElementById(id).addEventListener('input', applyFilters);
      document.getElementById(id).addEventListener('change', applyFilters);
    }});

    applyFilters();
  </script>
</body>
</html>"""


def _td(val: str) -> str:
    """Escape and wrap value in td tag."""
    return f"<td>{html.escape(str(val))}</td>"


def _build_row_html(r: dict) -> str:
    """Build HTML for a single table row."""
    paper_present = bool(r.get("paper_discovery10_mean"))
    delta = r.get("delta_vs_paper", "")
    delta_class = ""
    if delta:
        try:
            dval = float(delta)
            delta_class = "delta-pos" if dval <= 0 else "delta-neg"
        except Exception:
            pass

    cells = [
        _td(r.get("env", "")),
        _td(r.get("model", "")),
        _td(r.get("budget", "")),
        _td(r.get("raw_mean", "")),
        _td(r.get("raw_std", "")),
        _td(r.get("z_mean", "")),
        _td(r.get("z_std", "")),
        f"<td class='paper'>{html.escape(r.get('paper_discovery10_mean', ''))}</td>",
        f"<td class='paper'>{html.escape(r.get('paper_discovery10_se', ''))}</td>",
        f"<td class='{delta_class}'>{html.escape(delta)}</td>",
        _td(r.get("seed", "")),
        _td(r.get("include_prior", "")),
        _td(r.get("use_ppl", "")),
        _td(r.get("goal", "")),
        _td(r.get("experiment_type", "")),
        _td(r.get("path", "")),
    ]

    return (
        f"<tr data-env='{html.escape(str(r.get('env', '')))}' "
        f"data-model='{html.escape(str(r.get('model', '')))}' "
        f"data-budget='{html.escape(str(r.get('budget', '')))}' "
        f"data-prior='{html.escape(str(r.get('include_prior', '')))}' "
        f"data-paper={'1' if paper_present else '0'}>" + "".join(cells) + "</tr>"
    )


def write_html(rows: list[dict], html_out: str) -> None:
    """Write results to HTML file with filters and sorting."""
    os.makedirs(os.path.dirname(html_out) or ".", exist_ok=True)

    header_cells = [
        "Env",
        "Model",
        "Budget",
        "raw mean",
        "raw std",
        "z mean",
        "z std",
        "Paper mean",
        "Paper se",
        "Δ vs paper",
        "Seed",
        "Include prior",
        "Use PPL",
        "Goal",
        "Exp type",
        "Path",
    ]

    env_options = sorted({r.get("env", "") for r in rows})
    model_options = sorted({r.get("model", "") for r in rows})
    budget_options = sorted({str(r.get("budget")) for r in rows if r.get("budget") is not None})

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["env"],
            float(_coerce_budget(r.get("budget")))
            if _coerce_budget(r.get("budget")) is not None
            else float("inf"),
            r.get("model") or "",
        ),
    )
    rows_html = "".join(_build_row_html(r) for r in sorted_rows)

    env_opts_html = "<option value=''>All envs</option>" + "".join(
        f"<option value='{html.escape(e)}'>{html.escape(e)}</option>" for e in env_options
    )
    model_opts_html = "<option value=''>All models</option>" + "".join(
        f"<option value='{html.escape(m)}'>{html.escape(m)}</option>" for m in model_options
    )
    budget_opts_html = "<option value=''>All budgets</option>" + "".join(
        f"<option value='{html.escape(b)}'>{html.escape(b)}</option>" for b in budget_options
    )

    header_html = "".join(
        f"<th data-key='{html.escape(h.lower().replace(' ', '_'))}'>{html.escape(h)}</th>"
        for h in header_cells
    )

    html_doc = HTML_TEMPLATE.format(
        total_rows=len(rows),
        env_opts=env_opts_html,
        model_opts=model_opts_html,
        budget_opts=budget_opts_html,
        header_html=header_html,
        rows_html=rows_html,
    )

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"HTML written to {html_out}")


def main():
    """Main entry point for benchmark display."""
    args = parse_args()
    rows = collect_results(args.root)

    print_results(rows)

    if not rows:
        return

    if args.csv_out:
        write_csv(rows, args.csv_out)

    if args.html_out:
        write_html(rows, args.html_out)


if __name__ == "__main__":
    main()
