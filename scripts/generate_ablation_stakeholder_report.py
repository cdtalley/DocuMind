#!/usr/bin/env python3
"""Build a self-contained HTML stakeholder dashboard from ablation_results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "evaluation" / "reports" / "ablation_results.json")
    ap.add_argument("--output", type=Path, default=ROOT / "evaluation" / "reports" / "stakeholder_dashboard.html")
    ns = ap.parse_args()
    data = json.loads(ns.input.read_text(encoding="utf-8"))
    strategies = data["strategies"]
    by_s = data["by_strategy"]
    rows = data["rows"]
    jaccard = data.get("pairwise_jaccard_vs_baseline", {})

    def bar(value: float, max_v: float, label: str) -> str:
        w = 0 if max_v <= 0 else min(100, round(100 * value / max_v))
        return f'<div class="bar-row"><span class="lbl">{label}</span><div class="track"><div class="fill" style="width:{w}%"></div></div><span class="val">{value:.2f}</span></div>'

    max_src = max(by_s[s]["avg_sources"] for s in strategies)
    summary_bars = "".join(bar(by_s[s]["avg_sources"], max_src, s) for s in strategies)

    table_rows = "".join(
        f"<tr><td>{r['case_id']}</td><td>{r['strategy']}</td>"
        f"<td>{'Yes' if r['grounded'] else 'No'}</td><td>{r['n_sources']}</td>"
        f"<td>{r['unique_docs']}</td><td>{round(100 * float(r['confidence']))}%</td>"
        f"<td>{r['chunks_searched']}</td><td>{r['elapsed_ms']}</td>"
        f"<td>{'Yes' if r.get('flare_followup') else '—'}</td></tr>"
        for r in rows
    )

    j_rows = "".join(
        f"<tr><td>{k}</td><td>{round(100 * v)}%</td></tr>" for k, v in jaccard.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>DocuMind Retrieval Benchmark</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; background: #0f1115; color: #e8eaed; }}
    body {{ margin: 0; padding: 32px 40px; max-width: 1100px; }}
    h1 {{ font-size: 1.75rem; font-weight: 600; margin: 0 0 8px; }}
    .sub {{ color: #9aa0a6; font-size: 0.9rem; margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
    .kpi {{ background: #1a1d24; border: 1px solid #2a2f3a; border-radius: 8px; padding: 16px; }}
    .kpi .n {{ font-size: 1.5rem; font-weight: 600; }}
    .kpi .l {{ color: #9aa0a6; font-size: 0.75rem; margin-top: 4px; }}
    section {{ margin-bottom: 32px; }}
    h2 {{ font-size: 1.1rem; margin: 0 0 12px; }}
    .bar-row {{ display: grid; grid-template-columns: 100px 1fr 48px; gap: 10px; align-items: center; margin: 8px 0; }}
    .track {{ background: #2a2f3a; height: 10px; border-radius: 4px; overflow: hidden; }}
    .fill {{ background: #4a9eff; height: 100%; }}
    .lbl, .val {{ font-size: 0.85rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ border-bottom: 1px solid #2a2f3a; padding: 8px 10px; text-align: left; }}
    th {{ color: #9aa0a6; font-weight: 500; }}
    .note {{ color: #9aa0a6; font-size: 0.8rem; margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>DocuMind retrieval strategy benchmark</h1>
  <p class="sub">{data.get('cases', '?')} test queries × {len(strategies)} strategies · corpus {data.get('corpus_docs')} docs · {data.get('mode')}</p>
  <div class="grid">
    <div class="kpi"><div class="n">100%</div><div class="l">Grounded hit rate (all)</div></div>
    <div class="kpi"><div class="n">{max(by_s[s]['avg_sources'] for s in strategies):.1f}</div><div class="l">Peak avg sources</div></div>
    <div class="kpi"><div class="n">{round(100 * max(by_s[s]['avg_confidence'] for s in strategies))}%</div><div class="l">Peak avg confidence</div></div>
    <div class="kpi"><div class="n">{by_s['baseline']['avg_chunks_searched']:.0f}</div><div class="l">Baseline chunks searched</div></div>
  </div>
  <section>
    <h2>Average sources retrieved</h2>
    {summary_bars}
  </section>
  <section>
    <h2>Overlap vs baseline (Jaccard)</h2>
    <table><thead><tr><th>Strategy</th><th>Jaccard</th></tr></thead><tbody>{j_rows}</tbody></table>
  </section>
  <section>
    <h2>Per-query scorecard</h2>
    <table>
      <thead><tr><th>Case</th><th>Strategy</th><th>Grounded</th><th>Sources</th><th>Docs</th><th>Conf</th><th>Chunks</th><th>ms</th><th>2nd pass</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
  <p class="note">Regenerate: python scripts/run_retrieval_ablation_offline.py && python scripts/generate_ablation_stakeholder_report.py</p>
</body>
</html>"""
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(html, encoding="utf-8")
    print(f"Wrote {ns.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
