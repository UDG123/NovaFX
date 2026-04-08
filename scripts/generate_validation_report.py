#!/usr/bin/env python3
"""Generate markdown report from extended validation JSON.

Usage:
    python scripts/generate_validation_report.py results/extended_validation.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "results/extended_validation.json"
    output_path = Path(input_path).with_suffix(".md").name
    output_path = Path("results") / output_path.replace("extended_validation", "validation_report")

    with open(input_path) as f:
        data = json.load(f)

    meta = data["meta"]
    results = data["results"]

    sig = [r for r in results if r.get("statistically_significant")]
    rob = [r for r in results if r.get("robust")]
    mc_ran = [r for r in results if r.get("mc_ran")]

    lines = [
        "# NovaFX Extended Validation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Data range | {meta['range']} |",
        f"| Assets tested | {meta['n_assets']} |",
        f"| Strategies tested | {meta['n_strategies']} |",
        f"| Total combos evaluated | {meta['n_results']} |",
        f"| Skipped (blacklisted/few trades) | {meta['n_skipped']} |",
        f"| Errors | {meta['n_errors']} |",
        f"| Statistically significant (N>=50) | {len(sig)} |",
        f"| Robust (MC score>=70) | {len(rob)} |",
        f"| Monte Carlo tested | {len(mc_ran)} |",
        "",
    ]

    # Top 10 by Sharpe
    by_sharpe = sorted(results, key=lambda x: x.get("sharpe", -99), reverse=True)[:10]
    lines.append("## Top 10 by Sharpe Ratio")
    lines.append("")
    lines.append("| Strategy | Symbol | Sharpe | Return | WR | Trades | MC Sharpe 95%CI | Robust |")
    lines.append("|----------|--------|--------|--------|----|--------|-----------------|--------|")
    for r in by_sharpe:
        ci = ""
        if r.get("mc_ran"):
            ci = f"[{r.get('mc_sharpe_p5', 0):.2f}, {r.get('mc_sharpe_p95', 0):.2f}]"
        rob_tag = "Yes" if r.get("robust") else "No"
        lines.append(f"| {r['strategy']} | {r['symbol']} | {r['sharpe']:.3f} | "
                     f"{r['total_return']:.2%} | {r['win_rate']:.1%} | {r['n_trades']} | {ci} | {rob_tag} |")
    lines.append("")

    # Top 10 by Return
    by_return = sorted(results, key=lambda x: x.get("total_return", -99), reverse=True)[:10]
    lines.append("## Top 10 by Total Return")
    lines.append("")
    lines.append("| Strategy | Symbol | Return | Sharpe | MaxDD | Trades | P(loss) | Robust |")
    lines.append("|----------|--------|--------|--------|-------|--------|---------|--------|")
    for r in by_return:
        p_loss = f"{r.get('mc_p_loss', 0):.0%}" if r.get("mc_ran") else "N/A"
        rob_tag = "Yes" if r.get("robust") else "No"
        lines.append(f"| {r['strategy']} | {r['symbol']} | {r['total_return']:.2%} | "
                     f"{r['sharpe']:.3f} | {r['max_drawdown']:.2%} | {r['n_trades']} | {p_loss} | {rob_tag} |")
    lines.append("")

    # Recommendations
    paper_ready = [r for r in results if r.get("robust") and r.get("statistically_significant")]
    lines.append("## Recommendations: Ready for Paper Trading")
    lines.append("")
    if paper_ready:
        lines.append(f"{len(paper_ready)} combos meet both criteria (robust + statistically significant):")
        lines.append("")
        lines.append("| Strategy | Symbol | Sharpe | Return | WR | Trades |")
        lines.append("|----------|--------|--------|--------|----|--------|")
        for r in sorted(paper_ready, key=lambda x: x["sharpe"], reverse=True):
            lines.append(f"| {r['strategy']} | {r['symbol']} | {r['sharpe']:.3f} | "
                         f"{r['total_return']:.2%} | {r['win_rate']:.1%} | {r['n_trades']} |")
    else:
        lines.append("No combos meet both criteria. Consider:")
        lines.append("- Extending data range for more trades")
        lines.append("- Relaxing significance threshold to N>=30")
    lines.append("")

    # Strategy ranking
    strat_avg: dict[str, list] = {}
    for r in results:
        s = r["strategy"]
        strat_avg.setdefault(s, []).append(r.get("sharpe", 0))
    lines.append("## Strategy Ranking (Avg Sharpe Across Assets)")
    lines.append("")
    lines.append("| Strategy | Avg Sharpe | N Combos |")
    lines.append("|----------|-----------|----------|")
    for s, vals in sorted(strat_avg.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True):
        avg = sum(vals) / len(vals)
        lines.append(f"| {s} | {avg:.3f} | {len(vals)} |")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated from {input_path}*")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to {output_path}")
    print(report)


if __name__ == "__main__":
    main()
