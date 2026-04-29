"""CLI: python -m nnsi --demo  |  python -m nnsi --data-dir <path>"""
import argparse
import os
import sys
from pathlib import Path

from .analyzer import NetworkAnalyzer


def main():
    parser = argparse.ArgumentParser(
        prog="python -m nnsi",
        description="NNSI: Network Node Significance Index for SDN topology optimization.",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--demo", action="store_true",
        help="Run on the bundled 5-network sample (data/topology_zoo_sample/, ~2-3 min).",
    )
    g.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to a directory of .graphml files (e.g. the 261-network Topology Zoo).",
    )
    parser.add_argument(
        "--max-new-links", type=int, default=5,
        help="Max new links the optimizer may add per network (default: 5).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Critical-node threshold as fraction of network size (default: 0.10).",
    )
    args = parser.parse_args()

    if args.demo:
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data" / "topology_zoo_sample"
    else:
        data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    graphmls = list(data_dir.glob("*.graphml"))
    if not graphmls:
        print(f"❌ No .graphml files found in: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"🧠  NNSI — running on {len(graphmls)} networks from {data_dir}\n")

    analyzer = NetworkAnalyzer(data_directory=str(data_dir))
    results = analyzer.analyze_all_networks()
    stats = results["stats"]

    print("\n" + "═" * 70)
    print("  NNSI vs IVI — Summary")
    print("═" * 70)
    print(f"  Networks analyzed:           {stats['total_networks']}")
    nnsi_pct = 100 * stats["nnsi_better"] / max(stats["total_networks"], 1)
    ivi_pct = 100 * stats["ivi_better"] / max(stats["total_networks"], 1)
    print(f"  NNSI outperforms IVI on:     {stats['nnsi_better']} ({nnsi_pct:.1f}%)")
    print(f"  IVI outperforms NNSI on:     {stats['ivi_better']} ({ivi_pct:.1f}%)")
    print(f"  Average NNSI advantage:      {stats['avg_impact_diff']:.4f}")
    print(f"  Top-node overlap (avg):      {stats['avg_overlap']:.2f}%")
    print()
    print("  Post-optimization improvements:")
    print(f"    Latency reduction:         {stats['avg_latency_reduction']:.2f}%")
    print(f"    Resilience improvement:    {stats['avg_resilience_improvement']:.2f}%")
    print(f"    Link utilization gain:     {stats['avg_link_utilization_improvement']:.2f}%")
    print("═" * 70)

    out_dir = os.path.abspath("network_analysis_results")
    print(f"\n📊 Charts + CSVs saved to: {out_dir}")


if __name__ == "__main__":
    main()
