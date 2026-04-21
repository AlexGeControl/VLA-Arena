"""CLI orchestrator for Track D analytics.

Scans HDF5 episode files, computes CMF and Silhouette metrics per timestep,
aggregates per-episode and globally, and writes a structured YAML report.

Usage::

    cd <openpi_root>
    conda run -n openpi-vla-arena python \\
      src/openpi_interpret/analytics/run_analytics.py \\
        --data-dir src/openpi_interpret/data \\
        --output analytics_report.yaml \\
        --layer 17
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from analytics.cmf import compute_all_cmf
from analytics.constants import DEFAULT_CMF_LAYER, SAMPLED_LAYERS
from analytics.reader import AnalyticsReader, scan_episodes
from analytics.silhouette import compute_silhouette
from analytics.types import (
    AnalyticsReport, EpisodeAnalytics, LayerHeadVisualAttention,
    LayerSilhouette, TimestepCmf, TimestepSilhouette,
)
from analytics.visual_attention import compute_visual_attention, UNIFORM_VISUAL_BASELINE

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMF and Silhouette metrics on extracted HDF5 data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory containing .h5 episode files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analytics_report.yaml",
        help="Path for the YAML report output.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_CMF_LAYER,
        help=f"Transformer layer for metrics (sampled set: {SAMPLED_LAYERS}).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )
    return parser.parse_args()


def process_episode(
    reader: AnalyticsReader,
    layer: int,
) -> EpisodeAnalytics:
    """Compute all metrics for a single episode.

    Args:
        reader: AnalyticsReader for the episode HDF5.
        layer: Transformer layer to use.

    Returns:
        EpisodeAnalytics with per-timestep CMF and Silhouette scores.
    """
    ep_id = reader.episode_id
    n_ts = reader.num_timesteps
    logger.info("Processing %s (%d timesteps)", ep_id, n_ts)

    cmf_results: list[TimestepCmf] = []
    sil_results: list[TimestepSilhouette] = []

    has_cmf = reader.has_cmf_attended()
    if not has_cmf:
        logger.warning(
            "Episode %s has no cmf_attended data — CMF scores will be skipped. "
            "Re-run extraction to populate this group.",
            ep_id,
        )

    for t in range(n_ts):
        if has_cmf:
            _, q_suffix = reader.get_q_projections(t, layer)
            attended_lang, attended_vis = reader.get_cmf_attended(t, layer)
            cmf = compute_all_cmf(q_suffix, attended_lang, attended_vis)
            cmf_results.append(cmf)

        tsne_coords = reader.get_tsne(t, layer)
        sil = compute_silhouette(tsne_coords)
        sil_results.append(sil)

        logger.debug(
            "  ts=%d  sil=%.3f%s",
            t,
            sil.score,
            f"  cmf_A_to_L={cmf.A_to_L:.3f}" if has_cmf else "",
        )

    visual_attn_by_layer: list[LayerHeadVisualAttention] = []
    sil_by_layer: list[LayerSilhouette] = []

    for l in SAMPLED_LAYERS:
        layer_head_accum: list[np.ndarray] = []
        layer_sil_accum: list[float] = []

        for t in range(n_ts):
            attn = reader.get_attention(t, l)
            layer_head_accum.append(compute_visual_attention(attn))

            tsne_l = reader.get_tsne(t, l)
            layer_sil_accum.append(compute_silhouette(tsne_l).score)

        mean_heads = np.stack(layer_head_accum).mean(axis=0)
        visual_attn_by_layer.append(LayerHeadVisualAttention(
            layer=l,
            head_scores=mean_heads.tolist(),
            layer_mean=float(mean_heads.mean()),
        ))
        sil_by_layer.append(LayerSilhouette(
            layer=l,
            score=float(np.mean(layer_sil_accum)),
        ))

    return EpisodeAnalytics(
        episode_id=ep_id,
        num_timesteps=n_ts,
        cmf_per_timestep=cmf_results,
        silhouette_per_timestep=sil_results,
        visual_attention_by_layer=visual_attn_by_layer,
        silhouette_by_layer=sil_by_layer,
    )


def build_report(
    episodes: list[EpisodeAnalytics],
    layer: int,
) -> AnalyticsReport:
    return AnalyticsReport(layer=layer, episodes=episodes)


def report_to_dict(report: AnalyticsReport) -> dict:
    """Serialize an AnalyticsReport to a YAML-friendly dict."""
    episode_dicts = []
    for ep in report.episodes:
        ep_dict: dict = {
            "episode_id": ep.episode_id,
            "num_timesteps": ep.num_timesteps,
        }
        if ep.cmf_per_timestep:
            ep_dict["cmf"] = {}
            means = ep.cmf_means
            for pair_name in TimestepCmf.__dataclass_fields__:
                ep_dict["cmf"][pair_name] = {
                    "mean": round(means[pair_name], 4),
                    "per_timestep": [
                        round(getattr(ts, pair_name), 4)
                        for ts in ep.cmf_per_timestep
                    ],
                }
        ep_dict["silhouette"] = {
            "mean": round(ep.silhouette_mean, 4),
            "per_timestep": [
                round(ts.score, 4)
                for ts in ep.silhouette_per_timestep
            ],
        }
        episode_dicts.append(ep_dict)

    result: dict = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "layer": report.layer,
            "num_episodes": len(report.episodes),
        },
        "global": {
            "silhouette": round(report.global_silhouette, 4),
        },
        "episodes": episode_dicts,
    }
    if report.episodes and report.episodes[0].cmf_per_timestep:
        result["global"]["cmf"] = {
            k: round(v, 4) for k, v in report.global_cmf.items()
        }

    ranking = report.global_visual_attention_ranking
    if ranking:
        result["visual_attention_ranking"] = {
            "uniform_baseline": round(UNIFORM_VISUAL_BASELINE, 4),
            "global_mean": round(
                sum(e["layer_mean"] for e in ranking) / len(ranking), 4
            ),
            "layers": [
                {
                    "layer": e["layer"],
                    "layer_mean": round(e["layer_mean"], 4),
                    "heads": [
                        {"head": h["head"], "visual_share": round(h["visual_share"], 4)}
                        for h in e["heads"]
                    ],
                }
                for e in ranking
            ],
        }

    sil_profile = report.global_silhouette_profile
    if sil_profile:
        result["silhouette_profile"] = {
            "layers": [
                {"layer": ls.layer, "score": round(ls.score, 4)}
                for ls in sil_profile
            ],
        }

    return result


def write_yaml(report_dict: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(report_dict, f, default_flow_style=False, sort_keys=False)
    logger.info("Report written to %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.layer not in SAMPLED_LAYERS:
        logger.error("Layer %d not in sampled set %s", args.layer, SAMPLED_LAYERS)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    readers = scan_episodes(data_dir)
    if not readers:
        logger.error("No .h5 files found in %s", data_dir)
        sys.exit(1)
    logger.info("Found %d episode(s) in %s", len(readers), data_dir)

    episode_results = [
        process_episode(reader, args.layer) for reader in readers
    ]

    report = build_report(episode_results, args.layer)
    report_dict = report_to_dict(report)
    write_yaml(report_dict, Path(args.output))

    logger.info("=== Analytics Summary ===")
    logger.info("Global silhouette: %.4f", report.global_silhouette)
    if report.episodes and report.episodes[0].cmf_per_timestep:
        for k, v in report.global_cmf.items():
            logger.info("Global CMF %s: %.4f", k, v)

    ranking = report.global_visual_attention_ranking
    if ranking:
        logger.info("=== Visual Attention Ranking (uniform baseline=%.4f) ===",
                     UNIFORM_VISUAL_BASELINE)
        for e in ranking:
            head_strs = [f"h{h['head']}={h['visual_share']:.3f}" for h in e["heads"]]
            logger.info("  layer %2d: mean=%.4f  [%s]",
                        e["layer"], e["layer_mean"], ", ".join(head_strs))

    sil_profile = report.global_silhouette_profile
    if sil_profile:
        logger.info("=== Silhouette Profile (per layer) ===")
        for ls in sil_profile:
            logger.info("  layer %2d: %.4f", ls.layer, ls.score)


if __name__ == "__main__":
    main()
