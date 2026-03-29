"""Result dataclasses for analytics metrics."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TimestepCmf:
    """CMF scores for all 5 pairs at a single timestep."""

    S_to_L: float
    S_to_V: float
    A_to_L: float
    A_to_V: float
    A_to_S: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TimestepSilhouette:
    """Silhouette coefficient at a single timestep."""

    score: float


@dataclasses.dataclass
class EpisodeAnalytics:
    """Aggregated analytics for one episode."""

    episode_id: str
    num_timesteps: int
    cmf_per_timestep: list[TimestepCmf]
    silhouette_per_timestep: list[TimestepSilhouette]

    @property
    def cmf_means(self) -> dict[str, float]:
        """Mean CMF across timesteps for each pair."""
        if not self.cmf_per_timestep:
            return {k: 0.0 for k in TimestepCmf.__dataclass_fields__}
        accum: dict[str, float] = {}
        for ts in self.cmf_per_timestep:
            for k, v in ts.as_dict().items():
                accum[k] = accum.get(k, 0.0) + v
        n = len(self.cmf_per_timestep)
        return {k: v / n for k, v in accum.items()}

    @property
    def silhouette_mean(self) -> float:
        """Mean silhouette across timesteps."""
        if not self.silhouette_per_timestep:
            return 0.0
        return sum(t.score for t in self.silhouette_per_timestep) / len(
            self.silhouette_per_timestep
        )


@dataclasses.dataclass
class AnalyticsReport:
    """Top-level report aggregating across episodes."""

    layer: int
    episodes: list[EpisodeAnalytics]

    @property
    def global_cmf(self) -> dict[str, float]:
        """Mean CMF across all episodes."""
        if not self.episodes:
            return {k: 0.0 for k in TimestepCmf.__dataclass_fields__}
        accum: dict[str, float] = {}
        for ep in self.episodes:
            for k, v in ep.cmf_means.items():
                accum[k] = accum.get(k, 0.0) + v
        n = len(self.episodes)
        return {k: v / n for k, v in accum.items()}

    @property
    def global_silhouette(self) -> float:
        """Mean silhouette across all episodes."""
        if not self.episodes:
            return 0.0
        return sum(ep.silhouette_mean for ep in self.episodes) / len(
            self.episodes
        )
