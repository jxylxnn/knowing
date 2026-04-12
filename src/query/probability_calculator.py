import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from scipy import stats
import numpy as np


@dataclass
class ProbabilityResult:
    player_name: str
    stat: str
    line: float
    mean: float
    std: float
    prob_over: float
    prob_under: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    opponent: Optional[str] = None
    date: Optional[str] = None
    play_probability: float = 1.0
    recommendation: Optional[str] = None
    team: Optional[str] = None
    is_home: bool = False
    recent_games: Optional[list] = None
    recent_avg: Optional[dict] = None
    matchup_history: Optional[list] = None
    matchup_avg: Optional[dict] = None
    opponent_defense: Optional[dict] = None
    trend: Optional[dict] = None
    z_score: Optional[float] = None
    edge: Optional[float] = None
    base_mean: Optional[float] = None
    adjustments: Optional[List[Tuple[str, Any]]] = None
    num_sims: Optional[int] = None
    probability_method: Optional[str] = None
    fallback_used: bool = False
    sample_count: int = 0
    calibration_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        edge = abs(self.prob_over - 0.5)
        self.edge = edge
        if self.std and self.std > 0:
            self.z_score = (self.line - self.mean) / self.std
        else:
            self.z_score = None
            
        if edge > 0.15:
            if self.prob_over > 0.5:
                self.recommendation = "OVER (strong)"
            else:
                self.recommendation = "UNDER (strong)"
        elif edge > 0.05:
            if self.prob_over > 0.5:
                self.recommendation = "OVER (moderate)"
            else:
                self.recommendation = "UNDER (moderate)"
        else:
            self.recommendation = "PASS (too close)"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_name': self.player_name,
            'stat': self.stat,
            'line': self.line,
            'mean': self.mean,
            'std': self.std,
            'prob_over': self.prob_over,
            'prob_under': self.prob_under,
            'ci_low': self.ci_low,
            'ci_high': self.ci_high,
            'opponent': self.opponent,
            'date': self.date,
            'play_probability': self.play_probability,
            'recommendation': self.recommendation,
            'z_score': self.z_score,
            'edge': self.edge,
            'num_sims': self.num_sims,
            'probability_method': self.probability_method,
            'fallback_used': self.fallback_used,
            'sample_count': self.sample_count,
            'calibration_metrics': self.calibration_metrics,
        }


@dataclass
class DistributionSpec:
    stat: str
    family: str
    is_count: bool
    fallback_used: bool
    reason: str
    sample_count: int = 0
    zero_rate: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    use_empirical: bool = False
    samples: Optional[np.ndarray] = None
    play_probability: float = 1.0
    context_adjustment: float = 1.0


@dataclass
class ProbabilityComputation:
    prob_over: float
    prob_under: float
    mean: float
    std: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    method: str
    fallback_used: bool
    sample_count: int
    reason: str
    calibration_metrics: Optional[Dict[str, Any]] = None


class ProbabilityCalculator:
    STAT_STEP_SIZES = {
        'pts': 1,
        'reb': 1,
        'ast': 1,
        'stl': 1,
        'blk': 1,
        'tov': 1,
    }

    STAT_DISPLAY_NAMES = {
        'pts': 'Points',
        'reb': 'Rebounds',
        'ast': 'Assists',
        'stl': 'Steals',
        'blk': 'Blocks',
        'tov': 'Turnovers',
    }

    COUNT_STATS = {'stl', 'blk', 'tov'}
    CONTINUOUS_STATS = {'pts', 'reb', 'ast'}

    def __init__(self):
        self._rng = np.random.default_rng()

    def _normalize_stat(self, stat: str) -> str:
        return (stat or 'pts').lower().strip()

    def _is_count_stat(self, stat: str) -> bool:
        return self._normalize_stat(stat) in self.COUNT_STATS

    def _infer_std(
        self,
        mean: float,
        std: Optional[float],
        ci_low: Optional[float],
        ci_high: Optional[float],
        stat: str,
    ) -> float:
        if std is not None and std > 0:
            return float(std)
        if ci_low is not None and ci_high is not None and ci_high > ci_low:
            margin = (ci_high - ci_low) / 2.0
            return float(max(margin / 1.96, 0.01))
        if self._is_count_stat(stat):
            return float(max(math.sqrt(max(mean, 0.25)), 0.75))
        return float(max(abs(mean) * 0.35, 1.5))

    def _extract_stat_samples(
        self,
        stat: str,
        recent_games: Optional[list] = None,
        matchup_history: Optional[list] = None,
    ) -> np.ndarray:
        stat_key = self._normalize_stat(stat)
        recent_values: List[float] = []
        matchup_values: List[float] = []

        for game in recent_games or []:
            try:
                value = float(game.get(stat_key, np.nan))
            except (TypeError, ValueError, AttributeError):
                continue
            if np.isfinite(value):
                recent_values.append(value)

        for game in matchup_history or []:
            try:
                value = float(game.get(stat_key, np.nan))
            except (TypeError, ValueError, AttributeError):
                continue
            if np.isfinite(value):
                matchup_values.append(value)

        samples: List[float] = []
        if recent_values:
            samples.extend(recent_values)
            samples.extend(recent_values)
        if matchup_values:
            samples.extend(matchup_values)

        if not samples:
            return np.array([], dtype=float)
        return np.asarray(samples, dtype=float)

    def _context_minutes(
        self,
        recent_avg: Optional[dict],
        matchup_avg: Optional[dict],
        samples: np.ndarray,
    ) -> float:
        for source in (recent_avg, matchup_avg):
            if source and isinstance(source, dict):
                minutes = source.get('min')
                if minutes is not None:
                    try:
                        minutes = float(minutes)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(minutes) and minutes > 0:
                        return minutes
        if samples.size >= 3:
            return float(np.mean(samples))
        return 0.0

    def _estimate_context_volatility(
        self,
        stat: str,
        play_probability: float,
        sample_count: int,
        samples: np.ndarray,
        recent_avg: Optional[dict] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
    ) -> float:
        multiplier = 1.0
        stat_key = self._normalize_stat(stat)
        count_stat = self._is_count_stat(stat_key)

        if sample_count < 4:
            multiplier += 0.18

        if play_probability < 1.0:
            multiplier += (1.0 - play_probability) * 0.9

        minutes = self._context_minutes(recent_avg, matchup_avg, samples)
        if minutes and minutes < 18:
            multiplier += 0.18 if count_stat else 0.10
        elif minutes and minutes < 24:
            multiplier += 0.08 if count_stat else 0.04

        if trend:
            pct_change = trend.get('pct_change', 0)
            try:
                pct_change = abs(float(pct_change))
            except (TypeError, ValueError):
                pct_change = 0
            if pct_change >= 25:
                multiplier += 0.08

        if samples.size >= 2:
            sample_mean = float(np.mean(samples))
            sample_std = float(np.std(samples))
            if sample_mean > 0:
                cv = sample_std / max(sample_mean, 1e-6)
                if cv > 0.8:
                    multiplier += min((cv - 0.8) * 0.25, 0.25)

        if opponent_defense:
            try:
                rank = int(opponent_defense.get('league_rank', 15))
            except (TypeError, ValueError):
                rank = 15
            if rank <= 5 or rank >= 25:
                multiplier += 0.05

        return float(np.clip(multiplier, 0.85, 1.9))

    def _resolve_distribution_spec(
        self,
        mean: float,
        std: Optional[float],
        line: float,
        stat: str,
        recent_games: Optional[list] = None,
        matchup_history: Optional[list] = None,
        recent_avg: Optional[dict] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
        play_probability: float = 1.0,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
    ) -> DistributionSpec:
        stat_key = self._normalize_stat(stat)
        is_count = self._is_count_stat(stat_key)
        samples = self._extract_stat_samples(stat_key, recent_games, matchup_history)
        sample_count = int(samples.size)

        effective_mean = float(mean)
        effective_std = self._infer_std(effective_mean, std, ci_low, ci_high, stat_key)
        zero_rate = float(np.mean(samples <= 0)) if sample_count else (
            float(np.exp(-max(effective_mean, 0.0))) if is_count and effective_mean > 0 else 0.0
        )
        context_multiplier = self._estimate_context_volatility(
            stat_key,
            play_probability,
            sample_count,
            samples,
            recent_avg=recent_avg,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
        )
        effective_std = float(max(effective_std * context_multiplier, 0.1 if is_count else 0.35))
        variance = float(effective_std ** 2)

        if sample_count >= 5:
            family = 'empirical_bootstrap'
            fallback_used = False
            reason = f"empirical samples={sample_count}"
        else:
            fallback_used = True
            if is_count:
                if effective_mean <= 0:
                    family = 'degenerate_zero'
                elif zero_rate >= 0.35 and effective_mean < 3.5:
                    family = 'zero_inflated_poisson'
                elif variance > (effective_mean + 1e-6) * 1.15:
                    family = 'negative_binomial'
                else:
                    family = 'poisson'
            else:
                family = 'gamma' if effective_mean > 0 else 'normal'
            reason = (
                f"{family} fallback; samples={sample_count}; "
                f"play={play_probability:.2f}; vol={context_multiplier:.2f}"
            )

        return DistributionSpec(
            stat=stat_key,
            family=family,
            is_count=is_count,
            fallback_used=fallback_used,
            reason=reason,
            sample_count=sample_count,
            zero_rate=zero_rate,
            mean=effective_mean,
            std=effective_std,
            variance=variance,
            use_empirical=sample_count >= 5,
            samples=samples if sample_count > 0 else None,
            play_probability=float(np.clip(play_probability, 0.0, 1.0)),
            context_adjustment=context_multiplier,
        )

    def _gamma_params(self, mean: float, std: float) -> Tuple[float, float]:
        mean = max(float(mean), 1e-6)
        std = max(float(std), 1e-6)
        variance = max(std ** 2, 1e-6)
        shape = max((mean ** 2) / variance, 1e-6)
        scale = max(variance / mean, 1e-6)
        return shape, scale

    def _nb_params(self, mean: float, variance: float) -> Tuple[float, float]:
        mean = max(float(mean), 1e-6)
        variance = max(float(variance), mean + 1e-6)
        if variance <= mean:
            return 1e6, 0.999999
        r = max((mean ** 2) / (variance - mean), 1e-6)
        p = max(min(r / (r + mean), 0.999999), 1e-6)
        return r, p

    def _threshold_for_stat(self, line: float, is_count: bool) -> float:
        if is_count:
            return float(math.floor(line))
        return float(line)

    def _mix_play_probability(
        self,
        prob_over: float,
        prob_under: float,
        play_probability: float,
    ) -> Tuple[float, float]:
        play_probability = float(np.clip(play_probability, 0.0, 1.0))
        if play_probability >= 1.0:
            return prob_over, prob_under

        adjusted_over = prob_over * play_probability
        adjusted_under = prob_under * play_probability + (1.0 - play_probability)
        return adjusted_over, adjusted_under

    def _analytic_probability(self, spec: DistributionSpec, line: float) -> Tuple[float, float]:
        threshold = self._threshold_for_stat(line, spec.is_count)

        if spec.family == 'empirical_bootstrap' and spec.samples is not None and spec.samples.size > 0:
            if spec.is_count:
                prob_over = float(np.mean(spec.samples > threshold))
                prob_under = float(np.mean(spec.samples <= threshold))
            else:
                prob_over = float(np.mean(spec.samples > threshold))
                prob_under = float(np.mean(spec.samples < threshold))
            prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
            return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

        if spec.is_count:
            if spec.family == 'degenerate_zero':
                prob_over = 0.0 if threshold >= 0 else 1.0
                prob_over, prob_under = self._mix_play_probability(prob_over, 1.0 - prob_over, spec.play_probability)
                return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

            if spec.family == 'zero_inflated_poisson':
                zero_mass = float(np.clip(spec.zero_rate, 0.0, 0.95))
                lam = max(spec.mean / max(1.0 - zero_mass, 1e-6), 1e-6)
                prob_over = (1.0 - zero_mass) * float(stats.poisson.sf(threshold, lam))
                prob_under = 1.0 - prob_over
                prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
                return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

            if spec.family == 'negative_binomial':
                r, p = self._nb_params(spec.mean, spec.variance)
                prob_over = float(stats.nbinom.sf(threshold, r, p))
                prob_under = 1.0 - prob_over
                prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
                return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

            prob_over = float(stats.poisson.sf(threshold, max(spec.mean, 1e-6)))
            prob_under = 1.0 - prob_over
            prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
            return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

        if spec.family == 'gamma':
            shape, scale = self._gamma_params(spec.mean, spec.std)
            prob_over = float(stats.gamma.sf(threshold, a=shape, scale=scale))
            prob_under = float(stats.gamma.cdf(threshold, a=shape, scale=scale))
            prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
            return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

        if spec.std <= 0:
            prob_over = 1.0 if spec.mean > threshold else 0.0
            prob_under = 1.0 - prob_over
            prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
            return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

        prob_over = float(stats.norm.sf(threshold, loc=spec.mean, scale=spec.std))
        prob_under = float(stats.norm.cdf(threshold, loc=spec.mean, scale=spec.std))
        prob_over, prob_under = self._mix_play_probability(prob_over, prob_under, spec.play_probability)
        return max(0.0, min(1.0, prob_over)), max(0.0, min(1.0, prob_under))

    def _simulate_distribution(self, spec: DistributionSpec, num_sims: int) -> np.ndarray:
        num_sims = max(int(num_sims), 1)

        if spec.family == 'empirical_bootstrap' and spec.samples is not None and spec.samples.size > 0:
            centered = spec.samples - float(np.mean(spec.samples))
            draws = self._rng.choice(centered, size=num_sims, replace=True) + spec.mean
            jitter = self._rng.normal(0.0, max(spec.std * 0.2, 0.05), size=num_sims)
            draws = draws + jitter
        elif spec.family == 'degenerate_zero':
            draws = np.zeros(num_sims, dtype=float)
        elif spec.is_count:
            if spec.family == 'zero_inflated_poisson':
                zero_mass = float(np.clip(spec.zero_rate, 0.0, 0.95))
                lam = max(spec.mean / max(1.0 - zero_mass, 1e-6), 1e-6)
                draws = self._rng.poisson(lam, size=num_sims).astype(float)
                zero_mask = self._rng.random(num_sims) < zero_mass
                draws[zero_mask] = 0.0
            elif spec.family == 'negative_binomial':
                r, p = self._nb_params(spec.mean, spec.variance)
                draws = self._rng.negative_binomial(r, p, size=num_sims).astype(float)
            else:
                draws = self._rng.poisson(max(spec.mean, 1e-6), size=num_sims).astype(float)
            draws = np.clip(np.rint(draws), 0.0, None)
        elif spec.family == 'gamma':
            shape, scale = self._gamma_params(spec.mean, spec.std)
            draws = self._rng.gamma(shape, scale, size=num_sims).astype(float)
        else:
            draws = self._rng.normal(spec.mean, max(spec.std, 1e-6), size=num_sims).astype(float)
            draws = np.clip(draws, 0.0, None)

        if spec.play_probability < 1.0:
            play_mask = self._rng.random(num_sims) < spec.play_probability
            draws = np.where(play_mask, draws, 0.0)

        return draws

    def _evaluate_probability(
        self,
        mean: float,
        std: Optional[float],
        line: float,
        stat: str,
        *,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        play_probability: float = 1.0,
        recent_games: Optional[list] = None,
        matchup_history: Optional[list] = None,
        recent_avg: Optional[dict] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
        num_sims: Optional[int] = None,
        use_sampling: bool = False,
    ) -> ProbabilityComputation:
        spec = self._resolve_distribution_spec(
            mean=mean,
            std=std,
            line=line,
            stat=stat,
            recent_games=recent_games,
            matchup_history=matchup_history,
            recent_avg=recent_avg,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            play_probability=play_probability,
            ci_low=ci_low,
            ci_high=ci_high,
        )

        if use_sampling:
            draws = self._simulate_distribution(spec, num_sims or 5000)
            threshold = self._threshold_for_stat(line, spec.is_count)
            if spec.is_count:
                prob_over = float(np.mean(draws > threshold))
                prob_under = float(np.mean(draws <= threshold))
            else:
                prob_over = float(np.mean(draws > threshold))
                prob_under = float(np.mean(draws < threshold))
            mean_val = float(np.mean(draws))
            std_val = float(np.std(draws))
            ci_low_val = float(np.percentile(draws, 2.5))
            ci_high_val = float(np.percentile(draws, 97.5))
        else:
            prob_over, prob_under = self._analytic_probability(spec, line)
            mean_val = float(spec.mean)
            std_val = float(spec.std)
            ci_low_val = ci_low
            ci_high_val = ci_high

        return ProbabilityComputation(
            prob_over=max(0.0, min(1.0, float(prob_over))),
            prob_under=max(0.0, min(1.0, float(prob_under))),
            mean=mean_val,
            std=std_val,
            ci_low=ci_low_val,
            ci_high=ci_high_val,
            method=spec.family,
            fallback_used=spec.fallback_used,
            sample_count=spec.sample_count,
            reason=spec.reason,
        )

    def calculate_probability(
        self,
        mean: float,
        std: float,
        line: float,
        stat: str = 'pts',
        discrete: bool = True,
        recent_games: Optional[list] = None,
        matchup_history: Optional[list] = None,
        recent_avg: Optional[dict] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
        play_probability: float = 1.0,
    ) -> Tuple[float, float]:
        _ = discrete
        computation = self._evaluate_probability(
            mean,
            std,
            line,
            stat,
            play_probability=play_probability,
            recent_games=recent_games,
            matchup_history=matchup_history,
            recent_avg=recent_avg,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            use_sampling=False,
        )
        return computation.prob_over, computation.prob_under

    def calculate_from_projection(
        self,
        player_name: str,
        stat: str,
        line: float,
        mean: float,
        std: Optional[float] = None,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        opponent: Optional[str] = None,
        date: Optional[str] = None,
        play_probability: float = 1.0,
        num_sims: Optional[int] = None,
        team: Optional[str] = None,
        is_home: bool = False,
        recent_games: Optional[list] = None,
        recent_avg: Optional[dict] = None,
        matchup_history: Optional[list] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
        base_mean: Optional[float] = None,
        adjustments: Optional[List[Tuple[str, Any]]] = None,
    ) -> ProbabilityResult:
        computation = self._evaluate_probability(
            mean,
            std,
            line,
            stat,
            ci_low=ci_low,
            ci_high=ci_high,
            play_probability=play_probability,
            recent_games=recent_games,
            matchup_history=matchup_history,
            recent_avg=recent_avg,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            num_sims=num_sims,
            use_sampling=False,
        )

        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=computation.mean,
            std=computation.std,
            prob_over=computation.prob_over,
            prob_under=computation.prob_under,
            ci_low=computation.ci_low,
            ci_high=computation.ci_high,
            opponent=opponent,
            date=date,
            play_probability=play_probability,
            team=team,
            is_home=is_home,
            recent_games=recent_games,
            recent_avg=recent_avg,
            matchup_history=matchup_history,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            base_mean=base_mean if base_mean is not None else mean,
            adjustments=adjustments,
            num_sims=num_sims,
            probability_method=computation.method,
            fallback_used=computation.fallback_used,
            sample_count=computation.sample_count,
        )

    def run_monte_carlo_simulation(
        self,
        player_name: str,
        stat: str,
        line: float,
        mean: float,
        std: Optional[float] = None,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        opponent: Optional[str] = None,
        date: Optional[str] = None,
        play_probability: float = 1.0,
        num_sims: int = 5000,
        team: Optional[str] = None,
        is_home: bool = False,
        recent_games: Optional[list] = None,
        recent_avg: Optional[dict] = None,
        matchup_history: Optional[list] = None,
        matchup_avg: Optional[dict] = None,
        opponent_defense: Optional[dict] = None,
        trend: Optional[dict] = None,
        base_mean: Optional[float] = None,
        adjustments: Optional[List[Tuple[str, Any]]] = None,
    ) -> ProbabilityResult:
        computation = self._evaluate_probability(
            mean,
            std,
            line,
            stat,
            ci_low=ci_low,
            ci_high=ci_high,
            play_probability=play_probability,
            recent_games=recent_games,
            matchup_history=matchup_history,
            recent_avg=recent_avg,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            num_sims=num_sims,
            use_sampling=True,
        )

        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=computation.mean,
            std=computation.std,
            prob_over=computation.prob_over,
            prob_under=computation.prob_under,
            ci_low=computation.ci_low,
            ci_high=computation.ci_high,
            opponent=opponent,
            date=date,
            play_probability=play_probability,
            team=team,
            is_home=is_home,
            recent_games=recent_games,
            recent_avg=recent_avg,
            matchup_history=matchup_history,
            matchup_avg=matchup_avg,
            opponent_defense=opponent_defense,
            trend=trend,
            base_mean=base_mean if base_mean is not None else mean,
            adjustments=adjustments,
            num_sims=num_sims,
            probability_method=computation.method,
            fallback_used=computation.fallback_used,
            sample_count=computation.sample_count,
        )

    def calculate_from_simulations(
        self,
        player_name: str,
        stat: str,
        line: float,
        simulated_values: np.ndarray,
        played_mask: Optional[np.ndarray] = None,
        opponent: Optional[str] = None,
        date: Optional[str] = None,
        play_probability: float = 1.0,
        num_sims: Optional[int] = None,
    ) -> ProbabilityResult:
        simulated_values = np.asarray(simulated_values, dtype=float)

        if played_mask is not None:
            played_mask = np.asarray(played_mask, dtype=bool)
            valid_values = simulated_values[played_mask]
        else:
            valid_values = simulated_values[np.isfinite(simulated_values)]

        if len(valid_values) < 3:
            raise ValueError("Not enough valid simulations")

        if play_probability < 1.0:
            play_mask = self._rng.random(len(valid_values)) < play_probability
            valid_values = np.where(play_mask, valid_values, 0.0)

        stat_key = self._normalize_stat(stat)
        threshold = self._threshold_for_stat(line, self._is_count_stat(stat_key))

        mean = float(np.mean(valid_values))
        std = float(np.std(valid_values))

        if self._is_count_stat(stat_key):
            prob_over = float(np.mean(valid_values > threshold))
            prob_under = float(np.mean(valid_values <= threshold))
        else:
            prob_over = float(np.mean(valid_values > threshold))
            prob_under = float(np.mean(valid_values < threshold))

        ci_low = float(np.percentile(valid_values, 2.5))
        ci_high = float(np.percentile(valid_values, 97.5))

        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=mean,
            std=std,
            prob_over=max(0.0, min(1.0, prob_over)),
            prob_under=max(0.0, min(1.0, prob_under)),
            ci_low=ci_low,
            ci_high=ci_high,
            opponent=opponent,
            date=date,
            play_probability=play_probability,
            num_sims=num_sims,
            probability_method='simulation_empirical',
            fallback_used=False,
            sample_count=len(valid_values),
        )

    def evaluate_calibration(
        self,
        predicted_probabilities: List[float],
        actual_outcomes: List[int],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        probs = np.asarray(predicted_probabilities, dtype=float)
        outcomes = np.asarray(actual_outcomes, dtype=float)
        if probs.size == 0 or outcomes.size == 0 or probs.size != outcomes.size:
            raise ValueError("Predicted probabilities and actual outcomes must be non-empty and aligned")

        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        outcomes = np.clip(outcomes, 0, 1)

        brier_score = float(np.mean((probs - outcomes) ** 2))
        log_loss = float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        calibration_bins: List[Dict[str, Any]] = []
        for idx in range(n_bins):
            low = bins[idx]
            high = bins[idx + 1]
            if idx == n_bins - 1:
                mask = (probs >= low) & (probs <= high)
            else:
                mask = (probs >= low) & (probs < high)

            bucket_probs = probs[mask]
            bucket_outcomes = outcomes[mask]
            if bucket_probs.size:
                calibration_bins.append(
                    {
                        'bin_low': float(low),
                        'bin_high': float(high),
                        'count': int(bucket_probs.size),
                        'avg_predicted_prob': float(np.mean(bucket_probs)),
                        'actual_hit_rate': float(np.mean(bucket_outcomes)),
                    }
                )
            else:
                calibration_bins.append(
                    {
                        'bin_low': float(low),
                        'bin_high': float(high),
                        'count': 0,
                        'avg_predicted_prob': None,
                        'actual_hit_rate': None,
                    }
                )

        bucket_summary = [
            {
                'bucket': f"{int(low * 100)}-{int(high * 100)}%",
                'count': item['count'],
                'hit_rate': item['actual_hit_rate'],
            }
            for item, low, high in zip(calibration_bins, bins[:-1], bins[1:])
        ]

        return {
            'brier_score': brier_score,
            'log_loss': log_loss,
            'calibration_bins': calibration_bins,
            'confidence_buckets': bucket_summary,
            'sample_size': int(probs.size),
        }

    def format_result(self, result: ProbabilityResult) -> str:
        stat_display = self.STAT_DISPLAY_NAMES.get(result.stat.lower(), result.stat.upper())
        lines = [
            f"{'─' * 50}",
            f"{result.player_name}" + (f" vs {result.opponent}" if result.opponent else ""),
            f"{'─' * 50}",
        ]

        proj_line = f"{stat_display}: {result.mean:.1f} ± {result.std:.1f}"
        if result.ci_low is not None and result.ci_high is not None:
            proj_line += f" (95% CI: {result.ci_low:.1f} - {result.ci_high:.1f})"
        lines.append(proj_line)
        if result.play_probability < 1.0:
            lines.append(f"Play Probability: {result.play_probability * 100:.0f}%")
        if result.probability_method:
            method_label = self._format_distribution_label(result.probability_method)
            fallback_note = " [fallback]" if result.fallback_used else ""
            lines.append(f"Method: {method_label}{fallback_note}")
        lines.append(f"Line: {result.line}")
        lines.append("")
        over_pct = result.prob_over * 100
        under_pct = result.prob_under * 100
        lines.append(f"  OVER  {result.line}: {over_pct:5.1f}%  {self._make_bar(over_pct)}")
        lines.append(f"  UNDER {result.line}: {under_pct:5.1f}%  {self._make_bar(under_pct)}")
        if result.recommendation:
            lines.append(f"  ▸ Recommendation: {result.recommendation}")
        lines.append(f"{'─' * 50}")
        return "\n".join(lines)

    def _make_bar(self, percentage: float, width: int = 20) -> str:
        filled = int(percentage / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def compare_lines(
        self,
        player_name: str,
        stat: str,
        lines: list,
        mean: float,
        std: float,
        **kwargs,
    ) -> str:
        results = [
            self.calculate_from_projection(
                player_name=player_name,
                stat=stat,
                line=line,
                mean=mean,
                std=std,
                **kwargs,
            )
            for line in lines
        ]

        stat_display = self.STAT_DISPLAY_NAMES.get(stat.lower(), stat.upper())
        lines_out = [
            f"{'─' * 60}",
            f"{player_name} - {stat_display} Line Comparison",
            f"Projection: {mean:.1f} ± {std:.1f}",
            f"{'─' * 60}",
            f"{'Line':>8} {'OVER':>10} {'UNDER':>10} {'Recommendation':>20}",
            f"{'─' * 60}",
        ]
        for result in results:
            rec = result.recommendation or "-"
            lines_out.append(
                f"{result.line:>8.1f} {result.prob_over * 100:>9.1f}% {result.prob_under * 100:>9.1f}% {rec:>20}"
            )
        lines_out.append(f"{'─' * 60}")
        return "\n".join(lines_out)

    def format_detailed_result(self, result: ProbabilityResult) -> str:
        stat_display = self.STAT_DISPLAY_NAMES.get(result.stat.lower(), result.stat.upper())
        lines = [
            f"{'═' * 70}",
            f"{result.player_name}" + (f" ({result.team})" if result.team else "") + (
                f" vs {result.opponent}" if result.opponent else ""
            ),
            f"{'═' * 70}",
        ]

        if result.recent_games:
            lines.extend(self._format_recent_performance(result))
        if result.matchup_history or result.opponent_defense:
            lines.extend(self._format_matchup_analysis(result))
        lines.extend(self._format_projection_breakdown(result, stat_display))
        lines.extend(self._format_over_under(result, stat_display))
        lines.append(f"{'═' * 70}")
        return "\n".join(lines)

    def _format_recent_performance(self, result: ProbabilityResult) -> List[str]:
        lines = [
            "",
            "┌─ RECENT PERFORMANCE (Last 5 Games) ──────────────────────────────────┐",
            "│  DATE        MIN    PTS    REB    AST   STL   BLK   TOV   RESULT      │",
        ]
        for game in (result.recent_games or [])[:5]:
            lines.append(
                "│  {date} {minv:>5.1f} {pts:>6.0f} {reb:>6.0f} {ast:>6.0f} {stl:>5.0f} {blk:>5.0f} {tov:>5.0f}   {result:<7}│".format(
                    date=str(game.get('date_short', '')).ljust(10),
                    minv=float(game.get('min', 0) or 0),
                    pts=float(game.get('pts', 0) or 0),
                    reb=float(game.get('reb', 0) or 0),
                    ast=float(game.get('ast', 0) or 0),
                    stl=float(game.get('stl', 0) or 0),
                    blk=float(game.get('blk', 0) or 0),
                    tov=float(game.get('tov', 0) or 0),
                    result=str(game.get('result', '')),
                )
            )

        if result.recent_avg:
            avg = result.recent_avg
            lines.append("├──────────────────────────────────────────────────────────────────────┤")
            lines.append(
                f"│  5-GAME AVG: {avg.get('min', 0):.1f} MIN, {avg.get('pts', 0):.1f} PTS, "
                f"{avg.get('reb', 0):.1f} REB, {avg.get('ast', 0):.1f} AST, "
                f"{avg.get('stl', 0):.1f} STL, {avg.get('blk', 0):.1f} BLK, {avg.get('tov', 0):.1f} TOV{' ':>2}│"
            )
            if result.trend:
                lines.append(f"│  TREND: {result.trend.get('description', ''):<56}│")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        return lines

    def _format_matchup_analysis(self, result: ProbabilityResult) -> List[str]:
        stat_key = self._normalize_stat(result.stat)
        stat_display = self.STAT_DISPLAY_NAMES.get(stat_key, stat_key.upper())
        lines = [
            "",
            "┌─ MATCHUP ANALYSIS ───────────────────────────────────────────────────┐",
        ]

        if result.opponent:
            lines.append(f"│  OPPONENT: {result.opponent:<58}│")
        if result.opponent_defense:
            defense = result.opponent_defense
            rank_desc = self._get_rank_description(int(defense.get('league_rank', 15)))
            lines.append(f"│  Defense vs {stat_display:<54}│")
            if stat_key == 'pts':
                lines.append(f"│    • Points Allowed: {float(defense.get('pts_allowed_per_100', 115.0)):.1f}/100 poss ({rank_desc}){' ':>5}│")
            elif stat_key == 'reb':
                lines.append(f"│    • Rebounds Allowed: {float(defense.get('reb_allowed_per_game', 44.0)):.1f}/game ({rank_desc}){' ':>6}│")
            elif stat_key == 'ast':
                lines.append(f"│    • Assists Allowed: {float(defense.get('ast_allowed_per_game', 26.0)):.1f}/game ({rank_desc}){' ':>8}│")
            else:
                lines.append(f"│    • Volatile count prop: higher variance expected{' ':>18}│")

        if result.matchup_history:
            lines.append(f"│  vs {result.opponent} (Last {len(result.matchup_history)} meetings){' ':>33}│")
            for game in result.matchup_history[:3]:
                lines.append(
                    f"│    • {game.get('date_short', '')}: {float(game.get(stat_key, 0) or 0):.0f} {stat_display} in {float(game.get('min', 0) or 0):.0f} min{' ':>15}│"
                )
            if result.matchup_avg:
                lines.append(
                    f"│    • Average vs {result.opponent}: {float(result.matchup_avg.get(stat_key, 0) or 0):.1f} {stat_display}{' ':>23}│"
                )
        elif result.opponent:
            lines.append(f"│  No recent matchups vs {result.opponent} found.{' ':>30}│")

        lines.append(f"│  Venue: {'HOME' if result.is_home else 'AWAY':<4} (context included in distribution){' ':>21}│")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        return lines

    def _format_projection_breakdown(self, result: ProbabilityResult, stat_display: str) -> List[str]:
        base_mean = result.base_mean if result.base_mean is not None else result.mean
        lines = [
            "",
            "┌─ PROJECTION CALCULATION ─────────────────────────────────────────────┐",
            f"│  Base Projection:      {base_mean:.1f} {stat_display}{' ':>35}│",
        ]

        if result.adjustments:
            for adj_name, adj_val in result.adjustments:
                if isinstance(adj_val, (int, float)):
                    adj_sign = "+" if adj_val >= 0 else ""
                    adj_str = f"{adj_sign}{adj_val:.1f} {stat_display}"
                else:
                    adj_str = str(adj_val)
                lines.append(f"│  {adj_name}: {adj_str:<56}│")
        lines.append("│  ─────────────────────────────────────────────────────────────────── │")
        lines.append(f"│  FINAL PROJECTION:     {result.mean:.1f} ± {result.std:.1f} {stat_display}{' ':>28}│")
        if result.ci_low is not None and result.ci_high is not None:
            lines.append(f"│  95% Confidence: {result.ci_low:.1f} - {result.ci_high:.1f} {stat_display}{' ':>31}│")

        source_label = (
            f"Monte Carlo ({result.num_sims} simulations)" if result.num_sims is not None else "analytic probability model"
        )
        lines.append(f"│  Data Source: {source_label:<45}│")
        if result.probability_method:
            method_label = self._format_distribution_label(result.probability_method)
            fallback_note = "fallback" if result.fallback_used else "selected"
            lines.append(f"│  Distribution: {method_label} ({fallback_note}, samples={result.sample_count}){' ':>11}│")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        return lines

    def _format_over_under(self, result: ProbabilityResult, stat_display: str) -> List[str]:
        lines = [
            "",
            f"┌─ OVER/UNDER: {result.line} {stat_display.upper()} ─────────────────────────────────────────┐",
            "│                                                                      │",
        ]
        if result.probability_method:
            method_label = self._format_distribution_label(result.probability_method)
            lines.append(f"│  Distribution: {method_label:<54}│")
            if result.fallback_used:
                lines.append("│  Note: simplified fallback logic used where historical signal was thin│")
            lines.append("│                                                                      │")

        over_pct = result.prob_over * 100
        under_pct = result.prob_under * 100
        lines.append(f"│  OVER  {result.line:>5}: {over_pct:5.1f}%  {self._make_bar(over_pct)}{' ':>19}│")
        lines.append(f"│  UNDER {result.line:>5}: {under_pct:5.1f}%  {self._make_bar(under_pct)}{' ':>19}│")
        if result.play_probability < 1.0:
            dnp_pct = (1 - result.play_probability) * 100
            lines.append(f"│  (DNP risk: {dnp_pct:.0f}%){' ':>51}│")
        lines.append("│                                                                      │")
        if result.z_score is not None:
            lines.append(
                f"│  Z-SCORE: ({result.line:.1f} - {result.mean:.1f}) / {result.std:.1f} = {result.z_score:.2f}{' ':>23}│"
            )
        if result.recommendation:
            edge_pct = (result.edge * 100) if result.edge else 0
            lines.append(f"│  ▸ RECOMMENDATION: {result.recommendation}{' ':>41}│")
            lines.append(f"│    - Edge: {edge_pct:.1f}% above 50/50{' ':>44}│")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        return lines

    def _format_distribution_label(self, method: str) -> str:
        labels = {
            'empirical_bootstrap': 'Empirical bootstrap',
            'poisson': 'Poisson count model',
            'negative_binomial': 'Negative binomial',
            'zero_inflated_poisson': 'Zero-inflated Poisson',
            'gamma': 'Gamma positive model',
            'degenerate_zero': 'Zero-mass fallback',
            'normal': 'Normal fallback',
            'simulation_empirical': 'Simulation empirical',
        }
        return labels.get(method, method.replace('_', ' ').title())

    def _get_rank_description(self, rank: int) -> str:
        if rank <= 5:
            return f"#{rank} in NBA - Elite"
        if rank <= 10:
            return f"#{rank} in NBA - Strong"
        if rank <= 20:
            return f"#{rank} in NBA - Average"
        return f"#{rank} in NBA - Weak"
