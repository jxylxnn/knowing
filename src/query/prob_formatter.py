"""Formatting mixin for ProbabilityCalculator output."""

from typing import List, Optional

import numpy as np


class ProbFormatterMixin:
    """Mixin providing formatted output methods for probability results.

    Requires the host class to define ``STAT_DISPLAY_NAMES`` (class-level
    dict) and ``_normalize_stat`` (method).
    """

    STAT_DISPLAY_NAMES: dict

    def _normalize_stat(self, stat: str) -> str:
        raise NotImplementedError

    def format_result(self, result) -> str:
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

    def format_detailed_result(self, result) -> str:
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

    def _format_recent_performance(self, result) -> List[str]:
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

    def _format_matchup_analysis(self, result) -> List[str]:
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

    def _format_projection_breakdown(self, result, stat_display: str) -> List[str]:
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

    def _format_over_under(self, result, stat_display: str) -> List[str]:
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

    def _make_bar(self, percentage: float, width: int = 20) -> str:
        filled = int(percentage / 100 * width)
        return "█" * filled + "░" * (width - filled)
