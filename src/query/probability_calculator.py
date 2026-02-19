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
            'num_sims': self.num_sims
        }


class ProbabilityCalculator:
    STAT_STEP_SIZES = {
        'pts': 1,
        'reb': 1,
        'ast': 1,
        'stl': 1,
        'blk': 1,
        'tov': 1
    }
    
    STAT_DISPLAY_NAMES = {
        'pts': 'Points',
        'reb': 'Rebounds',
        'ast': 'Assists',
        'stl': 'Steals',
        'blk': 'Blocks',
        'tov': 'Turnovers'
    }
    
    def __init__(self):
        pass
    
    def calculate_probability(
        self,
        mean: float,
        std: float,
        line: float,
        stat: str = 'pts',
        discrete: bool = True
    ) -> Tuple[float, float]:
        if std <= 0:
            std = max(mean * 0.3, 1.0)
        
        prob_over = 1 - stats.norm.cdf(line, loc=mean, scale=std)
        prob_under = stats.norm.cdf(line, loc=mean, scale=std)
        
        prob_over = max(0.0, min(1.0, float(prob_over)))
        prob_under = max(0.0, min(1.0, float(prob_under)))
        
        return prob_over, prob_under
    
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
        num_sims: Optional[int] = None
    ) -> ProbabilityResult:
        if std is None or std <= 0:
            if ci_low is not None and ci_high is not None:
                margin = (ci_high - ci_low) / 2
                std = margin / 1.96
            else:
                std = max(mean * 0.35, 1.5)
        
        prob_over, prob_under = self.calculate_probability(mean, std, line, stat)
        
        if play_probability < 1.0:
            adjusted_prob_over = prob_over * play_probability
            adjusted_prob_under = prob_under * play_probability + (1 - play_probability)
            prob_over = adjusted_prob_over
            prob_under = adjusted_prob_under
        
        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=mean,
            std=std,
            prob_over=float(prob_over),
            prob_under=float(prob_under),
            ci_low=ci_low,
            ci_high=ci_high,
            opponent=opponent,
            date=date,
            play_probability=play_probability,
            num_sims=num_sims
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
        adjustments: Optional[List[Tuple[str, Any]]] = None
    ) -> ProbabilityResult:
        if std is None or std <= 0:
            if ci_low is not None and ci_high is not None:
                margin = (ci_high - ci_low) / 2
                std = margin / 1.96
            else:
                std = max(mean * 0.35, 1.5)
        
        rng = np.random.default_rng()
        simulated_values = rng.normal(mean, std, num_sims)
        
        sim_mean = float(np.mean(simulated_values))
        sim_std = float(np.std(simulated_values))
        
        prob_over_raw = float(np.mean(simulated_values > line))
        prob_under_raw = float(np.mean(simulated_values < line))
        
        if play_probability < 1.0:
            prob_over = prob_over_raw * play_probability
            prob_under = prob_under_raw * play_probability + (1 - play_probability)
        else:
            prob_over = prob_over_raw
            prob_under = prob_under_raw
        
        sim_ci_low = float(np.percentile(simulated_values, 2.5))
        sim_ci_high = float(np.percentile(simulated_values, 97.5))
        
        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=sim_mean,
            std=sim_std,
            prob_over=prob_over,
            prob_under=prob_under,
            ci_low=sim_ci_low,
            ci_high=sim_ci_high,
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
            num_sims=num_sims
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
        num_sims: Optional[int] = None
    ) -> ProbabilityResult:
        simulated_values = np.array(simulated_values)
        
        if played_mask is not None:
            played_mask = np.array(played_mask)
            valid_values = simulated_values[played_mask]
        else:
            valid_values = simulated_values[np.isfinite(simulated_values)]
        
        if len(valid_values) < 3:
            raise ValueError("Not enough valid simulations")
        
        mean = float(np.mean(valid_values))
        std = float(np.std(valid_values))
        
        prob_over_raw = np.mean(valid_values > line)
        prob_under_raw = np.mean(valid_values < line)
        
        if play_probability < 1.0:
            prob_over = prob_over_raw * play_probability
            prob_under = prob_under_raw * play_probability + (1 - play_probability)
        else:
            prob_over = prob_over_raw
            prob_under = prob_under_raw
        
        ci_low = float(np.percentile(valid_values, 2.5))
        ci_high = float(np.percentile(valid_values, 97.5))
        
        return ProbabilityResult(
            player_name=player_name,
            stat=stat,
            line=line,
            mean=mean,
            std=std,
            prob_over=float(prob_over),
            prob_under=float(prob_under),
            ci_low=ci_low,
            ci_high=ci_high,
            opponent=opponent,
            date=date,
            play_probability=play_probability,
            num_sims=num_sims
        )
    
    def format_result(self, result: ProbabilityResult) -> str:
        lines = []
        stat_display = self.STAT_DISPLAY_NAMES.get(result.stat, result.stat.upper())
        
        header_parts = [f"{result.player_name}"]
        if result.opponent:
            header_parts.append(f"vs {result.opponent}")
        if result.date:
            header_parts.append(f"({result.date})")
        
        lines.append(f"{'─' * 50}")
        lines.append(f"{' '.join(header_parts)}")
        lines.append(f"{'─' * 50}")
        
        proj_line = f"{stat_display}: {result.mean:.1f} ± {result.std:.1f}"
        if result.ci_low is not None and result.ci_high is not None:
            proj_line += f" (95% CI: {result.ci_low:.1f} - {result.ci_high:.1f})"
        lines.append(proj_line)
        
        if result.play_probability < 1.0:
            lines.append(f"Play Probability: {result.play_probability * 100:.0f}%")
        
        lines.append(f"Line: {result.line}")
        lines.append("")
        
        over_pct = result.prob_over * 100
        under_pct = result.prob_under * 100
        
        over_bar = self._make_bar(over_pct)
        under_bar = self._make_bar(under_pct)
        
        lines.append(f"  OVER  {result.line}: {over_pct:5.1f}%  {over_bar}")
        lines.append(f"  UNDER {result.line}: {under_pct:5.1f}%  {under_bar}")
        lines.append("")
        
        if result.recommendation:
            lines.append(f"  ▸ Recommendation: {result.recommendation}")
        
        lines.append(f"{'─' * 50}")
        
        return "\n".join(lines)
    
    def _make_bar(self, percentage: float, width: int = 20) -> str:
        filled = int(percentage / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar
    
    def compare_lines(
        self,
        player_name: str,
        stat: str,
        lines: list,
        mean: float,
        std: float,
        **kwargs
    ) -> str:
        results = []
        for line in lines:
            result = self.calculate_from_projection(
                player_name=player_name,
                stat=stat,
                line=line,
                mean=mean,
                std=std,
                **kwargs
            )
            results.append(result)
        
        lines_out = []
        stat_display = self.STAT_DISPLAY_NAMES.get(stat, stat.upper())
        lines_out.append(f"{'─' * 60}")
        lines_out.append(f"{player_name} - {stat_display} Line Comparison")
        lines_out.append(f"Projection: {mean:.1f} ± {std:.1f}")
        lines_out.append(f"{'─' * 60}")
        lines_out.append(f"{'Line':>8} {'OVER':>10} {'UNDER':>10} {'Recommendation':>20}")
        lines_out.append(f"{'─' * 60}")
        
        for r in results:
            rec = r.recommendation or "-"
            lines_out.append(
                f"{r.line:>8.1f} {r.prob_over * 100:>9.1f}% {r.prob_under * 100:>9.1f}% {rec:>20}"
            )
        
        lines_out.append(f"{'─' * 60}")
        return "\n".join(lines_out)
    
    def format_detailed_result(self, result: ProbabilityResult) -> str:
        lines = []
        stat_display = self.STAT_DISPLAY_NAMES.get(result.stat, result.stat.upper())
        stat_key = result.stat.lower()
        
        header = f"{result.player_name}"
        if result.team:
            header += f" ({result.team})"
        if result.opponent:
            header += f" vs {result.opponent}"
        if result.date:
            header += f" — {result.date}"
        
        lines.append(f"{'═' * 70}")
        lines.append(header)
        lines.append(f"{'═' * 70}")
        
        if result.recent_games:
            lines.extend(self._format_recent_performance(result))
        
        if result.matchup_history or result.opponent_defense:
            lines.extend(self._format_matchup_analysis(result))
        
        lines.extend(self._format_projection_breakdown(result, stat_display))
        
        lines.extend(self._format_over_under(result, stat_display))
        
        lines.append(f"{'═' * 70}")
        
        return "\n".join(lines)
    
    def _format_recent_performance(self, result: ProbabilityResult) -> List[str]:
        lines = []
        stat_key = result.stat.lower()
        
        lines.append("")
        lines.append("┌─ RECENT PERFORMANCE (Last 5 Games) ──────────────────────────────────┐")
        lines.append("│  DATE        MIN    PTS    REB    AST   FG%     RESULT              │")
        
        for game in result.recent_games[:5]:
            date_str = game.get('date_short', '').ljust(8)
            min_str = f"{game.get('min', 0):.1f}".rjust(5)
            pts_str = f"{game.get('pts', 0):.0f}".rjust(5)
            reb_str = f"{game.get('reb', 0):.0f}".rjust(5)
            ast_str = f"{game.get('ast', 0):.0f}".rjust(5)
            fg_str = f"{game.get('fg_pct', 0):.0f}%".rjust(4)
            result_str = game.get('result', '')
            matchup = game.get('matchup', '')[:12].ljust(12)
            result_display = f"{result_str} {matchup}"
            
            lines.append(f"│  {date_str} {min_str}  {pts_str}   {reb_str}   {ast_str}  {fg_str}    {result_display:19}│")
        
        if result.recent_avg:
            avg = result.recent_avg
            lines.append("├──────────────────────────────────────────────────────────────────────┤")
            lines.append(f"│  5-GAME AVG: {avg.get('min', 0):.1f} MIN, {avg.get('pts', 0):.1f} PTS, "
                        f"{avg.get('reb', 0):.1f} REB, {avg.get('ast', 0):.1f} AST{' ':>14}│")
            
            if result.trend:
                trend_desc = result.trend.get('description', '')
                lines.append(f"│  TREND: {trend_desc:55}│")
        
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        
        return lines
    
    def _format_matchup_analysis(self, result: ProbabilityResult) -> List[str]:
        lines = []
        stat_key = result.stat.lower()
        stat_display = self.STAT_DISPLAY_NAMES.get(result.stat, result.stat.upper())
        
        lines.append("")
        lines.append("┌─ MATCHUP ANALYSIS ───────────────────────────────────────────────────┐")
        
        if result.opponent:
            lines.append(f"│  OPPONENT: {result.opponent:58}│")
            lines.append("│                                                                      │")
        
        if result.opponent_defense:
            defense = result.opponent_defense
            pts_allowed = defense.get('pts_allowed_per_100', 115.0)
            reb_allowed = defense.get('reb_allowed_per_game', 44.0)
            rank = defense.get('league_rank', 15)
            
            rank_desc = self._get_rank_description(rank)
            
            lines.append(f"│  Defense vs {stat_display}:                                                │")
            
            if stat_key == 'pts':
                lines.append(f"│    • Points Allowed: {pts_allowed:.1f}/100 poss ({rank_desc}){' ':>19}│")
            elif stat_key == 'reb':
                lines.append(f"│    • Rebounds Allowed: {reb_allowed:.1f}/game ({rank_desc}){' ':>18}│")
            else:
                ast_allowed = defense.get('ast_allowed_per_game', 26.0)
                lines.append(f"│    • Assists Allowed: {ast_allowed:.1f}/game{' ':>32}│")
            
            lines.append("│                                                                      │")
        
        if result.matchup_history:
            lines.append(f"│  vs {result.opponent} (Last {len(result.matchup_history)} meetings):{' ':>40}│")
            
            for game in result.matchup_history[:3]:
                date_str = game.get('date_short', '')
                stat_val = game.get(stat_key, 0)
                mins = game.get('min', 0)
                lines.append(f"│    • {date_str}: {stat_val:.0f} {stat_display} in {mins:.0f} min{' ':>28}│")
            
            if result.matchup_avg:
                avg_stat = result.matchup_avg.get(stat_key, 0)
                lines.append(f"│    • Average vs {result.opponent}: {avg_stat:.1f} {stat_display}{' ':>28}│")
        else:
            lines.append(f"│  No recent matchups vs {result.opponent} found.{' ':>33}│")
        
        lines.append("│                                                                      │")
        home_away = "HOME" if result.is_home else "AWAY"
        home_factor = "+0.5" if result.is_home else "-0.3"
        lines.append(f"│  Venue: {home_away} (expected adjustment: {home_factor} {stat_display}){' ':>22}│")
        
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        
        return lines
    
    def _format_projection_breakdown(self, result: ProbabilityResult, stat_display: str) -> List[str]:
        lines = []
        stat_key = result.stat.lower()
        
        lines.append("")
        lines.append("┌─ PROJECTION CALCULATION ─────────────────────────────────────────────┐")
        
        base_mean = result.base_mean if result.base_mean is not None else result.mean
        lines.append(f"│  Base Projection:      {base_mean:.1f} {stat_display} (from Monte Carlo sim){' ':>17}│")
        
        if result.adjustments:
            for adj_name, adj_val in result.adjustments:
                if isinstance(adj_val, (int, float)):
                    adj_sign = "+" if adj_val >= 0 else ""
                    adj_str = f"{adj_sign}{adj_val:.1f} {stat_display}"
                else:
                    adj_str = str(adj_val)
                padding_len = max(0, 68 - (15 + len(adj_name) + len(adj_str)))
                lines.append(f"│  {adj_name}:           {adj_str}{' ' * padding_len}│")
        else:
            if result.is_home:
                lines.append(f"│  Home Court:           +0.5 {stat_display}{' ':>34}│")
            else:
                lines.append(f"│  Away Game:            -0.3 {stat_display}{' ':>34}│")
            
            if result.trend and result.trend.get('direction') == 'up':
                lines.append(f"│  Recent Form:           +0.3 {stat_display}{' ':>34}│")
            elif result.trend and result.trend.get('direction') == 'down':
                lines.append(f"│  Recent Form:           -0.3 {stat_display}{' ':>34}│")
        
        lines.append("│  ─────────────────────────────────────────────────────────────────── │")
        lines.append(f"│  FINAL PROJECTION:     {result.mean:.1f} ± {result.std:.1f} {stat_display}{' ':>28}│")
        lines.append("│                                                                      │")
        
        if result.ci_low is not None and result.ci_high is not None:
            lines.append(f"│  95% Confidence: {result.ci_low:.1f} - {result.ci_high:.1f} {stat_display}{' ':>31}│")
        
        num_sims_display = result.num_sims if result.num_sims is not None else 100
        lines.append(f"│  Data Source: Monte Carlo ({num_sims_display} simulations){' ':>26}│")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        
        return lines
    
    def _format_over_under(self, result: ProbabilityResult, stat_display: str) -> List[str]:
        lines = []
        
        lines.append("")
        lines.append(f"┌─ OVER/UNDER: {result.line} {stat_display.upper()} ─────────────────────────────────────────┐")
        lines.append("│                                                                      │")
        
        if result.z_score is not None:
            lines.append(f"│  Distribution: Normal(μ={result.mean:.1f}, σ={result.std:.1f}){' ':>30}│")
            lines.append("│                                                                      │")
        
        over_pct = result.prob_over * 100
        under_pct = result.prob_under * 100
        
        over_bar = self._make_bar(over_pct)
        under_bar = self._make_bar(under_pct)
        
        lines.append(f"│  OVER  {result.line:>5}: {over_pct:5.1f}%  {over_bar}{' ':>19}│")
        lines.append(f"│  UNDER {result.line:>5}: {under_pct:5.1f}%  {under_bar}{' ':>19}│")
        
        if result.play_probability < 1.0:
            dnp_pct = (1 - result.play_probability) * 100
            lines.append(f"│  (DNP risk: {dnp_pct:.0f}%){' ':>51}│")
        
        lines.append("│                                                                      │")
        
        if result.z_score is not None:
            lines.append(f"│  Z-SCORE: ({result.line:.1f} - {result.mean:.1f}) / {result.std:.1f} = {result.z_score:.2f}{' ':>23}│")
            
            if result.z_score < 0:
                p_calc = (1 - stats.norm.cdf(result.z_score)) * 100
                lines.append(f"│  P(X > {result.line}) = 1 - Φ({result.z_score:.2f}) = {p_calc:.1f}%{' ':>29}│")
            else:
                p_calc = stats.norm.cdf(result.z_score) * 100
                lines.append(f"│  P(X < {result.line}) = Φ({result.z_score:.2f}) = {p_calc:.1f}%{' ':>31}│")
        
        lines.append("│                                                                      │")
        
        if result.recommendation:
            rec = result.recommendation
            edge_pct = (result.edge * 100) if result.edge else 0
            lines.append(f"│  ▸ RECOMMENDATION: {rec}{' ':>41}│")
            lines.append(f"│    - Edge: {edge_pct:.1f}% above 50/50{' ':>44}│")
            
            implied_line = result.mean
            if result.mean > result.line:
                lines.append(f"│    - Fair line estimate: {implied_line:.1f} (you're getting value){' ':>18}│")
            else:
                lines.append(f"│    - Fair line estimate: {implied_line:.1f} (line may be high){' ':>21}│")
        
        lines.append("└──────────────────────────────────────────────────────────────────────┘")
        
        return lines
    
    def _get_rank_description(self, rank: int) -> str:
        if rank <= 5:
            return f"#{rank} in NBA - Elite"
        elif rank <= 10:
            return f"#{rank} in NBA - Strong"
        elif rank <= 20:
            return f"#{rank} in NBA - Average"
        else:
            return f"#{rank} in NBA - Weak"
