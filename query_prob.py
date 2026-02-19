import argparse
import sys
import logging

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        import io
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='NBA Player Stat Over/Under Probability Query Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python query_prob.py
    
  One-shot query:
    python query_prob.py --player "LeBron James" --stat pts --line 25.5
    python query_prob.py -p "Jokic" -s reb -l 12.5 --opponent DEN
    
  With custom simulations:
    python query_prob.py -p "Curry" -s pts -l 30.5 --sims 500
    
  List available data:
    python query_prob.py --list-players
    python query_prob.py --list-teams
        """
    )
    
    parser.add_argument(
        '-p', '--player',
        type=str,
        help='Player name to query'
    )
    
    parser.add_argument(
        '-s', '--stat',
        type=str,
        choices=['pts', 'reb', 'ast', 'stl', 'blk', 'tov'],
        help='Stat type (pts, reb, ast, stl, blk, tov)'
    )
    
    parser.add_argument(
        '-l', '--line',
        type=float,
        help='Over/under line'
    )
    
    parser.add_argument(
        '-o', '--opponent',
        type=str,
        help='Opponent team abbreviation'
    )
    
    parser.add_argument(
        '-t', '--team',
        type=str,
        help='Player team abbreviation'
    )
    
    parser.add_argument(
        '-d', '--date',
        type=str,
        help='Game date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--over',
        action='store_true',
        help='Show probability of going OVER the line'
    )
    
    parser.add_argument(
        '--under',
        action='store_true',
        help='Show probability of going UNDER the line'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple players (provide 2+ player names)'
    )
    
    parser.add_argument(
        '--list-players',
        action='store_true',
        help='List all available players'
    )
    
    parser.add_argument(
        '--list-teams',
        action='store_true',
        help='List all available teams'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Force reload projection data'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/sim_results',
        help='Directory containing projection data'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--sims',
        type=int,
        default=100,
        help='Number of simulations for live projections (default: 100)'
    )
    
    args = parser.parse_args()
    
    from src.query.interactive_cli import InteractiveCLI
    from src.query.probability_calculator import ProbabilityCalculator
    from src.query.projection_loader import ProjectionLoader
    
    cli = InteractiveCLI(data_dir=args.data_dir, num_sims=args.sims)
    
    if args.list_players:
        cli._list_players()
        return
    
    if args.list_teams:
        cli._list_teams()
        return
    
    if args.reload:
        cli._reload()
        return
    
    if args.compare and len(args.compare) >= 2:
        from src.query.query_parser import ParsedQuery, QueryType
        parsed = ParsedQuery(
            query_type=QueryType.COMPARE,
            compare_players=args.compare,
            stat=args.stat or 'pts'
        )
        cli._handle_compare(parsed)
        return
    
    if args.player:
        if not args.stat:
            args.stat = 'pts'
        
        loader = ProjectionLoader(data_dir=args.data_dir)
        projection = loader.find_player(
            player_name=args.player,
            team=args.team,
            opponent=args.opponent,
            date=args.date
        )
        
        if projection is None:
            print(f"Player '{args.player}' not found in projections.")
            if not args.json:
                print("\nRun in interactive mode to perform live simulations:")
                print("  python query_prob.py")
            else:
                print('{"error": "Player not found"}')
            return
        
        if args.line is None:
            mean_val = projection.get_stat_mean(args.stat)
            args.line = round(mean_val - 0.5) + 0.5
        
        context = loader.get_player_context(
            player_name=projection.player_name,
            opponent=projection.opponent,
            stat=args.stat
        )
        
        calculator = ProbabilityCalculator()
        result = calculator.calculate_from_projection(
            player_name=projection.player_name,
            stat=args.stat,
            line=args.line,
            mean=projection.get_stat_mean(args.stat),
            std=projection.get_stat_std(args.stat),
            ci_low=projection.get_stat_ci(args.stat)[0],
            ci_high=projection.get_stat_ci(args.stat)[1],
            opponent=projection.opponent,
            date=projection.date,
            play_probability=projection.play_probability,
            num_sims=args.sims
        )
        
        result.team = projection.team
        result.is_home = projection.is_home
        result.recent_games = context.get('recent_games')
        result.recent_avg = context.get('recent_avg')
        result.matchup_history = context.get('matchup_history')
        result.matchup_avg = context.get('matchup_avg')
        result.opponent_defense = context.get('opponent_defense')
        result.trend = context.get('trend')
        
        if args.json:
            import json
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(calculator.format_detailed_result(result))
        
        return
    
    cli.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
