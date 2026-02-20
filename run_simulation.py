import subprocess
import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run NBA Player Projections Simulation')
    parser.add_argument('--mode', choices=['today', 'date', 'week'], default='date',
                        help='Simulation mode: today, date, or week')
    parser.add_argument('--date', type=str, default='2026-02-19',
                        help='Date to simulate (YYYY-MM-DD)')
    parser.add_argument('--sims', type=int, default=100,
                        help='Number of simulations per matchup')
    args = parser.parse_args()

    root = Path(__file__).parent
    data_dir = root / "data"
    models_dir = root / "models"

    cmd = [
        sys.executable, str(root / "simulate_season.py"),
        "--data-dir", str(data_dir),
        "--models-dir", str(models_dir),
        "--sims", str(args.sims)
    ]

    if args.mode == 'today':
        cmd.append("--today")
    elif args.mode == 'date' and args.date:
        cmd.extend(["--date", args.date])
    elif args.mode == 'week':
        cmd.append("--week")

    print(f"Running: {' '.join(cmd)}\n")
    print("=" * 60)
    print("GENERATING PLAYER PROJECTIONS")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=root)
    
    print("\n" + "=" * 60)
    print("PROJECTIONS COMPLETE")
    print("=" * 60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())