"""
Centralized team mappings for NBA prediction system.

This module provides a single source of truth for all team-related data,
replacing duplicate mappings across multiple scrapers.
"""

from typing import Dict, Optional, List, Any
from pathlib import Path
import json

TEAM_MAPPINGS = {
    'ATL': 'ATL', 'Atlanta Hawks': 'ATL',
    'BOS': 'BOS', 'Boston Celtics': 'BOS',
    'BKN': 'BKN', 'Brooklyn Nets': 'BKN', 'NJN': 'BKN', 'New Jersey Nets': 'BKN',
    'CHA': 'CHA', 'Charlotte Hornets': 'CHA', 'CHH': 'CHA', 'CHO': 'CHA',
    'CHI': 'CHI', 'Chicago Bulls': 'CHI',
    'CLE': 'CLE', 'Cleveland Cavaliers': 'CLE',
    'DAL': 'DAL', 'Dallas Mavericks': 'DAL',
    'DEN': 'DEN', 'Denver Nuggets': 'DEN',
    'DET': 'DET', 'Detroit Pistons': 'DET',
    'GSW': 'GSW', 'Golden State Warriors': 'GSW',
    'HOU': 'HOU', 'Houston Rockets': 'HOU',
    'IND': 'IND', 'Indiana Pacers': 'IND',
    'LAC': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'LAL': 'LAL', 'Los Angeles Lakers': 'LAL',
    'MEM': 'MEM', 'Memphis Grizzlies': 'MEM',
    'MIA': 'MIA', 'Miami Heat': 'MIA',
    'MIL': 'MIL', 'Milwaukee Bucks': 'MIL',
    'MIN': 'MIN', 'Minnesota Timberwolves': 'MIN',
    'NOP': 'NOP', 'New Orleans Pelicans': 'NOP', 'NOH': 'NOP', 'NOK': 'NOP',
    'NYK': 'NYK', 'New York Knicks': 'NYK',
    'OKC': 'OKC', 'Oklahoma City Thunder': 'OKC', 'SEA': 'OKC', 'Seattle SuperSonics': 'OKC',
    'ORL': 'ORL', 'Orlando Magic': 'ORL',
    'PHI': 'PHI', 'Philadelphia 76ers': 'PHI',
    'PHX': 'PHX', 'Phoenix Suns': 'PHX', 'PHO': 'PHX',
    'POR': 'POR', 'Portland Trail Blazers': 'POR',
    'SAC': 'SAC', 'Sacramento Kings': 'SAC',
    'SAS': 'SAS', 'San Antonio Spurs': 'SAS',
    'TOR': 'TOR', 'Toronto Raptors': 'TOR',
    'UTA': 'UTA', 'Utah Jazz': 'UTA',
    'WAS': 'WAS', 'Washington Wizards': 'WAS'
}

ID_TO_ABBR = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612739: 'CLE',
    1610612740: 'NOP', 1610612741: 'CHI', 1610612742: 'DAL',
    1610612743: 'DEN', 1610612744: 'GSW', 1610612745: 'HOU',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612751: 'BKN',
    1610612752: 'NYK', 1610612753: 'ORL', 1610612754: 'IND',
    1610612755: 'PHI', 1610612756: 'PHX', 1610612757: 'POR',
    1610612758: 'SAC', 1610612759: 'SAS', 1610612760: 'OKC',
    1610612761: 'TOR', 1610612762: 'UTA', 1610612763: 'MEM',
    1610612764: 'WAS', 1610612765: 'DET', 1610612766: 'CHA',
}

ABBR_TO_ID = {v: k for k, v in ID_TO_ABBR.items()}

TEAMS = {
    'ATL': {
        'id': 1610612737, 'abbr': 'ATL', 'name': 'Hawks', 'city': 'Atlanta',
        'bref_abbr': 'ATL', 'conference': 'East', 'division': 'Southeast'
    },
    'BOS': {
        'id': 1610612738, 'abbr': 'BOS', 'name': 'Celtics', 'city': 'Boston',
        'bref_abbr': 'BOS', 'conference': 'East', 'division': 'Atlantic'
    },
    'BKN': {
        'id': 1610612751, 'abbr': 'BKN', 'name': 'Nets', 'city': 'Brooklyn',
        'bref_abbr': 'BRK', 'conference': 'East', 'division': 'Atlantic'
    },
    'CHA': {
        'id': 1610612766, 'abbr': 'CHA', 'name': 'Hornets', 'city': 'Charlotte',
        'bref_abbr': 'CHA', 'conference': 'East', 'division': 'Southeast'
    },
    'CHI': {
        'id': 1610612741, 'abbr': 'CHI', 'name': 'Bulls', 'city': 'Chicago',
        'bref_abbr': 'CHI', 'conference': 'East', 'division': 'Central'
    },
    'CLE': {
        'id': 1610612739, 'abbr': 'CLE', 'name': 'Cavaliers', 'city': 'Cleveland',
        'bref_abbr': 'CLE', 'conference': 'East', 'division': 'Central'
    },
    'DAL': {
        'id': 1610612742, 'abbr': 'DAL', 'name': 'Mavericks', 'city': 'Dallas',
        'bref_abbr': 'DAL', 'conference': 'West', 'division': 'Southwest'
    },
    'DEN': {
        'id': 1610612743, 'abbr': 'DEN', 'name': 'Nuggets', 'city': 'Denver',
        'bref_abbr': 'DEN', 'conference': 'West', 'division': 'Northwest'
    },
    'DET': {
        'id': 1610612765, 'abbr': 'DET', 'name': 'Pistons', 'city': 'Detroit',
        'bref_abbr': 'DET', 'conference': 'East', 'division': 'Central'
    },
    'GSW': {
        'id': 1610612744, 'abbr': 'GSW', 'name': 'Warriors', 'city': 'Golden State',
        'bref_abbr': 'GSW', 'conference': 'West', 'division': 'Pacific'
    },
    'HOU': {
        'id': 1610612745, 'abbr': 'HOU', 'name': 'Rockets', 'city': 'Houston',
        'bref_abbr': 'HOU', 'conference': 'West', 'division': 'Southwest'
    },
    'IND': {
        'id': 1610612754, 'abbr': 'IND', 'name': 'Pacers', 'city': 'Indiana',
        'bref_abbr': 'IND', 'conference': 'East', 'division': 'Central'
    },
    'LAC': {
        'id': 1610612746, 'abbr': 'LAC', 'name': 'Clippers', 'city': 'Los Angeles',
        'bref_abbr': 'LAC', 'conference': 'West', 'division': 'Pacific'
    },
    'LAL': {
        'id': 1610612747, 'abbr': 'LAL', 'name': 'Lakers', 'city': 'Los Angeles',
        'bref_abbr': 'LAL', 'conference': 'West', 'division': 'Pacific'
    },
    'MEM': {
        'id': 1610612763, 'abbr': 'MEM', 'name': 'Grizzlies', 'city': 'Memphis',
        'bref_abbr': 'MEM', 'conference': 'West', 'division': 'Southwest'
    },
    'MIA': {
        'id': 1610612748, 'abbr': 'MIA', 'name': 'Heat', 'city': 'Miami',
        'bref_abbr': 'MIA', 'conference': 'East', 'division': 'Southeast'
    },
    'MIL': {
        'id': 1610612749, 'abbr': 'MIL', 'name': 'Bucks', 'city': 'Milwaukee',
        'bref_abbr': 'MIL', 'conference': 'East', 'division': 'Central'
    },
    'MIN': {
        'id': 1610612750, 'abbr': 'MIN', 'name': 'Timberwolves', 'city': 'Minnesota',
        'bref_abbr': 'MIN', 'conference': 'West', 'division': 'Northwest'
    },
    'NOP': {
        'id': 1610612740, 'abbr': 'NOP', 'name': 'Pelicans', 'city': 'New Orleans',
        'bref_abbr': 'NOP', 'conference': 'West', 'division': 'Southwest'
    },
    'NYK': {
        'id': 1610612752, 'abbr': 'NYK', 'name': 'Knicks', 'city': 'New York',
        'bref_abbr': 'NYK', 'conference': 'East', 'division': 'Atlantic'
    },
    'OKC': {
        'id': 1610612760, 'abbr': 'OKC', 'name': 'Thunder', 'city': 'Oklahoma City',
        'bref_abbr': 'OKC', 'conference': 'West', 'division': 'Northwest'
    },
    'ORL': {
        'id': 1610612753, 'abbr': 'ORL', 'name': 'Magic', 'city': 'Orlando',
        'bref_abbr': 'ORL', 'conference': 'East', 'division': 'Southeast'
    },
    'PHI': {
        'id': 1610612755, 'abbr': 'PHI', 'name': '76ers', 'city': 'Philadelphia',
        'bref_abbr': 'PHI', 'conference': 'East', 'division': 'Atlantic'
    },
    'PHX': {
        'id': 1610612756, 'abbr': 'PHX', 'name': 'Suns', 'city': 'Phoenix',
        'bref_abbr': 'PHO', 'conference': 'West', 'division': 'Pacific'
    },
    'POR': {
        'id': 1610612757, 'abbr': 'POR', 'name': 'Trail Blazers', 'city': 'Portland',
        'bref_abbr': 'POR', 'conference': 'West', 'division': 'Northwest'
    },
    'SAC': {
        'id': 1610612758, 'abbr': 'SAC', 'name': 'Kings', 'city': 'Sacramento',
        'bref_abbr': 'SAC', 'conference': 'West', 'division': 'Pacific'
    },
    'SAS': {
        'id': 1610612759, 'abbr': 'SAS', 'name': 'Spurs', 'city': 'San Antonio',
        'bref_abbr': 'SAS', 'conference': 'West', 'division': 'Southwest'
    },
    'TOR': {
        'id': 1610612761, 'abbr': 'TOR', 'name': 'Raptors', 'city': 'Toronto',
        'bref_abbr': 'TOR', 'conference': 'East', 'division': 'Atlantic'
    },
    'UTA': {
        'id': 1610612762, 'abbr': 'UTA', 'name': 'Jazz', 'city': 'Utah',
        'bref_abbr': 'UTA', 'conference': 'West', 'division': 'Northwest'
    },
    'WAS': {
        'id': 1610612764, 'abbr': 'WAS', 'name': 'Wizards', 'city': 'Washington',
        'bref_abbr': 'WAS', 'conference': 'East', 'division': 'Southeast'
    },
}

POSITION_MAP = {
    'PG': 'Guard', 'SG': 'Guard', 'G': 'Guard',
    'SF': 'Forward', 'PF': 'Forward', 'F': 'Forward',
    'C': 'Center', 'FC': 'Center', 'CF': 'Center',
    'G/F': 'Wing', 'F/G': 'Wing',
}

def normalize_team(team_str: str) -> str:
    """Normalizes any team string (abbreviation, full name, or ID) to a canonical 3-letter code."""
    if not team_str:
        return 'UNK'
    
    if isinstance(team_str, int):
        return ID_TO_ABBR.get(team_str, 'UNK')
    
    clean_str = str(team_str).strip()
    
    try:
        team_id = int(clean_str)
        return ID_TO_ABBR.get(team_id, 'UNK')
    except ValueError:
        pass
    
    if clean_str in TEAM_MAPPINGS:
        return TEAM_MAPPINGS[clean_str]
    
    for key, val in TEAM_MAPPINGS.items():
        if clean_str.lower() == key.lower():
            return val
            
    return clean_str.upper()[:3]


def get_team_by_abbr(abbr: str) -> Optional[Dict[str, Any]]:
    """Get full team info by abbreviation."""
    return TEAMS.get(abbr.upper())


def get_team_by_id(team_id: int) -> Optional[Dict[str, Any]]:
    """Get full team info by NBA team ID."""
    abbr = ID_TO_ABBR.get(team_id)
    if abbr:
        return TEAMS.get(abbr)
    return None


def get_all_abbrs() -> List[str]:
    """Get list of all current NBA team abbreviations."""
    return list(TEAMS.keys())


def get_team_id(abbr: str) -> Optional[int]:
    """Get NBA team ID from abbreviation."""
    return ABBR_TO_ID.get(abbr.upper())


def get_bref_abbr(abbr: str) -> str:
    """Get Basketball Reference abbreviation for a team."""
    team = TEAMS.get(abbr.upper())
    return team['bref_abbr'] if team else abbr.upper()


def get_conference(abbr: str) -> Optional[str]:
    """Get conference (East/West) for a team."""
    team = TEAMS.get(abbr.upper())
    return team['conference'] if team else None


def get_division(abbr: str) -> Optional[str]:
    """Get division for a team."""
    team = TEAMS.get(abbr.upper())
    return team['division'] if team else None


def load_team_mappings_from_file(filepath: Path) -> Dict:
    """Load additional team mappings from JSON file."""
    if not Path(filepath).exists():
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def save_team_mappings_to_file(mappings: Dict, filepath: Path) -> None:
    """Save team mappings to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(mappings, f, indent=2)
