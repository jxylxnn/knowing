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
    1610612746: 'IND', 1610612747: 'LAC', 1610612748: 'LAL',
    1610612749: 'MEM', 1610612750: 'MIA', 1610612751: 'MIL',
    1610612752: 'MIN', 1610612753: 'BKN', 1610612754: 'NYK',
    1610612755: 'ORL', 1610612756: 'PHI', 1610612757: 'PHX',
    1610612758: 'POR', 1610612759: 'SAC', 1610612760: 'SAS',
    1610612761: 'OKC', 1610612762: 'TOR', 1610612763: 'UTA',
    1610612764: 'WAS', 1610612765: 'DET', 1610612766: 'CHA',
}

def normalize_team(team_str) -> str:
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
