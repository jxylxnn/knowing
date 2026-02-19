import re
from dataclasses import dataclass, field
from typing import Optional, List, Set
from enum import Enum


class QueryType(Enum):
    OVER_UNDER = "over_under"
    PROJECTION = "projection"
    COMPARE = "compare"
    LIST_PLAYERS = "list_players"
    HELP = "help"
    EXIT = "exit"
    UNKNOWN = "unknown"


STAT_ALIASES = {
    'pts': 'pts',
    'points': 'pts',
    'point': 'pts',
    'reb': 'reb',
    'rebounds': 'reb',
    'rebound': 'reb',
    'ast': 'ast',
    'assists': 'ast',
    'assist': 'ast',
    'stl': 'stl',
    'steals': 'stl',
    'steal': 'stl',
    'blk': 'blk',
    'blocks': 'blk',
    'block': 'blk',
    'tov': 'tov',
    'turnovers': 'tov',
    'turnover': 'tov',
    'to': 'tov'
}

DIRECTION_ALIASES = {
    'over': 'over',
    'o': 'over',
    'above': 'over',
    'under': 'under',
    'u': 'under',
    'below': 'under'
}

TEAM_ABBREVIATIONS = {
    'atlanta': 'ATL', 'hawks': 'ATL',
    'boston': 'BOS', 'celtics': 'BOS',
    'brooklyn': 'BKN', 'nets': 'BKN',
    'charlotte': 'CHA', 'hornets': 'CHA',
    'chicago': 'CHI', 'bulls': 'CHI',
    'cleveland': 'CLE', 'cavaliers': 'CLE', 'cavs': 'CLE',
    'dallas': 'DAL', 'mavericks': 'DAL', 'mavs': 'DAL',
    'denver': 'DEN', 'nuggets': 'DEN',
    'detroit': 'DET', 'pistons': 'DET',
    'golden state': 'GSW', 'warriors': 'GSW', 'gsw': 'GSW',
    'houston': 'HOU', 'rockets': 'HOU',
    'indiana': 'IND', 'pacers': 'IND',
    'la clippers': 'LAC', 'clippers': 'LAC', 'lac': 'LAC',
    'la lakers': 'LAL', 'lakers': 'LAL', 'lal': 'LAL',
    'memphis': 'MEM', 'grizzlies': 'MEM',
    'miami': 'MIA', 'heat': 'MIA',
    'milwaukee': 'MIL', 'bucks': 'MIL',
    'minnesota': 'MIN', 'timberwolves': 'MIN', 'wolves': 'MIN',
    'new orleans': 'NOP', 'pelicans': 'NOP', 'nop': 'NOP', 'no': 'NOP',
    'new york': 'NYK', 'knicks': 'NYK', 'nyk': 'NYK',
    'oklahoma': 'OKC', 'thunder': 'OKC', 'okc': 'OKC',
    'orlando': 'ORL', 'magic': 'ORL',
    'philadelphia': 'PHI', 'sixers': 'PHI', '76ers': 'PHI', 'phi': 'PHI',
    'phoenix': 'PHX', 'suns': 'PHX', 'phx': 'PHX',
    'portland': 'POR', 'blazers': 'POR', 'trail blazers': 'POR',
    'sacramento': 'SAC', 'kings': 'SAC',
    'san antonio': 'SAS', 'spurs': 'SAS', 'sas': 'SAS',
    'toronto': 'TOR', 'raptors': 'TOR',
    'utah': 'UTA', 'jazz': 'UTA',
    'washington': 'WAS', 'wizards': 'WAS'
}


@dataclass
class ParsedQuery:
    query_type: QueryType
    player_name: Optional[str] = None
    stat: Optional[str] = None
    line: Optional[float] = None
    direction: Optional[str] = None
    opponent: Optional[str] = None
    team: Optional[str] = None
    date: Optional[str] = None
    compare_players: Optional[List[str]] = None
    raw_query: str = ""
    confidence: float = 1.0


class QueryParser:
    NUMBER_PATTERN = re.compile(r'\b(\d+\.?\d*)\b')
    PLAYER_LINE_PATTERN = re.compile(
        r"([A-Za-z][A-Za-z' -]+?)\s+"
        r"(over|under|o|u)\s+"
        r"(\d+\.?\d*)\s*"
        r"(pts|points|reb|rebounds|ast|assists|stl|steals|blk|blocks|tov|turnovers)?",
        re.IGNORECASE
    )
    
    VS_PATTERN = re.compile(r'\bvs\.?\s+([A-Za-z]{2,3}|[A-Za-z ]+?)(?:\s|$|\?)', re.IGNORECASE)
    
    DATE_PATTERN = re.compile(
        r'\b(\d{4}-\d{2}-\d{2})\b|'
        r'\b(today|tonight|tomorrow)\b|'
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b',
        re.IGNORECASE
    )
    
    def __init__(self):
        self._last_context: dict = {}
    
    def parse(self, query: str) -> ParsedQuery:
        query = query.strip()
        raw_query = query
        
        query_lower = query.lower()
        
        if query_lower in ('exit', 'quit', 'q', 'bye'):
            return ParsedQuery(query_type=QueryType.EXIT, raw_query=raw_query)
        
        if query_lower in ('help', 'h', '?'):
            return ParsedQuery(query_type=QueryType.HELP, raw_query=raw_query)
        
        if 'list players' in query_lower or 'show players' in query_lower:
            return ParsedQuery(query_type=QueryType.LIST_PLAYERS, raw_query=raw_query)
        
        if 'compare' in query_lower:
            return self._parse_compare(query, raw_query)
        
        parsed = self._parse_over_under(query, raw_query)
        if parsed.query_type != QueryType.UNKNOWN:
            return parsed
        
        parsed = self._parse_projection(query, raw_query)
        if parsed.query_type != QueryType.UNKNOWN:
            return parsed
        
        return ParsedQuery(query_type=QueryType.UNKNOWN, raw_query=raw_query)
    
    def _parse_over_under(self, query: str, raw_query: str) -> ParsedQuery:
        query_lower = query.lower()
        
        direction = None
        for alias, dir_val in DIRECTION_ALIASES.items():
            if f' {alias} ' in f' {query_lower} ' or query_lower.startswith(f'{alias} '):
                direction = dir_val
                break
        
        if direction is None:
            return ParsedQuery(query_type=QueryType.UNKNOWN, raw_query=raw_query)
        
        numbers = self.NUMBER_PATTERN.findall(query)
        line = None
        if numbers:
            for num_str in numbers:
                num = float(num_str)
                if 0.5 <= num <= 60:
                    line = num
                    break
        
        stat = None
        for alias, stat_val in STAT_ALIASES.items():
            pattern = rf'\b{re.escape(alias)}\b'
            if re.search(pattern, query_lower):
                stat = stat_val
                break
        
        if stat is None:
            if 'points' in query_lower or 'pts' in query_lower:
                stat = 'pts'
            elif 'rebounds' in query_lower or 'reb' in query_lower:
                stat = 'reb'
            elif 'assists' in query_lower or 'ast' in query_lower:
                stat = 'ast'
        
        player_name = self._extract_player_name(query, direction, line, stat)
        
        opponent = self._extract_opponent(query)
        team = self._extract_team(query)
        date = self._extract_date(query)
        
        if player_name:
            if stat is None:
                stat = self._last_context.get('stat', 'pts')
            if line is None:
                line = self._infer_line(stat)
            
            self._last_context = {
                'player_name': player_name,
                'stat': stat,
                'opponent': opponent,
                'team': team
            }
            
            return ParsedQuery(
                query_type=QueryType.OVER_UNDER,
                player_name=player_name,
                stat=stat,
                line=line,
                direction=direction,
                opponent=opponent,
                team=team,
                date=date,
                raw_query=raw_query
            )
        
        return ParsedQuery(query_type=QueryType.UNKNOWN, raw_query=raw_query)
    
    def _parse_projection(self, query: str, raw_query: str) -> ParsedQuery:
        query_lower = query.lower()
        
        projection_keywords = ['projection', 'projected', 'expected', 'forecast', 'stats', 'line']
        if not any(kw in query_lower for kw in projection_keywords):
            if self._last_context.get('player_name'):
                pass
            else:
                return ParsedQuery(query_type=QueryType.UNKNOWN, raw_query=raw_query)
        
        stat = None
        for alias, stat_val in STAT_ALIASES.items():
            if alias in query_lower:
                stat = stat_val
                break
        
        player_name = self._extract_player_name_simple(query)
        
        if player_name is None and self._last_context.get('player_name'):
            player_name = self._last_context['player_name']
        
        if stat is None and self._last_context.get('stat'):
            stat = self._last_context['stat']
        
        opponent = self._extract_opponent(query)
        team = self._extract_team(query)
        
        if player_name:
            return ParsedQuery(
                query_type=QueryType.PROJECTION,
                player_name=player_name,
                stat=stat,
                opponent=opponent or self._last_context.get('opponent'),
                team=team or self._last_context.get('team'),
                raw_query=raw_query
            )
        
        return ParsedQuery(query_type=QueryType.UNKNOWN, raw_query=raw_query)
    
    def _parse_compare(self, query: str, raw_query: str) -> ParsedQuery:
        query_lower = query.lower()
        
        stat = None
        for alias, stat_val in STAT_ALIASES.items():
            if alias in query_lower:
                stat = stat_val
                break
        
        if stat is None:
            stat = 'pts'
        
        words = query.split()
        potential_names = []
        
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in STAT_ALIASES and word.lower() not in ('compare', 'vs', 'and'):
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    potential_names.append(f"{word} {words[i + 1]}")
                else:
                    potential_names.append(word)
        
        players = []
        seen = set()
        for name in potential_names:
            name_lower = name.lower()
            if name_lower not in seen:
                players.append(name)
                seen.add(name_lower)
        
        return ParsedQuery(
            query_type=QueryType.COMPARE,
            compare_players=players[:2] if len(players) >= 2 else None,
            stat=stat,
            raw_query=raw_query
        )
    
    def _extract_player_name(self, query: str, direction: str, line: float, stat: str) -> Optional[str]:
        query_lower = query.lower()
        
        dir_pattern = '|'.join(DIRECTION_ALIASES.keys())
        stat_pattern = '|'.join(STAT_ALIASES.keys())
        
        pattern = rf"^([A-Za-z][A-Za-z' -]+?)\s+(?:{dir_pattern})"
        match = re.match(pattern, query_lower)
        
        if match:
            name = match.group(1).strip()
            name = self._clean_player_name(name)
            return name
        
        pattern = rf"^([A-Za-z][A-Za-z' -]+?)(?:\s+(?:{stat_pattern}))?(?:\s+(?:{dir_pattern}))"
        match = re.match(pattern, query_lower)
        
        if match:
            name = match.group(1).strip()
            name = self._clean_player_name(name)
            return name
        
        words = query.split()
        name_parts = []
        
        for word in words:
            word_lower = word.lower().rstrip('?,.!')
            if word_lower in DIRECTION_ALIASES:
                break
            if word_lower in STAT_ALIASES:
                continue
            if self.NUMBER_PATTERN.match(word):
                continue
            if word_lower in ('vs', 'tonight', 'today', 'tomorrow'):
                break
            if word[0].isupper() or (len(name_parts) > 0):
                name_parts.append(word)
        
        if name_parts:
            name = ' '.join(name_parts)
            name = self._clean_player_name(name)
            return name
        
        return None
    
    def _extract_player_name_simple(self, query: str) -> Optional[str]:
        words = query.split()
        name_parts = []
        
        for word in words:
            word_lower = word.lower().rstrip('?,.!')
            if word_lower in ('what', 'how', 'is', 'are', 'the', 'for', 'projection', 'projected', 'stats'):
                continue
            if word_lower in STAT_ALIASES:
                continue
            if word_lower in ('vs', 'tonight', 'today', 'tomorrow'):
                break
            if word[0].isupper():
                name_parts.append(word)
            elif len(name_parts) > 0:
                name_parts.append(word)
        
        if name_parts:
            return self._clean_player_name(' '.join(name_parts))
        
        return None
    
    def _clean_player_name(self, name: str) -> str:
        name = name.strip()
        
        fillers = ['the', 'a', 'an', 'is', 'does', 'will', 'can', 'has']
        words = name.split()
        words = [w for w in words if w.lower() not in fillers]
        
        return ' '.join(words).title()
    
    def _extract_opponent(self, query: str) -> Optional[str]:
        match = self.VS_PATTERN.search(query)
        if match:
            opp_raw = match.group(1).strip().lower()
            
            if opp_raw.upper() in TEAM_ABBREVIATIONS.values():
                return opp_raw.upper()
            
            for city_name, abbr in TEAM_ABBREVIATIONS.items():
                if city_name == opp_raw or city_name in opp_raw:
                    return abbr
            
            if len(opp_raw) <= 3:
                return opp_raw.upper()
        
        return None
    
    def _extract_team(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        
        for city_name, abbr in TEAM_ABBREVIATIONS.items():
            if city_name in query_lower:
                return abbr
        
        return None
    
    def _extract_date(self, query: str) -> Optional[str]:
        match = self.DATE_PATTERN.search(query)
        if match:
            if match.group(1):
                return match.group(1)
            elif match.group(2):
                return match.group(2)
        
        return None
    
    def _infer_line(self, stat: str) -> float:
        default_lines = {
            'pts': 15.5,
            'reb': 5.5,
            'ast': 4.5,
            'stl': 1.5,
            'blk': 0.5,
            'tov': 2.5
        }
        return default_lines.get(stat, 10.5)
    
    def get_last_context(self) -> dict:
        return self._last_context.copy()
    
    def clear_context(self):
        self._last_context = {}
