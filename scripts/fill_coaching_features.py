"""
Fill coaching feature placeholders (indices 24-27) in L3/L4 team_features cache.

Reads from:
    - Coaches table (head coaches per team-season)
    - Games table (game chronology, results)
    - Teams table (team_id -> abbreviation mapping)
    - data/l3l4_cache/team_features.npz (existing cache)

Updates indices 24-27 in-place:
    24: coach_tenure_games  — normalized by /500
    25: coach_career_winpct — career win% before this game (default 0.5)
    26: is_new_coach_flag   — 1.0 if different coach from prev season
    27: games_since_coach_change — normalized by /200
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
CACHE_PATH = PROJECT_ROOT / "data" / "l3l4_cache" / "team_features.npz"

# Games uses "2001-2002", Coaches uses "2001-02"
HISTORICAL_TO_MODERN = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHH": "CHA",
    "VAN": "MEM",
    "WSB": "WAS",
    "SDC": "LAC",
    "KCK": "SAC",
    "BUF": "LAC",
    "NOJ": "NOP",
    "CLT": "CHA",
}

TENURE_NORM = 500.0
GAMES_SINCE_NORM = 200.0


def games_season_to_coaches_season(games_season: str) -> str:
    """Convert '2001-2002' -> '2001-02'."""
    parts = games_season.split("-")
    return f"{parts[0]}-{parts[1][2:]}"


def build_team_id_to_abbrev(conn: sqlite3.Connection) -> dict[str, str]:
    """Build team_id (numeric str) -> current abbreviation mapping."""
    cur = conn.execute("SELECT team_id, abbreviation FROM Teams")
    return {str(row[0]): row[1] for row in cur}


def build_coach_mapping(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """
    Build (team_id, coaches_season) -> coach_id mapping for head coaches.

    For seasons with multiple head coaches (mid-season firing), pick the first
    one returned (typically the one who started the season).
    """
    cur = conn.execute("""
        SELECT team_id, season, coach_id, coach_name
        FROM Coaches
        WHERE is_head_coach = 1
        ORDER BY team_id, season, ROWID
    """)

    mapping: dict[tuple[str, str], str] = {}
    for team_id, season, coach_id, coach_name in cur:
        key = (team_id, season)
        if key not in mapping:
            mapping[key] = coach_id

    return mapping


def load_game_chronology(conn: sqlite3.Connection, cache_game_ids: set[str]):
    """
    Load all completed games from 2001-2002+ in chronological order.

    Returns list of (game_id, date_str, home_abbr, away_abbr, season,
                     home_score, away_score) sorted by date.
    """
    # Build abbreviation -> team_id mapping (including historical)
    tid_to_abbr = {}
    abbr_to_tid = {}
    cur = conn.execute("SELECT team_id, abbreviation FROM Teams")
    for team_id, abbr in cur:
        tid_to_abbr[str(team_id)] = abbr
        abbr_to_tid[abbr] = str(team_id)
    for hist, modern in HISTORICAL_TO_MODERN.items():
        if modern in abbr_to_tid:
            abbr_to_tid[hist] = abbr_to_tid[modern]

    # Load all game metadata
    cur = conn.execute("""
        SELECT game_id, date_time_utc, home_team, away_team, season
        FROM Games
        WHERE season >= '2001-2002'
          AND status = 3
          AND season_type IN ('Regular Season', 'Post Season')
        ORDER BY date_time_utc, game_id
    """)
    game_rows = cur.fetchall()

    # Load TeamBox scores: (game_id, team_id) -> pts
    score_map: dict[tuple[str, str], int] = {}
    cur = conn.execute("""
        SELECT tb.game_id, tb.team_id, tb.pts
        FROM TeamBox tb
        JOIN Games g ON tb.game_id = g.game_id
        WHERE g.season >= '2001-2002'
          AND g.status = 3
          AND g.season_type IN ('Regular Season', 'Post Season')
    """)
    for game_id, team_id, pts in cur:
        score_map[(game_id, str(team_id))] = pts

    games = []
    for game_id, date_utc, home, away, season in game_rows:
        home_tid = abbr_to_tid.get(home)
        away_tid = abbr_to_tid.get(away)
        home_pts = score_map.get((game_id, home_tid)) if home_tid else None
        away_pts = score_map.get((game_id, away_tid)) if away_tid else None
        games.append((game_id, date_utc, home, away, season, home_pts, away_pts))

    return games


def normalize_abbrev(abbr: str) -> str:
    """Normalize historical abbreviations to modern."""
    return HISTORICAL_TO_MODERN.get(abbr, abbr)


def main():
    print("Loading existing cache...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    game_ids_arr = data["game_ids"]
    features = data["features"].copy()  # (N, 2, 34) — make writable copy
    cache_game_ids = set(game_ids_arr)
    print(f"  Cache: {len(game_ids_arr)} games, features shape {features.shape}")

    # Build game_id -> index mapping for fast updates
    gid_to_idx = {gid: i for i, gid in enumerate(game_ids_arr)}

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    print("Building team_id -> abbreviation mapping...")
    tid_to_abbr = build_team_id_to_abbrev(conn)
    abbr_to_tid = {v: k for k, v in tid_to_abbr.items()}

    # Also handle historical abbreviations
    for hist, modern in HISTORICAL_TO_MODERN.items():
        if modern in abbr_to_tid and hist not in abbr_to_tid:
            abbr_to_tid[hist] = abbr_to_tid[modern]

    print("Building coach mapping from Coaches table...")
    coach_map = build_coach_mapping(conn)
    print(f"  {len(coach_map)} (team, season) entries")

    # Build abbr-based coach mapping: (abbr, games_season) -> coach_id
    # We need to go from team abbreviation + Games-format season to coach_id
    abbr_coach_map: dict[tuple[str, str], str] = {}
    for (team_id, coaches_season), coach_id in coach_map.items():
        abbr = tid_to_abbr.get(team_id, "???")
        # Convert coaches_season back to games_season for lookup
        # "2001-02" -> "2001-2002"
        year1 = int(coaches_season[:4])
        year2_short = coaches_season[5:]
        year2 = year1 + 1  # This works for NBA seasons (always consecutive years)
        games_season = f"{year1}-{year2}"
        abbr_coach_map[(abbr, games_season)] = coach_id
        # Also store for historical abbrevs that map to this modern abbrev
        for hist, modern in HISTORICAL_TO_MODERN.items():
            if modern == abbr:
                abbr_coach_map[(hist, games_season)] = coach_id

    print("Loading game chronology...")
    games = load_game_chronology(conn, cache_game_ids)
    print(f"  {len(games)} games loaded")

    conn.close()

    # -----------------------------------------------------------------------
    # Compute coaching features game-by-game in chronological order
    # -----------------------------------------------------------------------
    print("Computing coaching features...")

    # Running state per coach
    # coach_tenure[coach_id][team_abbr_normalized] = count of games coached for that team
    coach_team_tenure: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # coach_career: (wins, total_games) across all teams
    coach_career: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, games]

    # Track last season's coach for each team (normalized abbr)
    # prev_season_coach[norm_abbr] = coach_id
    prev_season_coach: dict[str, str] = {}
    # Current season coach tracking
    current_season_coach: dict[str, str] = {}
    # Track which season we've processed for prev_season detection
    last_processed_season: dict[str, str] = {}  # norm_abbr -> last season

    # For games_since_coach_change: track when the last coaching change happened
    # coach_change_game_count[norm_abbr] = number of games since we started counting
    coach_change_game_count: dict[str, int] = defaultdict(int)
    # was_new_coach[norm_abbr] = True if current season has a new coach
    was_new_coach: dict[str, bool] = {}

    # Sorted seasons for computing "previous season"
    all_seasons_set = set()
    for g in games:
        all_seasons_set.add(g[4])
    sorted_seasons = sorted(all_seasons_set)
    season_to_prev = {}
    for i, s in enumerate(sorted_seasons):
        if i > 0:
            season_to_prev[s] = sorted_seasons[i - 1]

    # Track which games we've seen per team per season (for tenure within season)
    team_season_game_count: dict[tuple[str, str], int] = defaultdict(int)

    filled = 0
    skipped_no_coach = 0

    for game_id, date_utc, home_abbr, away_abbr, season, home_pts, away_pts in games:
        home_norm = normalize_abbrev(home_abbr)
        away_norm = normalize_abbrev(away_abbr)

        home_coach = abbr_coach_map.get((home_abbr, season)) or abbr_coach_map.get(
            (home_norm, season)
        )
        away_coach = abbr_coach_map.get((away_abbr, season)) or abbr_coach_map.get(
            (away_norm, season)
        )

        # Determine if this game is in the cache
        in_cache = game_id in gid_to_idx

        for team_idx, (abbr, norm_abbr, coach_id, pts, opp_pts) in enumerate(
            [
                (home_abbr, home_norm, home_coach, home_pts, away_pts),
                (away_abbr, away_norm, away_coach, away_pts, home_pts),
            ]
        ):
            if coach_id is None:
                if in_cache:
                    skipped_no_coach += 1
                # Still update game counts for the team even without coach
                team_season_game_count[(norm_abbr, season)] += 1
                continue

            # --- Detect season transition for this team ---
            prev_season = season_to_prev.get(season)
            if (
                norm_abbr not in last_processed_season
                or last_processed_season[norm_abbr] != season
            ):
                # First game of a new season for this team
                # Record previous season's coach before updating
                if norm_abbr in current_season_coach:
                    prev_season_coach[norm_abbr] = current_season_coach[norm_abbr]
                current_season_coach[norm_abbr] = coach_id
                last_processed_season[norm_abbr] = season

                # Determine if this is a new coach
                if norm_abbr in prev_season_coach:
                    is_new = prev_season_coach[norm_abbr] != coach_id
                else:
                    is_new = False  # First season in data — treat as not new

                was_new_coach[norm_abbr] = is_new

                # Reset games_since_coach_change if new coach
                if is_new:
                    coach_change_game_count[norm_abbr] = 0
                # If same coach and no prior change tracking, count from start of last season
                # We just keep incrementing from wherever we are

            # --- Compute features BEFORE this game ---
            tenure = coach_team_tenure[coach_id][norm_abbr]

            career_wins, career_games = coach_career[coach_id]
            career_winpct = career_wins / career_games if career_games > 0 else 0.5

            is_new_flag = 1.0 if was_new_coach.get(norm_abbr, False) else 0.0

            games_since = coach_change_game_count.get(norm_abbr, 0)

            # --- Write to cache ---
            if in_cache:
                cache_idx = gid_to_idx[game_id]
                # team_idx: 0 = home (slot 0), 1 = away (slot 1)
                features[cache_idx, team_idx, 24] = tenure / TENURE_NORM
                features[cache_idx, team_idx, 25] = career_winpct
                features[cache_idx, team_idx, 26] = is_new_flag
                features[cache_idx, team_idx, 27] = games_since / GAMES_SINCE_NORM
                filled += 1

            # --- Update running state AFTER using pre-game values ---
            coach_team_tenure[coach_id][norm_abbr] += 1

            is_win = pts is not None and opp_pts is not None and pts > opp_pts
            coach_career[coach_id][1] += 1
            if is_win:
                coach_career[coach_id][0] += 1

            coach_change_game_count[norm_abbr] = games_since + 1
            team_season_game_count[(norm_abbr, season)] += 1

    print(f"  Filled {filled} team-game entries ({filled // 2} games)")
    print(f"  Skipped {skipped_no_coach} entries with no coach data")

    # -----------------------------------------------------------------------
    # Verify
    # -----------------------------------------------------------------------
    print("\nVerification:")
    for idx, name in [
        (24, "coach_tenure_games"),
        (25, "coach_career_winpct"),
        (26, "is_new_coach_flag"),
        (27, "games_since_coach_change"),
    ]:
        vals = features[:, :, idx].flatten()
        nonzero = np.count_nonzero(vals)
        print(
            f"  [{idx}] {name:30s}: min={vals.min():.4f} max={vals.max():.4f} "
            f"mean={vals.mean():.4f} std={vals.std():.4f} nonzero={nonzero}/{len(vals)}"
        )

    # Check for NaN
    nans = np.isnan(features[:, :, 24:28]).sum()
    print(f"\n  NaN count in indices 24-27: {nans}")

    # -----------------------------------------------------------------------
    # Spot checks
    # -----------------------------------------------------------------------
    print("\nSpot checks:")

    # Find Spoelstra (MIA) in a late season — should have high tenure
    # Spoelstra started 2008-09, by 2024-25 that's ~16 seasons * ~82 games = ~1300 games
    # Let's find a 2024-25 MIA game
    spot_checks = [
        ("Spoelstra tenure (MIA 2024-25)", "MIA", "2024-2025"),
        ("Popovich tenure (SAS 2023-24)", "SAS", "2023-2024"),
        ("Mitch Johnson (SAS 2024-25, new coach)", "SAS", "2024-2025"),
    ]

    conn2 = sqlite3.connect(str(DB_PATH))
    for label, team, season in spot_checks:
        cur = conn2.execute(
            """
            SELECT game_id FROM Games
            WHERE (home_team = ? OR away_team = ?) AND season = ?
            AND status = 3 AND season_type = 'Regular Season'
            ORDER BY date_time_utc DESC LIMIT 1
        """,
            (team, team, season),
        )
        row = cur.fetchone()
        if row is None:
            print(f"  {label}: no game found")
            continue
        gid = row[0]
        if gid not in gid_to_idx:
            print(f"  {label}: game {gid} not in cache")
            continue
        ci = gid_to_idx[gid]
        # Figure out if team is home or away
        cur2 = conn2.execute("SELECT home_team FROM Games WHERE game_id = ?", (gid,))
        ht = cur2.fetchone()[0]
        slot = 0 if ht == team else 1
        tenure = features[ci, slot, 24] * TENURE_NORM
        winpct = features[ci, slot, 25]
        is_new = features[ci, slot, 26]
        games_since = features[ci, slot, 27] * GAMES_SINCE_NORM
        print(f"  {label}:")
        print(
            f"    game_id={gid}, tenure={tenure:.0f} games, career_winpct={winpct:.3f}, "
            f"is_new={is_new:.0f}, games_since_change={games_since:.0f}"
        )
    conn2.close()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving updated cache to {CACHE_PATH}...")
    np.savez(CACHE_PATH, game_ids=game_ids_arr, features=features)
    print("Done.")


if __name__ == "__main__":
    main()
