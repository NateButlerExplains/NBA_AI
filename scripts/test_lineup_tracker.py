#!/usr/bin/env python3
"""
Validation script for LineupTracker.
Read-only — does NOT modify any project code.

Tests:
1. Single game deep validation (BOS vs ATL, 2024-25 opener)
2. Cross-validation against NBA GameRotation API
3. Edge cases: older format (2005-06), overtime games, ejections
4. Scale test: 100 random games
"""

import sqlite3
import json
import sys
import time
import random
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pbp.event_parser import parse_events, detect_format
from src.pbp.lineup_tracker import LineupTracker

DB_PATH = "data/NBA_AI_full.sqlite"


def build_abbrev_to_team_id(conn):
    """Build abbreviation -> numeric team_id, including historical aliases."""
    mapping = {}
    for row in conn.execute(
        "SELECT team_id, abbreviation, alternatives FROM Teams"
    ).fetchall():
        tid_str, abbr, alts = row
        tid = int(tid_str)
        mapping[abbr] = tid
        if alts:
            for alt in json.loads(alts):
                if len(alt) <= 4 and alt == alt.upper():
                    mapping[alt] = tid
    return mapping


def load_game_events(conn, game_id):
    """Load and parse PBP events for a game."""
    rows = conn.execute(
        "SELECT play_id, log_data FROM PBP_Logs WHERE game_id = ? ORDER BY play_id",
        (game_id,),
    ).fetchall()
    if not rows:
        return None, []
    raw_events = [json.loads(log_data) for _, log_data in rows]
    fmt = detect_format(raw_events)
    events = parse_events(raw_events)
    return fmt, events


def get_team_ids_from_db(conn, game_id, abbrev_map):
    """Get home/away team IDs from Games table + abbreviation map."""
    row = conn.execute(
        "SELECT home_team, away_team FROM Games WHERE game_id = ?", (game_id,)
    ).fetchone()
    if not row:
        return None, None
    home_abbr, away_abbr = row
    home_id = abbrev_map.get(home_abbr)
    away_id = abbrev_map.get(away_abbr)
    return home_id, away_id


def get_team_ids_from_events(events):
    """Extract team IDs from events (fallback)."""
    team_ids = set()
    for evt in events:
        if evt.team_id and evt.team_id != 0:
            team_ids.add(evt.team_id)
    return sorted(team_ids)


def get_player_name(conn, person_id):
    """Look up player name from person_id."""
    row = conn.execute(
        "SELECT full_name FROM Players WHERE person_id = ?", (person_id,)
    ).fetchone()
    return row[0] if row else f"PID:{person_id}"


def validate_game(conn, game_id, abbrev_map, verbose=False):
    """
    Run LineupTracker on a game and validate outputs.
    Returns a dict with validation results.
    """
    result = {
        "game_id": game_id,
        "success": False,
        "format": None,
        "num_events": 0,
        "num_periods": 0,
        "starters_found_all": False,
        "always_5_per_team": True,
        "lineup_issues": [],
        "unique_players_per_team": {},
        "error": None,
    }

    try:
        fmt, events = load_game_events(conn, game_id)
        if not events:
            result["error"] = "No PBP events found"
            return result

        result["format"] = fmt
        result["num_events"] = len(events)

        # Get team IDs
        home_id, away_id = get_team_ids_from_db(conn, game_id, abbrev_map)
        if not home_id or not away_id:
            # Fallback: extract from events
            tids = get_team_ids_from_events(events)
            if len(tids) >= 2:
                home_id, away_id = tids[0], tids[1]
            else:
                result["error"] = f"Could not determine team IDs (found {tids})"
                return result

        # Create tracker
        tracker = LineupTracker(events, home_id, away_id)

        # Identify periods
        periods = sorted(set(evt.period for evt in events if evt.period > 0))
        result["num_periods"] = len(periods)

        all_players_seen = {home_id: set(), away_id: set()}
        issue_count = 0
        events_checked = 0
        events_with_wrong_count = 0

        for period in periods:
            tracker.initialize_period(period)
            period_events = [e for e in events if e.period == period]

            for evt in period_events:
                # Check lineup before processing each event
                current = tracker.get_current_players()
                for tid in [home_id, away_id]:
                    count = len(current.get(tid, set()))
                    all_players_seen[tid].update(current.get(tid, set()) - {0})

                    if count != 5:
                        events_with_wrong_count += 1
                        if issue_count < 10:
                            issue = (
                                f"P{period} {evt.clock_seconds:.0f}s "
                                f"team={tid} has {count} players "
                                f"(action={evt.action_type})"
                            )
                            result["lineup_issues"].append(issue)
                            if verbose:
                                print(f"  ISSUE: {issue}")
                        issue_count += 1

                events_checked += 1

                # Process substitutions
                if evt.is_substitution:
                    tracker.process_substitution(evt)

        result["always_5_per_team"] = events_with_wrong_count == 0
        result["events_checked"] = events_checked
        result["events_with_wrong_count"] = events_with_wrong_count
        result["starters_found_all"] = tracker.all_starters_found

        for tid in [home_id, away_id]:
            players = all_players_seen[tid] - {0}
            result["unique_players_per_team"][tid] = len(players)

        result["success"] = True

    except Exception as e:
        import traceback

        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()

    return result


# ===================================================================
# STEP 2: Deep test on a single recent game
# ===================================================================
def test_single_game(conn, abbrev_map, game_id="0022400001"):
    print("=" * 70)
    print(f"STEP 2: Deep validation on game {game_id}")
    print("=" * 70)

    home_id, away_id = get_team_ids_from_db(conn, game_id, abbrev_map)
    home_abbr = conn.execute(
        "SELECT abbreviation FROM Teams WHERE team_id = ?", (str(home_id),)
    ).fetchone()[0]
    away_abbr = conn.execute(
        "SELECT abbreviation FROM Teams WHERE team_id = ?", (str(away_id),)
    ).fetchone()[0]
    print(f"Game: {away_abbr} @ {home_abbr}")
    print(f"Home team ID: {home_id}, Away team ID: {away_id}")

    fmt, events = load_game_events(conn, game_id)
    print(f"Format: {fmt}, Events: {len(events)}")

    tracker = LineupTracker(events, home_id, away_id)

    periods = sorted(set(evt.period for evt in events if evt.period > 0))
    print(f"Periods: {periods}")

    for period in periods:
        print(f"\n--- Period {period} ---")
        starters = tracker.detect_period_starters(period)
        tracker.initialize_period(period)

        for tid in [home_id, away_id]:
            abbr = home_abbr if tid == home_id else away_abbr
            pids = starters[tid]
            names = [get_player_name(conn, pid) for pid in sorted(pids)]
            status = "OK" if len(pids) == 5 else f"ISSUE: {len(pids)} players"
            print(f"  {abbr} starters ({status}): {', '.join(names)}")

        # Track lineups through the period
        period_events = [e for e in events if e.period == period]
        sub_count = 0
        wrong_count = 0

        for evt in period_events:
            current = tracker.get_current_players()
            for tid in [home_id, away_id]:
                if len(current.get(tid, set())) != 5:
                    wrong_count += 1

            if evt.is_substitution:
                tracker.process_substitution(evt)
                sub_count += 1

        # Show final lineup of period
        final = tracker.get_current_players()
        for tid in [home_id, away_id]:
            abbr = home_abbr if tid == home_id else away_abbr
            final_names = [
                get_player_name(conn, pid) for pid in sorted(final.get(tid, set()))
            ]
            print(
                f"  {abbr} end-of-period: {len(final.get(tid, set()))} players - {', '.join(final_names)}"
            )

        print(f"  Substitution events: {sub_count}")
        print(f"  Events with wrong lineup count: {wrong_count}")

    # Final summary
    print(f"\n--- Summary ---")
    all_players = {home_id: set(), away_id: set()}
    tracker2 = LineupTracker(events, home_id, away_id)
    for period in periods:
        tracker2.initialize_period(period)
        for evt in [e for e in events if e.period == period]:
            for tid in [home_id, away_id]:
                all_players[tid].update(
                    tracker2.get_current_players().get(tid, set()) - {0}
                )
            if evt.is_substitution:
                tracker2.process_substitution(evt)

    for tid in [home_id, away_id]:
        abbr = home_abbr if tid == home_id else away_abbr
        n = len(all_players[tid])
        expected_ok = 8 <= n <= 13
        print(f"  {abbr}: {n} unique players {'OK' if expected_ok else 'UNEXPECTED'}")
        if n <= 15:
            names = [get_player_name(conn, pid) for pid in sorted(all_players[tid])]
            print(f"    Players: {', '.join(names)}")


# ===================================================================
# STEP 3: Cross-validate with GameRotation API
# ===================================================================
def test_cross_validate_api(conn, abbrev_map, game_id="0022400001"):
    print("\n" + "=" * 70)
    print(f"STEP 3: Cross-validate with NBA GameRotation API for {game_id}")
    print("=" * 70)

    try:
        from nba_api.stats.endpoints import gamerotation

        time.sleep(1)
        gr = gamerotation.GameRotation(game_id=game_id)
        dfs = gr.get_data_frames()
        away_rot = dfs[0]  # AwayTeam
        home_rot = dfs[1]  # HomeTeam

        print(
            f"API returned: {len(away_rot)} away rotation rows, {len(home_rot)} home rotation rows"
        )

        # Get unique players from API
        api_away_players = set(away_rot["PERSON_ID"].unique())
        api_home_players = set(home_rot["PERSON_ID"].unique())
        print(f"API away players: {len(api_away_players)}")
        print(f"API home players: {len(api_home_players)}")

        # Get our tracker's players
        home_id, away_id = get_team_ids_from_db(conn, game_id, abbrev_map)
        fmt, events = load_game_events(conn, game_id)
        tracker = LineupTracker(events, home_id, away_id)

        periods = sorted(set(evt.period for evt in events if evt.period > 0))
        our_players = {home_id: set(), away_id: set()}

        for period in periods:
            tracker.initialize_period(period)
            for evt in [e for e in events if e.period == period]:
                for tid in [home_id, away_id]:
                    our_players[tid].update(
                        tracker.get_current_players().get(tid, set()) - {0}
                    )
                if evt.is_substitution:
                    tracker.process_substitution(evt)

        print(f"\nOur tracker home players: {len(our_players[home_id])}")
        print(f"Our tracker away players: {len(our_players[away_id])}")

        # Compare
        def resolve_names(pids):
            return [f"{get_player_name(conn, pid)} ({pid})" for pid in pids]

        home_in_api_not_ours = api_home_players - our_players[home_id]
        home_in_ours_not_api = our_players[home_id] - api_home_players
        away_in_api_not_ours = api_away_players - our_players[away_id]
        away_in_ours_not_api = our_players[away_id] - api_away_players

        print(
            f"\nHome -- in API but not our tracker: {resolve_names(home_in_api_not_ours)}"
        )
        print(
            f"Home -- in our tracker but not API: {resolve_names(home_in_ours_not_api)}"
        )
        print(
            f"Away -- in API but not our tracker: {resolve_names(away_in_api_not_ours)}"
        )
        print(
            f"Away -- in our tracker but not API: {resolve_names(away_in_ours_not_api)}"
        )

        # Stint comparison: per-player stint counts from API
        home_stints = home_rot.groupby("PERSON_ID").size()
        away_stints = away_rot.groupby("PERSON_ID").size()

        print(f"\nStint comparison (home team):")
        for pid, count in home_stints.items():
            name = get_player_name(conn, pid)
            in_our = "YES" if pid in our_players[home_id] else "MISSING"
            print(f"    {name}: {count} stints [{in_our}]")

        print(f"  Away player stints (from API):")
        for pid, count in away_stints.items():
            name = get_player_name(conn, pid)
            in_our = "YES" if pid in our_players[away_id] else "MISSING"
            print(f"    {name}: {count} stints [{in_our}]")

        # Try a second game for additional validation
        print(f"\n--- Cross-validation on second game ---")
        game2 = "0022400100"
        time.sleep(1)
        gr2 = gamerotation.GameRotation(game_id=game2)
        dfs2 = gr2.get_data_frames()
        away_rot2 = dfs2[0]
        home_rot2 = dfs2[1]

        api_away_players2 = set(away_rot2["PERSON_ID"].unique())
        api_home_players2 = set(home_rot2["PERSON_ID"].unique())

        home_id2, away_id2 = get_team_ids_from_db(conn, game2, abbrev_map)
        fmt2, events2 = load_game_events(conn, game2)
        if events2 and home_id2 and away_id2:
            tracker2 = LineupTracker(events2, home_id2, away_id2)
            periods2 = sorted(set(evt.period for evt in events2 if evt.period > 0))
            our_players2 = {home_id2: set(), away_id2: set()}
            for period in periods2:
                tracker2.initialize_period(period)
                for evt in [e for e in events2 if e.period == period]:
                    for tid in [home_id2, away_id2]:
                        our_players2[tid].update(
                            tracker2.get_current_players().get(tid, set()) - {0}
                        )
                    if evt.is_substitution:
                        tracker2.process_substitution(evt)

            h_miss = api_home_players2 - our_players2[home_id2]
            a_miss = api_away_players2 - our_players2[away_id2]
            h_extra = our_players2[home_id2] - api_home_players2
            a_extra = our_players2[away_id2] - api_away_players2

            home_abbr2 = conn.execute(
                "SELECT abbreviation FROM Teams WHERE team_id = ?", (str(home_id2),)
            ).fetchone()
            away_abbr2 = conn.execute(
                "SELECT abbreviation FROM Teams WHERE team_id = ?", (str(away_id2),)
            ).fetchone()
            print(
                f"Game {game2} ({away_abbr2[0] if away_abbr2 else '?'} @ {home_abbr2[0] if home_abbr2 else '?'})"
            )
            print(
                f"  API: {len(api_home_players2)} home, {len(api_away_players2)} away"
            )
            print(
                f"  Ours: {len(our_players2[home_id2])} home, {len(our_players2[away_id2])} away"
            )
            print(f"  Home missing from tracker: {resolve_names(h_miss)}")
            print(f"  Away missing from tracker: {resolve_names(a_miss)}")
            print(f"  Home extra in tracker: {resolve_names(h_extra)}")
            print(f"  Away extra in tracker: {resolve_names(a_extra)}")

    except ImportError:
        print("nba_api not installed, skipping API cross-validation")
    except Exception as e:
        import traceback

        print(f"API cross-validation failed: {e}")
        traceback.print_exc()


# ===================================================================
# STEP 4: Edge cases
# ===================================================================
def test_edge_cases(conn, abbrev_map):
    print("\n" + "=" * 70)
    print("STEP 4: Edge case testing")
    print("=" * 70)

    # 4a: Older format game (try a few seasons)
    print("\n--- 4a: Older format game ---")
    for season_try in [2005, 2008, 2010, 2015]:
        old_game = conn.execute(
            """SELECT DISTINCT p.game_id FROM PBP_Logs p
               JOIN Games g ON p.game_id = g.game_id
               WHERE g.season = ?
               AND g.status = 3
               AND g.home_team != '' AND g.away_team != ''
               ORDER BY p.game_id LIMIT 1""",
            (season_try,),
        ).fetchone()
        if old_game:
            game_id = old_game[0]
            print(f"Testing game from {season_try}: {game_id}")
            r = validate_game(conn, game_id, abbrev_map, verbose=True)
            print(f"  Format: {r['format']}")
            print(f"  Events: {r['num_events']}, Periods: {r['num_periods']}")
            print(f"  Starters found all: {r['starters_found_all']}")
            print(f"  Always 5 per team: {r['always_5_per_team']}")
            print(f"  Events with wrong count: {r.get('events_with_wrong_count', '?')}")
            print(f"  Unique players: {r['unique_players_per_team']}")
            if r["error"]:
                print(f"  ERROR: {r['error']}")
            break
    else:
        print("  No older format game found")

    # 4b: Game with overtime (recent)
    print("\n--- 4b: Overtime game ---")
    ot_game = conn.execute(
        """SELECT p.game_id, COUNT(DISTINCT json_extract(p.log_data, '$.period')) as periods
           FROM PBP_Logs p
           JOIN Games g ON p.game_id = g.game_id
           WHERE g.season >= 2023 AND g.status = 3
           AND g.home_team != '' AND g.away_team != ''
           GROUP BY p.game_id
           HAVING periods > 4
           ORDER BY p.game_id
           LIMIT 1""",
    ).fetchone()
    if ot_game:
        game_id = ot_game[0]
        num_periods = ot_game[1]
        home_id, away_id = get_team_ids_from_db(conn, game_id, abbrev_map)
        print(f"Testing game: {game_id} ({num_periods} periods)")
        r = validate_game(conn, game_id, abbrev_map, verbose=True)
        print(f"  Format: {r['format']}")
        print(f"  Events: {r['num_events']}, Periods: {r['num_periods']}")
        print(f"  Starters found all: {r['starters_found_all']}")
        print(f"  Always 5 per team: {r['always_5_per_team']}")
        print(f"  Events with wrong count: {r.get('events_with_wrong_count', '?')}")
        print(f"  Unique players: {r['unique_players_per_team']}")
        if r["lineup_issues"]:
            for issue in r["lineup_issues"][:5]:
                print(f"    {issue}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
    else:
        print("  No overtime game found")

    # 4c: Game with ejection
    print("\n--- 4c: Game with ejection ---")
    ejection_game = conn.execute(
        """SELECT DISTINCT game_id FROM PBP_Logs
           WHERE json_extract(log_data, '$.actionType') = 'ejection'
           AND game_id IN (SELECT game_id FROM Games WHERE season >= 2023 AND status = 3)
           ORDER BY game_id
           LIMIT 1""",
    ).fetchone()
    if ejection_game:
        game_id = ejection_game[0]
        print(f"Testing game with ejection: {game_id}")
        r = validate_game(conn, game_id, abbrev_map, verbose=True)
        print(f"  Format: {r['format']}")
        print(f"  Events: {r['num_events']}, Periods: {r['num_periods']}")
        print(f"  Starters found all: {r['starters_found_all']}")
        print(f"  Always 5 per team: {r['always_5_per_team']}")
        print(f"  Events with wrong count: {r.get('events_with_wrong_count', '?')}")
        if r["lineup_issues"]:
            for issue in r["lineup_issues"][:5]:
                print(f"    {issue}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
    else:
        print("  No ejection game found in recent seasons")

    # 4d: Older format game with overtime
    print("\n--- 4d: Older format + OT ---")
    old_ot_game = conn.execute(
        """SELECT p.game_id, COUNT(DISTINCT json_extract(p.log_data, '$.period')) as periods
           FROM PBP_Logs p
           JOIN Games g ON p.game_id = g.game_id
           WHERE g.season BETWEEN 2005 AND 2015
           AND g.status = 3
           AND g.home_team != '' AND g.away_team != ''
           GROUP BY p.game_id
           HAVING periods > 4
           ORDER BY p.game_id
           LIMIT 1""",
    ).fetchone()
    if old_ot_game:
        game_id = old_ot_game[0]
        print(f"Testing game: {game_id} ({old_ot_game[1]} periods)")
        r = validate_game(conn, game_id, abbrev_map, verbose=True)
        print(f"  Format: {r['format']}")
        print(f"  Events: {r['num_events']}, Periods: {r['num_periods']}")
        print(f"  Starters found all: {r['starters_found_all']}")
        print(f"  Always 5 per team: {r['always_5_per_team']}")
        print(f"  Events with wrong count: {r.get('events_with_wrong_count', '?')}")
        if r["lineup_issues"]:
            for issue in r["lineup_issues"][:5]:
                print(f"    {issue}")
    else:
        print("  No older OT game found")


# ===================================================================
# STEP 5: Scale test
# ===================================================================
def test_at_scale(conn, abbrev_map, n_games=100):
    print("\n" + "=" * 70)
    print(f"STEP 5: Scale test -- {n_games} random games")
    print("=" * 70)

    random.seed(42)
    game_rows = conn.execute(
        """SELECT DISTINCT p.game_id, g.season
           FROM PBP_Logs p
           JOIN Games g ON p.game_id = g.game_id
           WHERE g.status = 3
           AND g.home_team != '' AND g.away_team != ''
           ORDER BY RANDOM()
           LIMIT ?""",
        (n_games,),
    ).fetchall()

    print(f"Selected {len(game_rows)} games")

    results = []
    season_counts = Counter()
    format_counts = Counter()
    success_count = 0
    always_5_count = 0
    starters_ok_count = 0
    error_types = Counter()
    total_events = 0
    total_wrong_events = 0

    for i, (game_id, season) in enumerate(game_rows):
        r = validate_game(conn, game_id, abbrev_map)
        results.append(r)
        season_counts[season] += 1

        if r["success"]:
            success_count += 1
            format_counts[r["format"]] += 1
            total_events += r.get("events_checked", 0)
            total_wrong_events += r.get("events_with_wrong_count", 0)

            if r["always_5_per_team"]:
                always_5_count += 1
            if r["starters_found_all"]:
                starters_ok_count += 1
        else:
            error_types[r.get("error", "unknown")] += 1

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(game_rows)}...")

    print(f"\n--- Results ---")
    print(f"Total games tested: {len(game_rows)}")
    print(
        f"Successfully processed: {success_count}/{len(game_rows)} "
        f"({100*success_count/len(game_rows):.1f}%)"
    )
    print(
        f"Always 5 per team: {always_5_count}/{success_count} "
        f"({100*always_5_count/max(success_count,1):.1f}%)"
    )
    print(
        f"All starters found: {starters_ok_count}/{success_count} "
        f"({100*starters_ok_count/max(success_count,1):.1f}%)"
    )
    print(f"Total events checked: {total_events}")
    print(
        f"Events with wrong lineup count: {total_wrong_events} "
        f"({100*total_wrong_events/max(total_events,1):.2f}%)"
    )

    print(f"\nBy format:")
    for fmt, count in format_counts.most_common():
        fmt_games = [r for r in results if r["format"] == fmt]
        fmt_ok = sum(1 for r in fmt_games if r["always_5_per_team"])
        fmt_starters_ok = sum(1 for r in fmt_games if r["starters_found_all"])
        print(
            f"  {fmt}: {count} games, {fmt_ok} perfect lineups ({100*fmt_ok/max(count,1):.1f}%), "
            f"{fmt_starters_ok} all starters found ({100*fmt_starters_ok/max(count,1):.1f}%)"
        )

    print(f"\nSeason distribution:")
    for season in sorted(season_counts.keys()):
        print(f"  {season}: {season_counts[season]} games")

    if error_types:
        print(f"\nErrors:")
        for err, count in error_types.most_common():
            print(f"  {err}: {count}")

    # Analyze failures
    failures = [r for r in results if r["success"] and not r["always_5_per_team"]]
    if failures:
        print(f"\n--- Failure analysis ({len(failures)} games with wrong counts) ---")

        # Group by format
        older_fails = [r for r in failures if r["format"] == "older"]
        v3_fails = [r for r in failures if r["format"] == "v3"]
        print(f"  Older format failures: {len(older_fails)}")
        print(f"  V3 format failures: {len(v3_fails)}")

        # Show details for some
        for r in failures[:10]:
            game_id = r["game_id"]
            season = next((s for gid, s in game_rows if gid == game_id), "?")
            wrong_pct = (
                100
                * r.get("events_with_wrong_count", 0)
                / max(r.get("events_checked", 1), 1)
            )
            print(
                f"  {game_id} (season {season}, format {r['format']}): "
                f"{r.get('events_with_wrong_count', '?')} wrong events ({wrong_pct:.1f}%), "
                f"starters={r['starters_found_all']}"
            )
            for issue in r["lineup_issues"][:3]:
                print(f"      {issue}")

    # Player count distribution (for successful games)
    print(f"\n--- Player count distribution (per team-game) ---")
    player_counts = []
    for r in results:
        if r["success"]:
            for tid, count in r["unique_players_per_team"].items():
                player_counts.append(count)
    if player_counts:
        from statistics import mean, median

        print(
            f"  Min: {min(player_counts)}, Max: {max(player_counts)}, "
            f"Mean: {mean(player_counts):.1f}, Median: {median(player_counts):.0f}"
        )
        count_dist = Counter(player_counts)
        for n in sorted(count_dist.keys()):
            print(f"    {n} players: {count_dist[n]} team-games")

    return results


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    abbrev_map = build_abbrev_to_team_id(conn)

    try:
        test_single_game(conn, abbrev_map, "0022400001")
        test_cross_validate_api(conn, abbrev_map, "0022400001")
        test_edge_cases(conn, abbrev_map)
        test_at_scale(conn, abbrev_map, n_games=100)
    finally:
        conn.close()
