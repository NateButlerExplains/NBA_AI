#!/usr/bin/env python3
"""
Fix unmatched player names in RAPMArchive table.

Progressively matches RAPM player names to Players table person_ids using:
1. Hard-coded manual overrides for known nickname/alternate name issues
2. Strip periods from initials (C.J. -> CJ)
3. Remove suffixes (Jr., Sr., III, IV, II) and re-match
4. Unicode/accent normalization (Dončić -> Doncic)
5. Fuzzy matching with difflib.SequenceMatcher (threshold 0.85)
6. First-name-initial + last-name matching
"""

import sqlite3
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.database import get_db

# ── Manual overrides for names that cannot be resolved algorithmically ──────
# Maps (rapm_name) -> person_id
# These are nicknames, legal name changes, or reversed East Asian names
# that no heuristic can reliably resolve.
MANUAL_OVERRIDES = {
    # Nicknames / alternate first names
    "Isaac Austin": 1134,  # Ike Austin
    "Isaac Fontaine": 1829,  # Ike Fontaine
    "Eugene Jeter": 200817,  # Pooh Jeter
    "Luigi Datome": 203540,  # Gigi Datome
    "Stanislav Medvedenko": 2098,  # Slava Medvedenko
    "Ronald Murray": 2436,  # Flip Murray
    "Mike Sweetney": 2552,  # Michael Sweetney
    "Efthimi Rentzias": 981,  # Efthimios Rentzias
    "Ibo Kutluay": 2825,  # Ibrahim Kutluay
    "Sviatoslav Mykhailiuk": 1629004,  # Svi Mykhailiuk
    "LaMark Baker": 1858,  # Mark Baker (only Baker active ~1998)
    "Norm Richardson": 2369,  # Norman Richardson
    "Rich Manning": 316,  # Richard Manning
    # Legal name changes
    "Nene Hilario": 2403,  # Nene (dropped Hilario)
    "Enes Kanter": 202683,  # Enes Freedom
    "Taurean Waller-Prince": 1627752,  # Taurean Prince
    # East Asian name order / DB format mismatch
    "Wang Zhizhi": 1917,  # DB: Zhi-zhi, Wang
    "Ha Seung-Jin": 2775,  # DB: Ha, Ha (likely data issue)
    "Sun Yue": 201180,  # DB: Sun, Sun (likely data issue)
    # Extra surname in RAPM
    "Horacio Llamas Grey": 1024,  # DB: Horacio Llamas
    "Didier Ilunga-Mbenga": 2788,  # DB: DJ Mbenga
    # Abbreviated first name in DB
    "Clarence Weatherspoon": 221,  # DB: Clar. Weatherspoon
    "Danny Schayes": 7,  # DB: Dan Schayes
    "Jeffrey Sheppard": 1852,  # DB: Jeff Sheppard
    "Steve Smith": 120,  # DB: Steven Smith (1991-2004)
    "Vince Hunter": 1626205,  # DB: Vincent Hunter
    "Cameron Reynolds": 1629244,  # DB: Cam Reynolds
    "Melvin Frazier": 1628982,  # DB: Melvin Frazier Jr.
    # Bogdan vs Bojan (different players, accent in DB)
    "Bogdan Bogdanovic": 203992,  # DB: Bogdan Bogdanović (with accent)
    # Walt Lemon comma variant
    "Walt Lemon, Jr.": 1627215,  # DB: Walt Lemon Jr.
    # Hyphenated surname
    "Nigel Hayes": 1628502,  # DB: Nigel Hayes-Davis
}


def strip_periods(name: str) -> str:
    """Strip periods from initials: 'C.J. McCollum' -> 'CJ McCollum'."""
    return name.replace(".", "")


def remove_suffixes(name: str) -> str:
    """Remove Jr., Sr., III, IV, II suffixes."""
    suffixes = [" Jr.", " Jr", " Sr.", " Sr", " III", " IV", " II"]
    result = name
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    # Also handle comma-separated suffix: "Walt Lemon, Jr."
    if ", Jr." in result or ", Sr." in result:
        result = result.replace(", Jr.", "").replace(", Sr.", "")
    return result.strip()


def normalize_unicode(name: str) -> str:
    """Normalize unicode characters: Dončić -> Doncic, Schröder -> Schroder."""
    # NFD decomposition splits accented chars into base + combining mark
    nfkd = unicodedata.normalize("NFKD", name)
    # Keep only non-combining characters
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def build_player_lookup(conn: sqlite3.Connection) -> dict:
    """
    Build lookup structures from the Players table.

    Returns dict with:
        - 'by_full': {normalized_display_name: person_id}
        - 'by_last_first': {(last_norm, first_norm): person_id}
        - 'by_last': {last_norm: [(person_id, first_name, last_name)]}
        - 'all_display': [(display_name, person_id)]  for fuzzy matching
    """
    cur = conn.cursor()
    cur.execute("SELECT person_id, first_name, last_name, full_name FROM Players")
    rows = cur.fetchall()

    by_full = {}  # "First Last" normalized -> person_id
    by_last_first = {}  # (last_norm, first_norm) -> person_id
    by_last = defaultdict(list)  # last_norm -> [(person_id, first, last)]
    all_display = []  # [(display_name, person_id)]

    for person_id, first_name, last_name, full_name in rows:
        first = (first_name or "").strip()
        last = (last_name or "").strip()

        # Build display name: "First Last" (what RAPM uses)
        if first and last:
            display = f"{first} {last}"
        elif last:
            display = last
        else:
            display = first

        # Store multiple normalized forms
        display_norm = strip_periods(display)
        display_no_suffix = remove_suffixes(display_norm)
        display_unicode = normalize_unicode(display_norm)
        display_both = normalize_unicode(display_no_suffix)

        for variant in {
            display,
            display_norm,
            display_no_suffix,
            display_unicode,
            display_both,
        }:
            key = variant.lower()
            if key not in by_full:
                by_full[key] = person_id

        first_norm = strip_periods(first).lower()
        last_norm = strip_periods(last).lower()
        by_last_first[(last_norm, first_norm)] = person_id
        by_last[last_norm].append((person_id, first, last))

        all_display.append((display, person_id))

    return {
        "by_full": by_full,
        "by_last_first": by_last_first,
        "by_last": by_last,
        "all_display": all_display,
    }


def match_player(
    rapm_name: str,
    lookup: dict,
    season: str = None,
) -> tuple[int | None, str]:
    """
    Try to match an RAPM player name to a person_id.

    Returns (person_id, match_method) or (None, "unmatched").
    """
    # 0. Manual overrides
    if rapm_name in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[rapm_name], "manual"

    by_full = lookup["by_full"]
    by_last = lookup["by_last"]
    all_display = lookup["all_display"]

    # 1. Strip periods (C.J. -> CJ)
    stripped = strip_periods(rapm_name).lower()
    if stripped in by_full:
        return by_full[stripped], "strip_periods"

    # 2. Remove suffixes (Dennis Smith -> match Dennis Smith Jr.)
    no_suffix = remove_suffixes(stripped).lower()
    if no_suffix != stripped and no_suffix in by_full:
        return by_full[no_suffix], "remove_suffix"

    # 2b. Add common suffixes to try matching
    for suffix in [" jr.", " sr.", " iii", " iv", " ii"]:
        with_suffix = no_suffix + suffix
        if with_suffix in by_full:
            return by_full[with_suffix], "add_suffix"

    # 3. Unicode normalization (Doncic -> match Dončić)
    unicode_norm = normalize_unicode(stripped)
    if unicode_norm in by_full:
        return by_full[unicode_norm], "unicode_norm"

    # 3b. Unicode + suffix removal
    unicode_no_suffix = normalize_unicode(no_suffix)
    if unicode_no_suffix in by_full:
        return by_full[unicode_no_suffix], "unicode_no_suffix"

    # 3c. Unicode + add suffix
    for suffix in [" jr.", " sr.", " iii", " iv", " ii"]:
        with_suffix = unicode_no_suffix + suffix
        if with_suffix in by_full:
            return by_full[with_suffix], "unicode_add_suffix"

    # 4. Last name matching with fuzzy first name
    parts = rapm_name.split()
    if len(parts) >= 2:
        rapm_first = parts[0]
        rapm_last = " ".join(parts[1:])  # Handle multi-word last names
        last_norm = strip_periods(rapm_last).lower()
        last_unicode = normalize_unicode(last_norm)

        # Try both normalized last name forms
        for last_key in {last_norm, last_unicode}:
            candidates = by_last.get(last_key, [])
            if len(candidates) == 1:
                # Only one player with that last name — high confidence
                return candidates[0][0], "unique_last_name"

    # 5. Fuzzy match (SequenceMatcher, threshold 0.85)
    best_ratio = 0.0
    best_pid = None
    rapm_lower = stripped
    for display, pid in all_display:
        display_lower = strip_periods(display).lower()
        ratio = SequenceMatcher(None, rapm_lower, display_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pid = pid
    if best_ratio >= 0.85:
        # Find the display name for reporting
        return best_pid, f"fuzzy({best_ratio:.3f})"

    # 6. First initial + last name
    if len(parts) >= 2:
        rapm_initial = strip_periods(parts[0])[0].lower()
        rapm_last = strip_periods(" ".join(parts[1:])).lower()
        rapm_last_unicode = normalize_unicode(rapm_last)

        for last_key in {rapm_last, rapm_last_unicode}:
            candidates = by_last.get(last_key, [])
            initial_matches = [
                (pid, first, last)
                for pid, first, last in candidates
                if first and strip_periods(first)[0].lower() == rapm_initial
            ]
            if len(initial_matches) == 1:
                return initial_matches[0][0], "initial_last"

    return None, "unmatched"


def main():
    with get_db() as conn:
        # 1. Query unmatched rows
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT player_name FROM RAPMArchive WHERE person_id IS NULL"
        )
        unmatched_names = sorted(set(row[0] for row in cur.fetchall()))
        print(f"Found {len(unmatched_names)} unique unmatched names\n")

        # 2. Build lookup from Players table
        lookup = build_player_lookup(conn)
        print(f"Loaded {len(lookup['all_display'])} players from Players table\n")

        # 3. Match each name
        matched = []
        still_unmatched = []

        for name in unmatched_names:
            pid, method = match_player(name, lookup)
            if pid is not None:
                matched.append((name, pid, method))
            else:
                still_unmatched.append(name)

        # 4. Print match report
        print("=" * 70)
        print(f"MATCHES FOUND: {len(matched)}")
        print("=" * 70)

        by_method = defaultdict(list)
        for name, pid, method in matched:
            by_method[method].append((name, pid))

        for method in sorted(by_method.keys()):
            entries = by_method[method]
            print(f"\n  [{method}] ({len(entries)} matches):")
            for name, pid in sorted(entries):
                # Look up the DB name for verification
                cur.execute(
                    "SELECT first_name, last_name FROM Players WHERE person_id = ?",
                    (pid,),
                )
                row = cur.fetchone()
                db_name = f"{row[0]} {row[1]}" if row else "???"
                print(f"    {name:40s} -> {db_name:30s} (id={pid})")

        print()
        print("=" * 70)
        print(f"STILL UNMATCHED: {len(still_unmatched)}")
        print("=" * 70)
        for name in still_unmatched:
            cur.execute(
                "SELECT season, team FROM RAPMArchive WHERE player_name = ? LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            ctx = f"({row[0]}, {row[1]})" if row else ""
            print(f"  {name:40s} {ctx}")

        # 5. Apply updates
        print()
        print("=" * 70)
        print("APPLYING UPDATES...")
        print("=" * 70)

        total_updated = 0
        for name, pid, method in matched:
            cur.execute(
                "UPDATE RAPMArchive SET person_id = ? WHERE player_name = ? AND person_id IS NULL",
                (pid, name),
            )
            rows_affected = cur.rowcount
            total_updated += rows_affected
            print(f"  Updated {rows_affected:3d} rows for {name}")

        conn.commit()
        print(f"\nTotal rows updated: {total_updated}")

        # 6. Final stats
        cur.execute("SELECT COUNT(*) FROM RAPMArchive")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM RAPMArchive WHERE person_id IS NOT NULL")
        matched_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM RAPMArchive WHERE person_id IS NULL")
        null_count = cur.fetchone()[0]

        print(f"\nFinal state:")
        print(f"  Total rows:    {total}")
        print(f"  Matched:       {matched_count} ({100*matched_count/total:.1f}%)")
        print(f"  Still NULL:    {null_count} ({100*null_count/total:.1f}%)")

        if null_count > 0:
            cur.execute(
                "SELECT DISTINCT player_name FROM RAPMArchive WHERE person_id IS NULL"
            )
            remaining = sorted(row[0] for row in cur.fetchall())
            print(f"  Remaining unmatched names ({len(remaining)}):")
            for name in remaining:
                print(f"    {name}")


if __name__ == "__main__":
    main()
