"""
Event Tokenizer for NBA Play-by-Play Data.

Converts raw PBP JSON (joined with GameStates) into discrete tokens for transformer input.
Uses separate embeddings for action types and subtypes that are concatenated.

-- WHY A TOKENIZER? --
Neural networks (including transformers) only understand numbers, not strings.
This module works like an NLP tokenizer (think: word -> integer lookup table), but
instead of tokenizing English sentences, it tokenizes NBA play-by-play events.

Each play event from the NBA API is a JSON blob with fields like:
    {"actionType": "2pt", "subType": "Layup", "personId": 203999, "clock": "PT11M59.00S", ...}

The tokenizer assigns every unique string value a unique integer ID. For example:
    "2pt"   -> 5
    "Layup" -> 12
    "Dunk"  -> 13
These integer IDs are what get fed into the transformer's embedding layers.

-- VOCABULARY --
Each field (action type, sub type, player, etc.) maintains its own vocabulary,
which is simply a Python dict mapping raw values to integer IDs. Vocabularies
are built once by scanning the database, then saved to JSON for reuse.

Usage:
    tokenizer = PBPTokenizer()
    tokenizer.build_vocab_from_db(seasons=['2023-2024'])  # Build vocabulary
    tokens = tokenizer.tokenize_game(game_id)  # Tokenize a single game
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# NumPy is the standard library for numerical arrays in Python.
# We use it here to create arrays of integer token IDs that PyTorch can
# later wrap as tensors for model input.
import numpy as np

# get_db provides a managed SQLite connection to our NBA database.
from src.database import get_db

# ---------------------------------------------------------------------------
# Special Tokens
# ---------------------------------------------------------------------------
# These reserved tokens are standard in transformer/NLP work. They occupy the
# first few IDs (0-4) in every vocabulary so they are always available:
#
#   [PAD] (ID 0) - Padding token. When batching sequences of different lengths,
#                  shorter sequences are padded with this token so every sequence
#                  in the batch has the same length. The model learns to ignore it.
#
#   [UNK] (ID 1) - Unknown token. Used when a value was not seen during
#                  vocabulary building (or was too rare and got filtered out).
#                  Acts as a catch-all for out-of-vocabulary values.
#
#   [CLS] (ID 2) - Classification token. Placed at the start of a sequence;
#                  its final hidden state is often used as a summary
#                  representation for classification tasks.
#
#   [SEP] (ID 3) - Separator token. Used to mark boundaries between segments
#                  (e.g., between the first and second half of a game).
#
#   [MASK] (ID 4) - Mask token. Used during masked-language-model style
#                  pre-training: randomly replace some tokens with [MASK] and
#                  train the model to predict the original value.
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

# The order here determines each token's integer ID (index in the list).
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


@dataclass
class TokenizedPlay:
    """
    A single tokenized play-by-play event, with every field converted to an integer.

    Think of this as one "token" in our sequence. In NLP, a token might just be
    a single word ID; here, each token is a *composite* of multiple fields that
    together describe what happened on the court at a given moment.

    Each field has its own vocabulary / integer range so the transformer can
    learn separate embedding vectors for each feature dimension.
    """

    # Integer ID from the action_type vocabulary (e.g., "2pt" -> 5, "foul" -> 7).
    action_type_id: int

    # Integer ID from the sub_type vocabulary (e.g., "Layup" -> 12, "Driving" -> 14).
    # Sub type provides finer-grained detail about the action.
    sub_type_id: int

    # Game period (quarter): 1-4 for regulation, 5+ for overtime, clamped to 10 max.
    period: int

    # Clock bucket: seconds remaining in the period (0-720).
    # See _clock_to_bucket() for how the NBA's ISO-8601-style clock string
    # (e.g., "PT11M59.00S") gets converted to an integer bucket.
    clock_bucket: int

    # Which team is involved: 0 = home, 1 = away, 2 = neutral/unknown.
    team_indicator: int

    # Score differential bucket: encodes (home_score - away_score) as an integer
    # from 0 to 120, where bucket 60 means a tied game.  See _score_diff_to_bucket().
    score_diff_bucket: int

    # Integer ID from the player vocabulary, or None for events without a player
    # (e.g., period-start, timeout).
    player_id: Optional[int]

    # Shot outcome: 0 = not a shot, 1 = made, 2 = missed.
    shot_result: int


@dataclass
class TokenizedGame:
    """
    Container for an entire tokenized game.

    Holds the ordered sequence of TokenizedPlay objects (the model's input
    sequence) plus metadata about the game that may be useful for labels,
    filtering, or roster-aware embeddings.
    """

    game_id: str  # Unique NBA game identifier (e.g., "0022300001")
    home_team: str  # Home team tricode (e.g., "LAL")
    away_team: str  # Away team tricode (e.g., "BOS")
    plays: list[TokenizedPlay]  # Chronologically ordered tokenized plays
    home_score: (
        int  # Training target score (regulation-end when filtering OT, else final)
    )
    away_score: (
        int  # Training target score (regulation-end when filtering OT, else final)
    )
    final_home_score: int  # Always the true final score (including OT if played)
    final_away_score: int  # Always the true final score (including OT if played)
    is_overtime: bool  # Whether this game went to overtime (period > 4)
    player_ids_home: list[int]  # NBA player IDs who appeared for the home team
    player_ids_away: list[int]  # NBA player IDs who appeared for the away team


class PBPTokenizer:
    """
    Tokenizer for NBA play-by-play data.

    Builds separate vocabularies for action types and subtypes,
    and provides methods to tokenize games into sequences.

    High-level workflow:
        1. build_vocab_from_db()  -- Scan the database to discover all unique
           action types, sub types, and players. Assign each one an integer ID.
        2. tokenize_game()       -- Convert one game's raw JSON plays into a
           list of TokenizedPlay objects (all integers).
        3. plays_to_tensors()    -- Pack those TokenizedPlay objects into NumPy
           arrays that PyTorch can directly consume.
        4. save() / load()       -- Persist the vocabularies to JSON so you
           don't have to re-scan the database every time you train.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            db_path: Path to database. If None, uses config default.
        """
        # If no explicit database path is given, read it from the project's
        # central config file (config.yaml or similar).
        if db_path is None:
            from src.config import config

            db_path = config["database"]["path"]

        self.db_path = db_path

        # ------------------------------------------------------------------
        # Forward vocabularies: map raw value -> integer ID
        # ------------------------------------------------------------------
        # Each vocabulary is a simple dict. For example:
        #   action_type_vocab = {"[PAD]": 0, "[UNK]": 1, ..., "2pt": 5, "3pt": 6, ...}
        # The integer IDs are later used as indices into an nn.Embedding table
        # inside the transformer model.
        self.action_type_vocab: dict[str, int] = {}
        self.sub_type_vocab: dict[str, int] = {}
        # Player vocab maps NBA person IDs (large ints like 203999) to a
        # compact contiguous range starting at 0, suitable for embedding lookup.
        self.player_vocab: dict[int, int] = {}

        # ------------------------------------------------------------------
        # Reverse vocabularies: map integer ID -> raw value
        # ------------------------------------------------------------------
        # Useful for decoding / debugging: given a token ID, look up what
        # string or player it represents.
        self.id_to_action_type: dict[int, str] = {}
        self.id_to_sub_type: dict[int, str] = {}
        self.id_to_player: dict[int, int] = {}

        # ------------------------------------------------------------------
        # Clock discretization
        # ------------------------------------------------------------------
        # An NBA regulation period is 12 minutes = 720 seconds.  We convert
        # the game clock to "seconds remaining" and use that directly as the
        # bucket index.  721 buckets cover 0 through 720 seconds inclusive.
        # Each bucket = 1 second of game time.
        self.clock_buckets = 721  # 0 to 720 seconds inclusive

        # ------------------------------------------------------------------
        # Score differential discretization
        # ------------------------------------------------------------------
        # We encode the score difference (home - away) at the time of each
        # play.  The range -60 to +60 covers essentially all realistic NBA
        # score margins.  With a step of 1, that gives us 121 buckets:
        #   bucket 0   = diff of -60 (away blowout)
        #   bucket 60  = diff of  0  (tied game)
        #   bucket 120 = diff of +60 (home blowout)
        # Any actual differential outside [-60, +60] is clamped to the edge.
        self.score_diff_min = -60
        self.score_diff_max = 60
        self.score_diff_step = 1
        self.num_score_buckets = (
            self.score_diff_max - self.score_diff_min
        ) // self.score_diff_step + 1  # = 121

        # Guard flag: prevents tokenize_game() from being called before
        # the vocabulary has been built or loaded.
        self._initialized = False

    def build_vocab_from_db(
        self, seasons: Optional[list[str]] = None, min_count: int = 10
    ) -> None:
        """
        Build vocabularies by scanning every play-by-play row in the database.

        This is the "training" step for the tokenizer itself (not the model).
        It does three things:
            1. Counts how often each action type and sub type string appears.
            2. Collects every unique player ID.
            3. Filters out rare tokens that appear fewer than `min_count` times,
               since they are likely garbage / one-off data errors and would
               waste embedding table capacity.

        After this method returns, every vocabulary dict is populated and the
        tokenizer is ready to convert raw plays into integer IDs.

        Args:
            seasons: List of seasons to include (e.g., ['2023-2024']).
                    If None, uses all available data.
            min_count: Minimum number of occurrences for a token to earn its
                      own ID.  Tokens below this threshold get mapped to [UNK].
        """
        logging.info("Building vocabulary from database...")

        # Temporary counters -- we tally occurrences first, then decide which
        # tokens meet the min_count threshold.
        action_type_counts: dict[str, int] = {}
        sub_type_counts: dict[str, int] = {}
        player_ids: set[int] = set()

        with get_db(self.db_path) as conn:
            # Build an optional SQL WHERE clause to restrict to certain seasons.
            # Uses parameterized queries ("?") to prevent SQL injection.
            if seasons:
                placeholders = ",".join(["?"] * len(seasons))
                season_filter = f"AND g.season IN ({placeholders})"
                params = seasons
            else:
                season_filter = ""
                params = []

            # Pull every play-by-play JSON blob from finalized (status=3) games.
            # We JOIN on the Games table so we can filter by season.
            # Only include Regular Season and Post Season games — Pre Season
            # and All-Star games have different dynamics and rosters, and their
            # tokens would inflate the vocabulary without appearing in training.
            query = f"""
                SELECT p.log_data
                FROM PbP_Logs p
                JOIN Games g ON p.game_id = g.game_id
                WHERE g.game_data_finalized = 1
                AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                {season_filter}
            """

            cursor = conn.execute(query, params)

            # ------------------------------------------------------------------
            # Pass 1: Iterate over every play and count token occurrences.
            # ------------------------------------------------------------------
            count = 0
            for (log_data,) in cursor:
                # Each log_data is a JSON string; parse it into a Python dict.
                data = json.loads(log_data)

                # Count action types (e.g., "2pt", "3pt", "foul", "turnover")
                # Normalize: strip whitespace and lowercase to collapse duplicates
                # from inconsistent NBA API formatting across seasons.
                action_type = data.get("actionType", "").strip().lower()
                if action_type:
                    action_type_counts[action_type] = (
                        action_type_counts.get(action_type, 0) + 1
                    )

                # Count sub types (e.g., "Layup", "Driving", "Jump Shot")
                sub_type = data.get("subType", "").strip().lower()
                if sub_type:
                    sub_type_counts[sub_type] = sub_type_counts.get(sub_type, 0) + 1

                # Collect unique player IDs (personId=0 means "no player")
                person_id = data.get("personId")
                if person_id and person_id != 0:
                    player_ids.add(person_id)

                count += 1
                if count % 100000 == 0:
                    logging.debug(f"Processed {count} plays...")

        logging.info(f"Processed {count} total plays")

        # ------------------------------------------------------------------
        # Pass 2: Build the vocabularies from the collected counts.
        # ------------------------------------------------------------------

        # --- Action type vocabulary ---
        # Start by reserving IDs 0-4 for the special tokens ([PAD], [UNK], ...).
        self.action_type_vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        # Add real action types sorted by frequency (most common first).
        # Only include tokens that appeared at least `min_count` times.
        for action_type, cnt in sorted(action_type_counts.items(), key=lambda x: -x[1]):
            if cnt >= min_count:
                # Assign the next available integer ID.
                self.action_type_vocab[action_type] = len(self.action_type_vocab)

        # --- Sub type vocabulary ---
        # Same logic: special tokens first, then real sub types by frequency.
        self.sub_type_vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for sub_type, cnt in sorted(sub_type_counts.items(), key=lambda x: -x[1]):
            if cnt >= min_count:
                self.sub_type_vocab[sub_type] = len(self.sub_type_vocab)

        # --- Player vocabulary ---
        # Map NBA person IDs to a compact contiguous range (0, 1, 2, ...).
        # ID 0 is reserved for "no player" / padding.
        self.player_vocab = {0: 0}  # 0 = padding/no player
        for i, player_id in enumerate(sorted(player_ids), start=1):
            self.player_vocab[player_id] = i

        # ------------------------------------------------------------------
        # Build reverse (ID -> token) lookups for decoding / debugging.
        # ------------------------------------------------------------------
        self.id_to_action_type = {v: k for k, v in self.action_type_vocab.items()}
        self.id_to_sub_type = {v: k for k, v in self.sub_type_vocab.items()}
        self.id_to_player = {v: k for k, v in self.player_vocab.items()}

        # Mark the tokenizer as ready to use.
        self._initialized = True

        logging.info(
            f"Vocabulary built: {len(self.action_type_vocab)} action types, "
            f"{len(self.sub_type_vocab)} sub types, {len(self.player_vocab)} players"
        )

    def _clock_to_bucket(self, clock_str: str) -> int:
        """
        Convert an NBA clock string to an integer bucket (seconds remaining).

        The NBA API provides clock values in an ISO-8601-ish duration format:
            "PT11M59.00S"  ->  11 minutes and 59.00 seconds remaining
            "PT00M03.50S"  ->  0 minutes and 3.50 seconds remaining

        We convert this to total seconds (truncated to integer) and use that
        directly as the bucket index.  A regulation period is 12 minutes =
        720 seconds, so valid buckets are 0 through 720 (721 total).

        Why bucketize?  Neural networks need a fixed, finite number of
        possible values for each feature.  By converting the continuous clock
        into 721 discrete buckets (one per second), we can represent it as
        a single integer index into an embedding table.

        Returns 0 (period-end / buzzer) for missing or unparseable values.
        """
        if not clock_str:
            return 0

        try:
            # Regex captures minutes and seconds from "PT{M}M{S}S" format.
            match = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
            if match:
                minutes = int(match.group(1))
                seconds = int(float(match.group(2)))  # Truncate fractional seconds
                total_seconds = minutes * 60 + seconds
                # Clamp to valid range [0, 720] in case of overtime or data quirks.
                return min(total_seconds, self.clock_buckets - 1)
            return 0
        except Exception:
            return 0

    def _score_diff_to_bucket(self, home_score: int, away_score: int) -> int:
        """
        Convert the current score differential to an integer bucket index.

        The differential is always computed as (home_score - away_score), so:
            - Positive values mean the home team is winning.
            - Negative values mean the away team is winning.

        We clamp the differential to the range [-60, +60] (any blowout beyond
        60 points is mapped to the edge bucket).  Then we shift it so the
        result is a non-negative index suitable for embedding lookup:

            raw diff  ->  bucket index
            -60       ->  0
            -59       ->  1
             0        ->  60   (tied game)
            +59       ->  119
            +60       ->  120

        This gives 121 total buckets at 1-point resolution.
        """
        diff = home_score - away_score
        # Clamp to [-60, +60] to keep the bucket range bounded.
        diff = max(self.score_diff_min, min(self.score_diff_max, diff))
        # Shift so the minimum diff (-60) maps to bucket 0.
        # Since step=1, no division is needed.
        bucket = diff - self.score_diff_min
        return bucket

    def _get_team_indicator(
        self, team_tricode: Optional[str], home_team: str, away_team: str
    ) -> int:
        """
        Map a team tricode to a simple integer indicator.

        Returns:
            0 if the play's team matches the home team,
            1 if it matches the away team,
            2 for neutral events (e.g., jump ball, period start) or when
              the team tricode is missing / doesn't match either team.
        """
        if not team_tricode:
            return 2
        if team_tricode == home_team:
            return 0
        if team_tricode == away_team:
            return 1
        return 2

    def _get_shot_result(self, data: dict) -> int:
        """
        Extract the shot outcome from a play's JSON data.

        Returns:
            0 if the play is not a shot (most plays: fouls, turnovers, etc.),
            1 if the shot was made,
            2 if the shot was missed.

        This three-way encoding lets the model learn different patterns for
        makes vs. misses without needing a separate "is this a shot?" flag.
        """
        shot_result = data.get("shotResult", "")
        if shot_result == "Made":
            return 1
        if shot_result == "Missed":
            return 2
        return 0

    def tokenize_play(
        self, data: dict, home_team: str, away_team: str
    ) -> TokenizedPlay:
        """
        Tokenize a single play-by-play event.

        This is the core conversion step: it takes one raw JSON dict from
        the NBA API and converts every relevant field into an integer
        suitable for neural network input.

        For string fields (action type, sub type), we look them up in the
        vocabulary dict built by build_vocab_from_db().  If a value was not
        seen during vocab building (or was too rare), it falls back to the
        [UNK] token ID.

        For numeric/continuous fields (clock, score), we use bucket functions
        to discretize them into a fixed number of integer bins.

        Args:
            data: Raw PBP JSON data for the play (one dict from the NBA API).
            home_team: Home team tricode (e.g., "LAL").
            away_team: Away team tricode (e.g., "BOS").

        Returns:
            TokenizedPlay with all fields converted to integer IDs.
        """
        # --- Action type: look up in vocab, fall back to [UNK] if unseen ---
        # Normalize to match the vocabulary (built with .strip().lower()).
        action_type = data.get("actionType", "").strip().lower()
        action_type_id = self.action_type_vocab.get(
            action_type, self.action_type_vocab[UNK_TOKEN]
        )

        # --- Sub type: same vocab-lookup pattern ---
        sub_type = data.get("subType", "").strip().lower()
        sub_type_id = self.sub_type_vocab.get(sub_type, self.sub_type_vocab[UNK_TOKEN])

        # --- Period: 1-4 for regulation quarters, 5+ for overtime ---
        # Clamp to [1, 10] to avoid out-of-range embedding indices.
        period = min(max(data.get("period", 1), 1), 10)

        # --- Clock: convert ISO-duration string to seconds-remaining bucket ---
        clock_bucket = self._clock_to_bucket(data.get("clock", ""))

        # --- Team indicator: home (0), away (1), or neutral (2) ---
        team_tricode = data.get("teamTricode")
        team_indicator = self._get_team_indicator(team_tricode, home_team, away_team)

        # --- Score differential bucket ---
        # The NBA API sometimes sends empty strings or None for scores at the
        # very start of the game, so we guard against those edge cases.
        home_score_raw = data.get("scoreHome", 0)
        away_score_raw = data.get("scoreAway", 0)
        home_score = int(home_score_raw) if home_score_raw not in (None, "", " ") else 0
        away_score = int(away_score_raw) if away_score_raw not in (None, "", " ") else 0
        score_diff_bucket = self._score_diff_to_bucket(home_score, away_score)

        # --- Player ID: map the NBA person ID to our compact vocab index ---
        # Events without a player (period start, timeouts, etc.) get None.
        person_id = data.get("personId")
        if person_id and person_id != 0:
            # Falls back to 0 (padding) if the player wasn't in the vocab.
            player_id = self.player_vocab.get(person_id, 0)
        else:
            player_id = None

        # --- Shot result: 0=not a shot, 1=made, 2=missed ---
        shot_result = self._get_shot_result(data)

        return TokenizedPlay(
            action_type_id=action_type_id,
            sub_type_id=sub_type_id,
            period=period,
            clock_bucket=clock_bucket,
            team_indicator=team_indicator,
            score_diff_bucket=score_diff_bucket,
            player_id=player_id,
            shot_result=shot_result,
        )

    def tokenize_game(self, game_id: str) -> Optional[TokenizedGame]:
        """
        Tokenize every play in a single game, producing the full input sequence.

        This method:
            1. Looks up the game's home/away teams from the Games table.
            2. Fetches all play-by-play rows (joined with GameStates to get
               running scores and the final score flag).
            3. Calls tokenize_play() on each row to convert it to integers.
            4. Collects per-team rosters (which players appeared for each side).
            5. Returns a TokenizedGame containing the ordered list of plays.

        OT plays (period > 4) are excluded from the play sequence and
        regulation-end scores are used as training targets. Final scores
        (including OT) are always captured for evaluation purposes.

        The resulting TokenizedGame is the "sentence" that the transformer
        will consume -- each TokenizedPlay is one "word" in the sequence.

        Args:
            game_id: NBA game ID (e.g., "0022300001").

        Returns:
            TokenizedGame with all plays tokenized, or None if the game is
            not found or has no usable plays.
        """
        # Safety check: vocabulary must exist before we can map values to IDs.
        if not self._initialized:
            raise RuntimeError(
                "Tokenizer not initialized. Call build_vocab_from_db() first."
            )

        with get_db(self.db_path) as conn:
            # Step 1: Fetch basic game metadata (home/away team tricodes).
            cursor = conn.execute(
                """
                SELECT home_team, away_team
                FROM Games
                WHERE game_id = ?
                """,
                (game_id,),
            )
            row = cursor.fetchone()
            if not row:
                logging.warning(f"Game {game_id} not found")
                return None

            home_team, away_team = row

            # Step 2: Fetch every play-by-play log, joined with the
            # corresponding GameState row so we also have the running score
            # and a flag indicating whether this is the game's final state.
            # We always fetch ALL plays (including OT) because we need the
            # final score and rosters regardless of filtering.
            cursor = conn.execute(
                """
                SELECT p.log_data, gs.home_score, gs.away_score, gs.is_final_state
                FROM PbP_Logs p
                JOIN GameStates gs ON p.game_id = gs.game_id AND p.play_id = gs.play_id
                WHERE p.game_id = ?
                ORDER BY p.play_id
                """,
                (game_id,),
            )

            plays = []
            final_home_score = 0
            final_away_score = 0
            reg_home_score = 0  # Score at end of regulation (last period <= 4 play)
            reg_away_score = 0
            is_overtime = False
            player_ids_home = set()
            player_ids_away = set()

            # Step 3: Iterate over plays in chronological order and tokenize.
            for log_data, home_score, away_score, is_final in cursor:
                data = json.loads(log_data)

                # Skip "system" plays that lack a description (e.g., internal
                # NBA API housekeeping events).  These were already filtered
                # during game_states.py ingestion but we double-check here.
                if "description" not in data:
                    continue

                period = data.get("period", 1)

                # Track regulation-end score: update on every regulation play.
                # Since plays are chronological, the last update before any
                # period-5 play is the regulation-end score.
                if period <= 4:
                    reg_home_score = home_score
                    reg_away_score = away_score

                if period > 4:
                    is_overtime = True

                # Skip OT plays but still capture the final score and
                # collect roster info.
                if period > 4:
                    if is_final:
                        final_home_score = home_score
                        final_away_score = away_score
                    # Still collect roster from OT plays — a player who only
                    # appeared in OT is still part of the game roster.
                    person_id = data.get("personId")
                    team_tricode = data.get("teamTricode")
                    if person_id and person_id != 0 and team_tricode:
                        if team_tricode == home_team:
                            player_ids_home.add(person_id)
                        elif team_tricode == away_team:
                            player_ids_away.add(person_id)
                    continue

                # Convert this play's JSON into a TokenizedPlay (all integers).
                tokenized = self.tokenize_play(data, home_team, away_team)
                plays.append(tokenized)

                # Remember the final score when we hit the last state.
                if is_final:
                    final_home_score = home_score
                    final_away_score = away_score

                # Build per-team rosters by collecting player IDs as we go.
                person_id = data.get("personId")
                team_tricode = data.get("teamTricode")
                if person_id and person_id != 0 and team_tricode:
                    if team_tricode == home_team:
                        player_ids_home.add(person_id)
                    elif team_tricode == away_team:
                        player_ids_away.add(person_id)

            if not plays:
                logging.warning(f"No plays found for game {game_id}")
                return None

            # For non-OT games, regulation scores equal final scores.
            if not is_overtime:
                reg_home_score = final_home_score
                reg_away_score = final_away_score

            # Package everything into a TokenizedGame for downstream use.
            # home_score/away_score are regulation scores (training targets).
            # final_home_score/final_away_score include OT (for evaluation).
            return TokenizedGame(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                plays=plays,
                home_score=reg_home_score,
                away_score=reg_away_score,
                final_home_score=final_home_score,
                final_away_score=final_away_score,
                is_overtime=is_overtime,
                player_ids_home=list(player_ids_home),
                player_ids_away=list(player_ids_away),
            )

    def plays_to_tensors(self, plays: list[TokenizedPlay]) -> dict[str, np.ndarray]:
        """
        Convert a list of TokenizedPlay dataclass objects into NumPy arrays.

        PyTorch models expect numerical tensors, not Python dataclasses.  This
        method bridges that gap by packing each field of TokenizedPlay into its
        own 1-D NumPy array of shape (num_plays,).

        The returned dict can be passed directly to torch.from_numpy() or
        used to construct a PyTorch Dataset.  Each array uses int64 dtype
        because PyTorch's nn.Embedding requires LongTensor (int64) indices.

        Example output for a game with 450 plays:
            {
                "action_type_ids":    np.array of shape (450,),
                "sub_type_ids":       np.array of shape (450,),
                "periods":            np.array of shape (450,),
                "clock_buckets":      np.array of shape (450,),
                "team_indicators":    np.array of shape (450,),
                "score_diff_buckets": np.array of shape (450,),
                "player_ids":         np.array of shape (450,),
                "shot_results":       np.array of shape (450,),
            }

        Returns:
            Dict mapping field names to 1-D int64 NumPy arrays.
        """
        n = len(plays)

        return {
            "action_type_ids": np.array(
                [p.action_type_id for p in plays], dtype=np.int64
            ),
            "sub_type_ids": np.array([p.sub_type_id for p in plays], dtype=np.int64),
            "periods": np.array([p.period for p in plays], dtype=np.int64),
            "clock_buckets": np.array([p.clock_bucket for p in plays], dtype=np.int64),
            "team_indicators": np.array(
                [p.team_indicator for p in plays], dtype=np.int64
            ),
            "score_diff_buckets": np.array(
                [p.score_diff_bucket for p in plays], dtype=np.int64
            ),
            # Replace None player IDs with 0 (the padding index) so every
            # element in the array is a valid integer.
            "player_ids": np.array(
                [p.player_id if p.player_id is not None else 0 for p in plays],
                dtype=np.int64,
            ),
            "shot_results": np.array([p.shot_result for p in plays], dtype=np.int64),
        }

    def save(self, path: str) -> None:
        """
        Serialize the tokenizer's vocabulary mappings to a JSON file.

        This lets you build the vocabulary once (slow database scan) and
        then reload it instantly for future training runs or inference.
        The saved file contains:
            - action_type_vocab: {string -> int} for action types
            - sub_type_vocab:    {string -> int} for sub types
            - player_vocab:      {nba_player_id -> int} for players
            - Discretization parameters (clock_buckets, score_diff_*)

        Note: JSON keys must be strings, so player_vocab's integer keys
        are converted to strings on save and back to ints on load.
        """
        data = {
            "action_type_vocab": self.action_type_vocab,
            "sub_type_vocab": self.sub_type_vocab,
            # JSON requires string keys, so we stringify the NBA player IDs.
            "player_vocab": {str(k): v for k, v in self.player_vocab.items()},
            "clock_buckets": self.clock_buckets,
            "score_diff_min": self.score_diff_min,
            "score_diff_max": self.score_diff_max,
            "score_diff_step": self.score_diff_step,
        }

        path = Path(path)
        # Create parent directories if they don't already exist.
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Tokenizer saved to {path}")

    def load(self, path: str) -> None:
        """
        Load previously saved vocabulary mappings from a JSON file.

        This is the counterpart to save().  After loading, the tokenizer is
        immediately ready to tokenize plays -- no need to call
        build_vocab_from_db() again.

        The player vocab keys are converted back from strings to ints
        (JSON only supports string keys).
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Restore forward vocabularies.
        self.action_type_vocab = data["action_type_vocab"]
        self.sub_type_vocab = data["sub_type_vocab"]
        # Convert stringified NBA player IDs back to integers.
        self.player_vocab = {int(k): v for k, v in data["player_vocab"].items()}

        # Restore discretization parameters.
        self.clock_buckets = data["clock_buckets"]
        self.score_diff_min = data["score_diff_min"]
        self.score_diff_max = data["score_diff_max"]
        self.score_diff_step = data["score_diff_step"]

        # Rebuild reverse vocabularies (ID -> token) from the forward vocabs
        # so we can decode token IDs back to human-readable strings.
        self.id_to_action_type = {v: k for k, v in self.action_type_vocab.items()}
        self.id_to_sub_type = {v: k for k, v in self.sub_type_vocab.items()}
        self.id_to_player = {v: k for k, v in self.player_vocab.items()}

        # Recompute the number of score buckets from the loaded parameters.
        self.num_score_buckets = (
            self.score_diff_max - self.score_diff_min
        ) // self.score_diff_step + 1

        self._initialized = True
        logging.info(f"Tokenizer loaded from {path}")

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """
        Return the number of unique values for each token field.

        These sizes are used to configure the nn.Embedding layers in the
        transformer model.  Each embedding layer needs to know how many
        rows its lookup table should have (one row per possible token ID).

        For example, if action_type has a vocab size of 25, the model
        creates an embedding table of shape (25, embed_dim) so that each
        of the 25 action types gets its own learned vector.
        """
        return {
            "action_type": len(self.action_type_vocab),
            "sub_type": len(self.sub_type_vocab),
            "player": len(self.player_vocab),
            "period": 11,  # Periods 1-10 + index 0 reserved for padding
            "clock_bucket": self.clock_buckets,  # 721 (0-720 seconds)
            "team_indicator": 3,  # home, away, neutral
            "score_diff_bucket": self.num_score_buckets,  # 121 (-60 to +60)
            "shot_result": 3,  # not a shot, made, missed
        }


def test_tokenizer():
    """
    Smoke-test the full tokenizer pipeline on a single game.

    Exercises the complete workflow:
        1. Build vocabulary from the database.
        2. Tokenize one game into a sequence of integer IDs.
        3. Convert the tokenized plays to NumPy arrays (tensor-ready).
        4. Save the vocabulary to disk and reload it to verify round-tripping.
    """
    import sys

    logging.basicConfig(level=logging.INFO)

    tokenizer = PBPTokenizer()

    # Step 1: Build vocabulary from one season of data.
    # min_count=5 is more permissive than the default (10), keeping more
    # tokens for dev/testing purposes.
    tokenizer.build_vocab_from_db(seasons=["2023-2024"], min_count=5)

    print(f"\nVocabulary sizes: {tokenizer.vocab_sizes}")

    # Grab the first finalized game from the database to use as a test case.
    with get_db(tokenizer.db_path) as conn:
        cursor = conn.execute(
            "SELECT game_id FROM Games WHERE status = 3 AND game_data_finalized = 1 LIMIT 1"
        )
        game_id = cursor.fetchone()[0]

    # Step 2: Tokenize that game's play-by-play sequence.
    print(f"\nTokenizing game {game_id}...")
    result = tokenizer.tokenize_game(game_id)

    if result:
        print(f"  Home: {result.home_team}, Away: {result.away_team}")
        print(f"  Final: {result.home_score} - {result.away_score}")
        print(f"  Total plays: {len(result.plays)}")
        print(f"  Home players: {len(result.player_ids_home)}")
        print(f"  Away players: {len(result.player_ids_away)}")

        # Decode the first few plays back to human-readable strings using
        # the reverse vocabulary, to verify the round-trip makes sense.
        print("\n  First 5 plays:")
        for i, play in enumerate(result.plays[:5]):
            action = tokenizer.id_to_action_type.get(play.action_type_id, "?")
            sub = tokenizer.id_to_sub_type.get(play.sub_type_id, "?")
            print(
                f"    {i+1}. {action}|{sub} P{play.period} "
                f"clock={play.clock_bucket} team={play.team_indicator} "
                f"score_diff={play.score_diff_bucket}"
            )

        # Step 3: Convert tokenized plays to NumPy arrays (what PyTorch eats).
        tensors = tokenizer.plays_to_tensors(result.plays)
        print(f"\n  Tensor shapes:")
        for name, arr in tensors.items():
            print(f"    {name}: {arr.shape}")

        # Step 4: Test save/load round-trip to confirm the vocab survives
        # serialization to JSON and back.
        save_path = "data/tokenized/test_tokenizer.json"
        tokenizer.save(save_path)
        print(f"\n  Saved to {save_path}")

        tokenizer2 = PBPTokenizer()
        tokenizer2.load(save_path)
        print(f"  Reloaded vocab sizes: {tokenizer2.vocab_sizes}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_tokenizer())
