"""
Test PlayerValidator to verify validation checks.
"""

import sqlite3
import tempfile

from src.database_updater.validators import PlayerValidator, Severity


def test_player_validation():
    """Test PlayerValidator detects NULL names and validates count."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    # Setup test data
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create Players table
        cursor.execute(
            """
            CREATE TABLE Players (
                person_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                from_year INTEGER,
                to_year INTEGER,
                roster_status TEXT,
                team TEXT
            )
        """
        )

        # Insert test players
        cursor.execute(
            """
            INSERT INTO Players VALUES 
            (1, 'LeBron', 'James', 'James, LeBron', 2003, 2025, 'Active', 'LAL'),
            (2, 'Stephen', 'Curry', 'Curry, Stephen', 2009, 2025, 'Active', 'GSW'),
            (3, NULL, 'Durant', 'Durant, Kevin', 2007, 2025, 'Active', 'PHX'),
            (4, 'Giannis', NULL, 'Antetokounmpo, Giannis', 2013, 2025, 'Active', 'MIL'),
            (5, 'Luka', 'Doncic', NULL, 2018, 2025, 'Active', 'DAL')
        """
        )
        conn.commit()

        # Test validation
        validator = PlayerValidator()

        # Test 1: Validate specific players (should find NULL names)
        result = validator.validate([1, 2, 3, 4, 5], cursor)

        print(f"\nValidation result: {result.summary()}")
        print(f"Has critical issues: {result.has_critical_issues}")
        print(f"Has warnings: {result.has_warnings}")
        print(f"Log suffix: {result.log_suffix()}")

        # Should find 3 players with NULL names (person_id 3, 4, 5)
        assert result.has_critical_issues, "Should detect NULL names"
        null_name_issue = next(
            (i for i in result.issues if i.check_id == "NULL_NAMES"), None
        )
        assert null_name_issue is not None, "Should have NULL_NAMES issue"
        assert (
            null_name_issue.count == 3
        ), f"Expected 3 NULL names, got {null_name_issue.count}"
        assert (
            null_name_issue.severity == Severity.CRITICAL
        ), "NULL names should be CRITICAL"

        print("✅ NULL name validation works")

        # Test 2: Validate total count (should be low for only 5 players)
        count_result = validator.validate_total_count(cursor)

        print(f"\nCount validation: {count_result.summary()}")

        # Should find LOW_PLAYER_COUNT issue (only 5 players, expected 400+)
        assert count_result.has_critical_issues, "Should detect low player count"
        low_count_issue = next(
            (i for i in count_result.issues if i.check_id == "LOW_PLAYER_COUNT"), None
        )
        assert low_count_issue is not None, "Should have LOW_PLAYER_COUNT issue"
        assert (
            low_count_issue.count == 5
        ), f"Expected count 5, got {low_count_issue.count}"

        print("✅ Player count validation works")


if __name__ == "__main__":
    test_player_validation()
    print("\n✅ All PlayerValidator tests passed")
