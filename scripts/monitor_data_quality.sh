#!/bin/bash
# Daily database health monitoring script
# Run via cron: 0 6 * * * /path/to/NBA_AI/scripts/monitor_data_quality.sh
#
# Uses database_evaluator.py to check database health

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
cd "$PROJECT_ROOT"
source venv/bin/activate

# Get current date for log filename
DATE=$(date +%Y-%m-%d)
REPORT_DIR="data/quality_reports"
mkdir -p $REPORT_DIR

# Run database evaluation for current season
CURRENT_SEASON="2024-2025"
REPORT_FILE="${REPORT_DIR}/health_check_${CURRENT_SEASON}_${DATE}.json"
echo "Checking database health - $CURRENT_SEASON season..."
python -m src.database_evaluator \
    --season=$CURRENT_SEASON \
    --output="${REPORT_FILE}"

EXIT_CODE=$?

# Optional: Run on all seasons (slower, run weekly)
# Uncomment for weekly comprehensive check
# if [ $(date +%u) -eq 1 ]; then  # Monday only
#     echo "Weekly comprehensive check: 2023-2024 and 2024-2025..."
#     python -m src.database_evaluator \
#         --season=2023-2024 \
#         --output="${REPORT_DIR}/health_check_2023-2024_${DATE}.json"
# fi

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Health check completed: ${DATE}"
else
    echo "❌ Health check failed: ${DATE}"
    # Send alert (email, Slack, etc.)
    # Example: python -m src.utils.send_alert "Database health check failed"
fi

# Cleanup old reports (keep last 30 days)
find $REPORT_DIR -name "*.json" -mtime +30 -delete

echo "Report saved to: ${REPORT_DIR}/dev_${CURRENT_SEASON}_${DATE}.json"
