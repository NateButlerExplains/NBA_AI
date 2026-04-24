#!/bin/bash

# Deploy NBA_AI to Google Cloud Run
# Usage: ./deploy-cloudrun.sh

set -e

PROJECT_ID="insight-bet"
SERVICE_NAME="nba-ai"
REGION="us-central1"

echo "🚀 Deploying NBA_AI to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"

# Deploy using gcloud
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --project $PROJECT_ID \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 100

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Get the service URL with:"
echo "  gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)'"
