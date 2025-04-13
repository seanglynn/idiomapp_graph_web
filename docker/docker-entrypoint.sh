#!/bin/bash
set -e

# Create Streamlit credentials file to skip onboarding
mkdir -p /root/.streamlit
echo '[general]
email = ""' > /root/.streamlit/credentials.toml

# Create directories if they don't exist
mkdir -p /app/logs

# Log startup information
echo "Starting IdiomApp Graph Explorer in Docker"
echo "- Streamlit server will be available at: http://localhost:${STREAMLIT_SERVER_PORT:-8503}"
echo "- Server address binding: ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
echo "- Ollama host: ${OLLAMA_HOST:-http://ollama:11434}"
echo "- Default LLM model: ${DEFAULT_MODEL:-llama3.2:latest}"
echo "- Log level: ${LOG_LEVEL:-INFO}"

# Execute the command provided (or default to streamlit run)
if [ "$#" -eq 0 ]; then
  exec poetry run streamlit run idiomapp/streamlit/app.py \
    --server.headless=true \
    --server.port=${STREAMLIT_SERVER_PORT:-8503} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --browser.gatherUsageStats=false
else
  exec "$@"
fi 