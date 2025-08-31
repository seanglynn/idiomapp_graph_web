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
echo "- Log level: ${LOG_LEVEL:-INFO}"
echo "- LLM Provider: ${LLM_PROVIDER:-ollama}"
echo "- Package Manager: uv"

# Provider-specific information
if [ "${LLM_PROVIDER:-ollama}" = "ollama" ]; then
  echo "- Ollama host: ${OLLAMA_HOST:-http://ollama:11434}"
  echo "- Default LLM model: ${DEFAULT_MODEL:-llama3.2:latest}"
elif [ "${LLM_PROVIDER:-ollama}" = "openai" ]; then
  echo "- OpenAI model: ${OPENAI_MODEL:-gpt-3.5-turbo}"
  if [ -z "${OPENAI_API_KEY}" ]; then
    echo "- WARNING: OPENAI_API_KEY is not set. You will need to provide it in the UI."
  else
    echo "- OpenAI API key: [CONFIGURED]"
  fi
  if [ -n "${OPENAI_ORGANIZATION}" ]; then
    echo "- OpenAI organization: ${OPENAI_ORGANIZATION}"
  fi
fi

# Execute the command provided (or default to streamlit run)
if [ "$#" -eq 0 ]; then
  exec streamlit run idiomapp/streamlit/app.py \
    --server.headless=${STREAMLIT_SERVER_HEADLESS:-true} \
    --server.port=${STREAMLIT_SERVER_PORT:-8503} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --browser.gatherUsageStats=false
else
  exec "$@"
fi 