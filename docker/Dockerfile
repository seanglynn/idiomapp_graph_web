FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy uv configuration files and README
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies using uv and ensure streamlit is available
RUN uv pip install --system -e . && \
    uv pip install --system streamlit

# Create directory for Streamlit configuration
RUN mkdir -p /root/.streamlit

# Copy Streamlit configuration
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Copy application code
COPY idiomapp .

# Ensure the entrypoint script is executable
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Define build arguments with defaults
ARG PORT=8503
ARG LOG_LEVEL=INFO
ARG LLM_PROVIDER=ollama
ARG OLLAMA_HOST=http://ollama:11434
ARG DEFAULT_MODEL=llama3.2:latest
ARG OPENAI_MODEL=gpt-3.5-turbo
ARG OPENAI_API_KEY=""
ARG OPENAI_ORGANIZATION=""
ARG STREAMLIT_SERVER_HEADLESS=true
ARG STREAMLIT_SERVER_ENABLECORS=false
ARG STREAMLIT_SERVER_ENABLEXSRFPROTECTION=true
ARG STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set environment variables from build arguments
ENV PYTHONPATH=/app
ENV PATH="/usr/local/bin:$PATH"
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV LLM_PROVIDER=${LLM_PROVIDER}
ENV OLLAMA_HOST=${OLLAMA_HOST}
ENV DEFAULT_MODEL=${DEFAULT_MODEL}
ENV OPENAI_MODEL=${OPENAI_MODEL}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION}
ENV STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS}
ENV STREAMLIT_SERVER_ENABLECORS=${STREAMLIT_SERVER_ENABLECORS}
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=${STREAMLIT_SERVER_ENABLEXSRFPROTECTION}
ENV STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS}
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port
EXPOSE ${PORT}

# Entrypoint 
ENTRYPOINT ["/docker-entrypoint.sh"] 