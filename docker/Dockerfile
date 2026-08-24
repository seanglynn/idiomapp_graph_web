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

# Pre-download the spaCy language models. Without this the app downloads them at
# first request - slow on a cold container, and it degrades to a blank model with
# no network.
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm && \
    python -m spacy download ca_core_news_sm

# Run as a non-root user matching the host UID/GID. docker-compose bind-mounts the
# whole repo at .:/app, so whatever UID this container writes as is the UID that
# ends up owning logs/ and graph_storage/ on the host - as root, that leaves files
# the host user can't touch without sudo. Override at build time with
# --build-arg UID=$(id -u) --build-arg GID=$(id -g) if the host user isn't 1000.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser
ENV HOME=/home/appuser

# Create directory for Streamlit configuration
RUN mkdir -p ${HOME}/.streamlit

# Copy Streamlit configuration
COPY .streamlit/config.toml ${HOME}/.streamlit/config.toml

# Copy application code (as a package, so `import idiomapp...` resolves)
COPY idiomapp /app/idiomapp

# Ensure the entrypoint script is executable
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# /app is overlaid by the docker-compose bind mount at runtime, but chown it here
# too so `docker run` without that bind mount still works as this user.
RUN chown -R ${UID}:${GID} /app ${HOME}

# Define build arguments with defaults
ARG PORT=8503
ARG LOG_LEVEL=INFO
ARG LLM_PROVIDER=ollama
ARG OLLAMA_HOST=http://ollama:11434
ARG DEFAULT_MODEL=llama3.2:latest
ARG OPENAI_MODEL=gpt-3.5-turbo
ARG OPENAI_API_KEY=""
ARG OPENAI_ORGANIZATION=""
ARG ANTHROPIC_MODEL=claude-haiku-4-5
ARG ANTHROPIC_API_KEY=""
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
ENV ANTHROPIC_MODEL=${ANTHROPIC_MODEL}
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS}
ENV STREAMLIT_SERVER_ENABLECORS=${STREAMLIT_SERVER_ENABLECORS}
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=${STREAMLIT_SERVER_ENABLEXSRFPROTECTION}
ENV STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS}
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port
EXPOSE ${PORT}

USER appuser

# Entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]