# IdiomApp

[![CI](https://github.com/seanglynn/idiomapp_graph_web/actions/workflows/ci.yml/badge.svg)](https://github.com/seanglynn/idiomapp_graph_web/actions/workflows/ci.yml)

Visualizing linguistic connections through interactive graphs and networks.

**¿Qué es esto?**
- Establecer relaciones entre palabras y frases en varios idiomas.

**¿Por qué?**
- Quiero aprender idioma**s** más rápido.

**¿Cómo?**
- Con LLMs, NLP, semantic graph & co-occurrence networks obvio!

## Project Structure

- `idiomapp/streamlit/`: Main Streamlit application
- `idiomapp/utils/`: LLM integration, NLP, TTS, state, and logging
  - `llm_utils.py`: LLM client abstractions for Ollama, OpenAI, and Anthropic (Claude)
  - `schemas.py`: Typed Pydantic schemas for LLM word-analysis output
  - `json_utils.py`: JSON extraction fallback for LLM responses
  - `nlp_utils.py`: NLP processing and graph generation
  - `state_utils.py`: Session state management and caching
  - `audio_utils.py`: Text-to-speech
  - `logging_utils.py`: Centralized logging config
- `tests/`: pytest suite (backend + Streamlit `AppTest` UI coverage)
- `docker/`: Dockerfile and container entrypoint

## Quick Start

```bash
make install        # uv sync
make run-graph-dev   # streamlit run, with auto-reload, at http://localhost:8503
```

Or with Docker:

```bash
make docker-start    # build + run in the foreground, at http://localhost:8503
make docker-down     # stop
```

Other Docker targets: `make docker-start-detached` (background), `make docker-logs`,
`make docker-shell` / `make ollama-shell` (debugging).

## Configuration

Copy `env.example` to `.env` and adjust — every setting is documented inline there.
The one required choice is `LLM_PROVIDER` (`ollama`, `openai`, or `anthropic`); each
provider needs its own API key/host set below that.

```bash
cp env.example .env
```

## LLM Providers

Switch providers via `LLM_PROVIDER` in `.env` or from the sidebar at runtime.

- **Ollama** (default) — local models like `llama3.2:latest`. Needs Ollama running
  and reachable at `OLLAMA_HOST`.
- **OpenAI** — needs `OPENAI_API_KEY`.
- **Anthropic (Claude)** — needs `ANTHROPIC_API_KEY`. Defaults to `claude-haiku-4-5`
  (fastest/cheapest); switch to `claude-opus-5` for higher-quality analysis. Claude
  models reject `temperature`/`top_p` — response depth is controlled by
  `ANTHROPIC_EFFORT` instead, which only applies to models that support it (see
  `ANTHROPIC_MODEL_CAPABILITIES` in `idiomapp/config.py`).

## Development

```bash
make test     # pytest
make lint     # flake8 + black --check
make format   # black
```

CI runs `test` and `lint` on every push.

## Security Note

Locally, the app binds to `localhost` only. In Docker, it binds to `0.0.0.0` inside
the container, isolated, with only the mapped port (`8503`) reachable from your host.

## Troubleshooting

- **Docker can't reach Ollama** — confirm the Ollama container/service is up and
  `OLLAMA_HOST` points at it (`http://ollama:11434` inside Docker, not `localhost`).
- **Files under `logs/` or `graph_storage/` are owned by `root`** — the Docker image
  runs as a non-root user matching your host UID (default `1000`). If your host user
  has a different UID, rebuild with `docker compose build --build-arg UID=$(id -u)
  --build-arg GID=$(id -g)`.
- For anything else: `make docker-logs`, or `make docker-shell` / `make ollama-shell`
  to poke around inside a container.
