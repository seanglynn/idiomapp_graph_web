"""
Async execution helpers for the Streamlit app.

Streamlit is synchronous: every user interaction re-runs the script top to bottom,
each time on a *new* ScriptRunner thread. Our LLM clients are async, so somewhere a
coroutine has to be driven to completion.

The naive approach - `asyncio.new_event_loop()` ... `loop.close()` around every
action - is what this module replaces. Closing the loop after each action defeats
connection pooling: an httpx-backed SDK client (AsyncAnthropic, AsyncOpenAI) binds
its connection pool to the loop that created it, so a client cached across reruns
would be left holding a pool attached to a dead loop.

Instead we keep one loop alive per Streamlit session and reuse it. That is safe
because only one script run is ever active per session, so the loop is never driven
concurrently - and it lets pooled TLS connections survive reruns.
"""

import asyncio
from typing import Any, Awaitable, TypeVar

import streamlit as st

from idiomapp.utils.logging_utils import get_logger

logger = get_logger("async_utils")

T = TypeVar("T")

# Session-state key holding the per-session event loop.
_LOOP_KEY = "_idiomapp_event_loop"


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get (or create) this Streamlit session's long-lived event loop.

    Deliberately NOT `@st.cache_resource`: cached resources are shared across every
    session and thread, and two sessions calling `run_until_complete` on one loop
    raises "This event loop is already running". Session state is per-session, which
    is exactly the isolation we need.
    """
    loop = st.session_state.get(_LOOP_KEY)

    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state[_LOOP_KEY] = loop
        logger.debug("Created a new event loop for this session")

    # Each Streamlit rerun executes on a different thread, so the loop must be
    # re-registered as that thread's current loop before it can be used.
    asyncio.set_event_loop(loop)
    return loop


def run_async(coro: Awaitable[T]) -> T:
    """
    Run a coroutine to completion on this session's event loop.

    Args:
        coro: The awaitable to run.

    Returns:
        Whatever the coroutine returns.
    """
    return get_event_loop().run_until_complete(coro)


def loop_key() -> Any:
    """
    Identify the running event loop, for keying loop-bound resources.

    SDK clients hold connection pools tied to one loop. Callers memoise a client
    against this key and rebuild it if the key changes, so a client can never be
    used from a loop it was not created on.
    """
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        # Not inside a coroutine - fall back to the session loop.
        return id(get_event_loop())
