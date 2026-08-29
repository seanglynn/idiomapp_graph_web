"""
Tests for idiomapp/utils/async_utils.py.

`AppTest`-based, not plain pytest functions: `get_event_loop`/`run_async` read
and write `st.session_state`, which behaves like a real per-session dict only
inside an actual (or AppTest-simulated) Streamlit script run.
"""

from streamlit.testing.v1 import AppTest

_BASELINE_SCRIPT = """
import streamlit as st
from idiomapp.utils.async_utils import run_async

async def fast():
    return "fast-done"

st.session_state["result"] = run_async(fast())
"""


def test_run_async_returns_the_coroutines_result():
    at = AppTest.from_string(_BASELINE_SCRIPT)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["result"] == "fast-done"


# Streamlit starts a new rerun - and a new ScriptRunner thread - as soon as a new
# interaction comes in, without waiting for a still-in-flight rerun's blocking
# calls to return first. This reproduces exactly that: a background thread holds
# this session's event loop genuinely running (via its own run_until_complete),
# standing in for a stale rerun's still-in-flight async call, while the "current"
# script thread calls run_async for a second, unrelated coroutine on the same
# loop. Before the fix, this raised "This event loop is already running" and
# silently dropped the second coroutine instead of running it.
_RACE_SCRIPT = """
import asyncio
import threading
import time

import streamlit as st
from idiomapp.utils.async_utils import run_async, get_event_loop

loop = get_event_loop()

async def slow():
    await asyncio.sleep(0.3)
    return "slow-done"

async def fast():
    return "fast-done"

def run_slow_on_loop():
    loop.run_until_complete(slow())

stale_rerun_thread = threading.Thread(target=run_slow_on_loop)
stale_rerun_thread.start()
time.sleep(0.05)  # let the background thread actually start running the loop

st.session_state["fast_result"] = run_async(fast())
stale_rerun_thread.join()
"""


def test_run_async_survives_an_already_running_loop_from_a_stale_rerun():
    at = AppTest.from_string(_RACE_SCRIPT)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["fast_result"] == "fast-done"
