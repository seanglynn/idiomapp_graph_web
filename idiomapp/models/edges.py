"""Edge payloads: what a click on a graph edge/connection means."""

from typing import Optional

from pydantic import BaseModel, ValidationError

from idiomapp.utils.logging_utils import get_logger

logger = get_logger("models.edges")


class EdgePayload(BaseModel):
    """What a click on an edge/connection means - always just informational
    (no edge triggers an action), so unlike nodes every edge kind shares one
    shape rather than needing its own class."""

    kind: str
    label: str = ""
    description: str = ""
    strength: Optional[float] = None
    weight: Optional[int] = None


def parse_edge_payload(raw: dict) -> Optional[EdgePayload]:
    """Reconstruct a typed EdgePayload from the plain dict a browser click
    hands back, or None (logged as a warning, for the same reason as
    `nodes.parse_node_payload`) if it doesn't look like a valid edge
    payload."""
    try:
        return EdgePayload.model_validate(raw)
    except ValidationError as exc:
        logger.warning(f"Could not parse a clicked edge's data: {exc}")
        return None
