"""Adapters that turn this app's graph data into ECharts graph-series options.

Pure Python — no `streamlit` or `streamlit_echarts` import here, so every
function is directly unit-testable. Each of the three graphs in app.py owns
its own small `_style_*` helpers (this is deliberately where the "preserve
current visual semantics" logic lives, using the same GROUP_COLORS /
POS_BORDER_COLORS / RELATION_COLORS constants the old pyvis code used), then
funnels through the one shared, purely-mechanical `build_graph_echarts_options`.

Every node/edge carries its complete original source dict under "raw", so a
click round-trips full data (word, language, POS, ...), not just an id.
"""

import re
from typing import Any, Optional

from idiomapp.config import (
    GROUP_COLORS,
    LANGUAGE_MAP,
    POS_BORDER_COLORS,
    RELATION_COLORS,
)
from idiomapp.utils.nlp_utils import get_language_color

# Graph category config for the per-word knowledge graph: (data_key, label, color)
GRAPH_CATEGORIES = [
    ("cognates", "🌍 Cognates", "#E91E63"),
    ("synonyms", "≈ Synonyms", "#3498DB"),
    ("antonyms", "≠ Antonyms", "#E74C3C"),
    ("idioms", "🎭 Idioms", "#F39C12"),
    ("collocations", "🔗 Collocations", "#00BCD4"),
]

# Registered on the frontend via st_echarts(..., events={"click": GRAPH_CLICK_JS}).
# `params.data` is whatever dict build_graph_echarts_options put in "data"/"links"
# for that item, including our "raw" field - ECharts passes custom keys through
# untouched, confirmed empirically (a live click round-tripped `raw` intact).
GRAPH_CLICK_JS = (
    "function(params) { return {dataType: params.dataType, ...params.data}; }"
)


def sanitize_tooltip_text(text: str) -> str:
    """Strip HTML tags from tooltip text and escape any remaining angle brackets."""
    text = re.sub(r"<[^>]*>", "", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def format_entry(item: Any) -> str:
    """Render one canonical {term, gloss} entry (or a plain value) for display."""
    if isinstance(item, dict):
        term = item.get("term", "")
        gloss = item.get("gloss")
        return f"{term}: {gloss}" if gloss else str(term)
    return str(item)


def format_entries(data: dict, key: str, *, limit: int = 8) -> list:
    """
    Render data[key] as a list of display strings, truncated to *limit*.

    Analysis data reaches this point already canonicalised by
    WordAnalysis.to_display_dict(): every Entries field is a list of
    {"term", "gloss"} dicts and every StrList field is a flat list of strings -
    never a bare mapping.
    """
    items = data.get(key) or []
    return [format_entry(item) for item in items[:limit]]


def adjust_color(hex_color: str, amount: int) -> str:
    """Lighten (positive amount) or darken (negative) a hex color."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    adjusted = [max(0, min(255, channel + amount)) for channel in rgb]
    return "#%02x%02x%02x" % tuple(adjusted)


def _ensure_sentence_group_colors() -> None:
    """Memoize per-sentence GROUP_COLORS variants onto the shared config dict.

    Mirrors the old inline behavior: a one-time computation cached on the
    module-level constant itself rather than recomputed on every render.
    """
    if "en-s9" in GROUP_COLORS:
        return
    for i in range(1, 10):
        suffix = f"-s{i}"
        for base in ("en", "es", "ca", "en-related", "es-related", "ca-related"):
            GROUP_COLORS[f"{base}{suffix}"] = adjust_color(GROUP_COLORS[base], i * 10)


def build_graph_echarts_options(graph_data: dict, *, layout: str = "force") -> dict:
    """Turn a styled {"nodes": [...], "edges": [...]} dict into a full ECharts
    `option` payload for a graph series.

    Purely mechanical: reads each item's precomputed style fields (symbolSize,
    symbol, itemStyle, tooltip, style) - no per-source color/business logic
    lives here, that's each call site's own `_style_*` helpers.
    """
    nodes = [
        {
            "name": n["id"],
            "value": n.get("label", n["id"]),
            "symbolSize": n.get("symbolSize", 30),
            "symbol": n.get("symbol", "circle"),
            "itemStyle": n.get("itemStyle", {}),
            "label": {"show": True, "formatter": n.get("label", n["id"])},
            "tooltip": {"formatter": n.get("tooltip", "")},
            "raw": n.get("raw", n),
        }
        for n in graph_data.get("nodes", [])
    ]
    edges = [
        {
            "source": e["from"],
            "target": e["to"],
            "lineStyle": e.get("style", {}),
            "tooltip": {"formatter": e.get("tooltip", "")},
            "raw": e.get("raw", e),
        }
        for e in graph_data.get("edges", [])
    ]
    return {
        "tooltip": {},
        "series": [
            {
                "type": "graph",
                "layout": layout,
                "roam": True,
                "draggable": True,
                "label": {"show": True, "position": "right"},
                "data": nodes,
                "links": edges,
                "force": {"repulsion": 250, "edgeLength": 90},
                "emphasis": {"focus": "adjacency"},
            }
        ],
    }


def resolve_graph_click(chart_event: Optional[dict]) -> Optional[dict]:
    """Turn the raw dict GRAPH_CLICK_JS returns into {"kind", "data"}, or None."""
    if not chart_event:
        return None
    kind = chart_event.get("dataType")
    if kind not in ("node", "edge"):
        return None
    raw = chart_event.get("raw")
    if raw is None:
        return None
    return {"kind": kind, "data": raw}


# ---------------------------------------------------------------------------
# Semantic / translation graph
# ---------------------------------------------------------------------------


def filter_invalid_nodes(graph_data: dict) -> dict:
    """Drop error-placeholder nodes (and any edge referencing one) from a
    translation graph. Returns a new dict; does not mutate the input."""
    error_keywords = ("translation", "failed", "error", "try", "again")
    valid_nodes = [
        n
        for n in graph_data.get("nodes", [])
        if not any(kw in n.get("label", "").lower() for kw in error_keywords)
    ]
    valid_ids = {n["id"] for n in valid_nodes}
    valid_edges = [
        e
        for e in graph_data.get("edges", [])
        if e.get("from") in valid_ids and e.get("to") in valid_ids
    ]
    return {**graph_data, "nodes": valid_nodes, "edges": valid_edges}


def _style_semantic_node(node: dict) -> dict:
    group = node.get("group", "default")
    color = GROUP_COLORS.get(group, "#4CC9F0")
    pos = node.get("pos", "unknown")
    border_color = POS_BORDER_COLORS.get(pos.lower(), "#4361EE")
    size = 30 if node.get("node_type") == "primary" else 20
    lang_code = node.get("language", "unknown")
    lang_name = LANGUAGE_MAP.get(lang_code, {}).get("name", lang_code.upper())
    details = sanitize_tooltip_text(node.get("details", ""))
    tooltip = f"{node['label']} ({lang_name}); Part of speech: {pos}"
    if details:
        tooltip += f"; Details: {details}"
    return {
        "id": node["id"],
        "label": node["label"],
        "symbolSize": size,
        "symbol": "circle",
        "itemStyle": {"color": color, "borderColor": border_color, "borderWidth": 2},
        "tooltip": tooltip,
        "raw": node,
    }


def _style_semantic_edge(edge: dict) -> dict:
    relation = edge.get("relation", "related")
    strength = edge.get("strength", 0.5)

    if relation in ("direct_translation", "translation"):
        width, color, dashed = 3 * strength, "#FFFFFF", False
    elif relation == "cognate":
        width, color, dashed = 2.5 * strength, "#FFD700", False
    elif relation == "semantic_equivalent":
        width, color, dashed = 2.5 * strength, "#00B8D4", False
    elif relation == "cross_sentence":
        width, color, dashed = 1.5 * strength, edge.get("color", "#AA44BB"), True
    elif "semantic_similarity" in relation:
        width, color, dashed = 2 * strength, edge.get("color", "#FFAA00"), True
    else:
        width, color, dashed = (
            1 + strength,
            RELATION_COLORS.get(relation, "#AAAAAA"),
            False,
        )

    # Matches the original pyvis rule exactly: an edge that merely HAS a
    # "dashes" key renders dashed regardless of that key's actual value, same
    # as relation being exactly cross_sentence/semantic_similarity.
    if relation in ("cross_sentence", "semantic_similarity") or "dashes" in edge:
        dashed = True

    label = sanitize_tooltip_text(edge.get("label", relation.replace("_", " ")))
    if "title" in edge:
        tooltip = sanitize_tooltip_text(edge["title"])
    elif "description" in edge:
        tooltip = sanitize_tooltip_text(edge["description"])
    else:
        tooltip = f"{label} ({strength:.2f})"

    return {
        "from": edge["from"],
        "to": edge["to"],
        "style": {
            "color": color,
            "width": width,
            "type": "dashed" if dashed else "solid",
            "curveness": 0.15,
        },
        "tooltip": tooltip,
        "raw": edge,
    }


def semantic_graph_to_echarts_data(graph_data: dict) -> dict:
    """Adapt a (pre-filtered) translation-graph {"nodes","edges"} dict into the
    normalized shape build_graph_echarts_options expects."""
    _ensure_sentence_group_colors()
    return {
        "nodes": [_style_semantic_node(n) for n in graph_data.get("nodes", [])],
        "edges": [_style_semantic_edge(e) for e in graph_data.get("edges", [])],
    }


# ---------------------------------------------------------------------------
# Co-occurrence network
# ---------------------------------------------------------------------------


def cooccurrence_graph_to_echarts_data(graph, lang_code: Optional[str] = None) -> dict:
    """Adapt a networkx co-occurrence graph into the normalized shape
    build_graph_echarts_options expects."""
    color = get_language_color(lang_code) if lang_code else "#4CC9F0"
    nodes = []
    for node in graph.nodes():
        node_id = str(node)
        degree = graph.degree(node)
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "symbolSize": 20 + degree * 3,
                "symbol": "circle",
                "itemStyle": {
                    "color": color,
                    "borderColor": "#4361EE",
                    "borderWidth": 2,
                },
                "tooltip": f"Word: {sanitize_tooltip_text(node_id)}; Co-occurrences: {degree}",
                "raw": {
                    "kind": "cooccurrence_word",
                    "word": node_id,
                    "language": lang_code,
                    "degree": degree,
                },
            }
        )
    edges = []
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        edges.append(
            {
                "from": str(source),
                "to": str(target),
                "style": {
                    "color": "#FFFFFF" if weight > 2 else "#AAAAAA",
                    "width": 1 + weight / 2,
                    "type": "solid",
                    "curveness": 0.1,
                },
                "tooltip": f"Co-occurrence: {weight}",
                "raw": {
                    "kind": "cooccurrence_edge",
                    "source": str(source),
                    "target": str(target),
                    "weight": weight,
                },
            }
        )
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Per-word knowledge graph
# ---------------------------------------------------------------------------


def word_analysis_to_echarts_data(
    word: str, language: str, analysis_data: dict
) -> dict:
    """Rebuild the word-analysis hub-and-spoke knowledge graph as node/edge
    data: one central "main" node, then a hub per non-empty category (etymology,
    the standard GRAPH_CATEGORIES, conjugations, examples, forms) with up to 8
    leaf nodes each. Mirrors the old direct net.add_node/add_edge structure
    exactly, as real data instead of imperative graph-library calls.
    """
    nodes: list = []
    edges: list = []
    counter = 0

    pos = analysis_data.get("pos", "UNKNOWN")
    nodes.append(
        {
            "id": "main",
            "label": word,
            "symbolSize": 50,
            "symbol": "circle",
            "itemStyle": {"color": "#FF6B6B"},
            "tooltip": f"{word} · {pos} · {language}",
            "raw": {"kind": "main", "word": word, "language": language, "pos": pos},
        }
    )

    def add_category(cat_id: str, label: str, color: str, items: list) -> None:
        nonlocal counter
        if not items:
            return
        hub_id = f"cat_{cat_id}"
        nodes.append(
            {
                "id": hub_id,
                "label": label,
                "symbolSize": 30,
                "symbol": "diamond",
                "itemStyle": {"color": color},
                "tooltip": label,
                "raw": {"kind": "category", "category": cat_id, "label": label},
            }
        )
        edges.append(
            {
                "from": "main",
                "to": hub_id,
                "style": {
                    "color": color,
                    "width": 3,
                    "type": "solid",
                    "curveness": 0.1,
                },
                "tooltip": label,
                "raw": {"kind": "category_edge", "category": cat_id},
            }
        )
        for item in items[:8]:
            counter += 1
            item_id = f"item_{counter}"
            display = item[:27] + "..." if len(item) > 30 else item
            nodes.append(
                {
                    "id": item_id,
                    "label": display,
                    "symbolSize": 20,
                    "symbol": "circle",
                    "itemStyle": {"color": color},
                    "tooltip": item,
                    "raw": {
                        "kind": "item",
                        "category": cat_id,
                        "text": item,
                        "word": item,
                        "language": language,
                    },
                }
            )
            edges.append(
                {
                    "from": hub_id,
                    "to": item_id,
                    "style": {
                        "color": color,
                        "width": 1,
                        "type": "solid",
                        "curveness": 0.1,
                    },
                    "tooltip": item,
                    "raw": {"kind": "item_edge", "category": cat_id},
                }
            )

    etym_color = "#9B59B6"
    if "etymology" in analysis_data:
        nodes.append(
            {
                "id": "etymology",
                "label": "📜 Etymology",
                "symbolSize": 30,
                "symbol": "diamond",
                "itemStyle": {"color": etym_color},
                "tooltip": analysis_data["etymology"],
                "raw": {"kind": "etymology", "text": analysis_data["etymology"]},
            }
        )
        edges.append(
            {
                "from": "main",
                "to": "etymology",
                "style": {
                    "color": etym_color,
                    "width": 3,
                    "type": "solid",
                    "curveness": 0.1,
                },
                "tooltip": "Etymology",
                "raw": {"kind": "etymology_edge"},
            }
        )
        for sub_key, prefix in (("language_origin", "Origin"), ("root", "Root")):
            if sub_key in analysis_data:
                counter += 1
                label = (
                    f"{prefix}: {analysis_data[sub_key]}"
                    if sub_key == "root"
                    else analysis_data[sub_key]
                )
                sub_id = f"{sub_key}_{counter}"
                nodes.append(
                    {
                        "id": sub_id,
                        "label": label,
                        "symbolSize": 20,
                        "symbol": "circle",
                        "itemStyle": {"color": etym_color},
                        "tooltip": f"{prefix}: {analysis_data[sub_key]}",
                        "raw": {
                            "kind": "etymology_detail",
                            "field": sub_key,
                            "text": str(analysis_data[sub_key]),
                        },
                    }
                )
                edges.append(
                    {
                        "from": "etymology",
                        "to": sub_id,
                        "style": {
                            "color": etym_color,
                            "width": 1,
                            "type": "solid",
                            "curveness": 0.1,
                        },
                        "tooltip": prefix,
                        "raw": {"kind": "etymology_detail_edge"},
                    }
                )

    for key, label, color in GRAPH_CATEGORIES:
        add_category(key, label, color, format_entries(analysis_data, key))

    conj = format_entries(analysis_data, "conjugations", limit=6)
    if conj:
        add_category("conjugations", "📝 Conjugations", "#2ECC71", conj)

    examples = analysis_data.get("examples", [])
    if isinstance(examples, list) and examples:
        short = [ex[:40] + "..." if len(ex) > 40 else ex for ex in examples[:4]]
        add_category("examples", "💬 Examples", "#1ABC9C", short)

    forms = []
    for fkey, flabel in (
        ("plural", "Plural"),
        ("gender", "Gender"),
        ("infinitive", "Infinitive"),
        ("verb_type", "Verb type"),
    ):
        val = analysis_data.get(fkey)
        if val:
            forms.append(f"{flabel}: {val}")
    for entries_key in ("gender_forms", "related_forms"):
        forms.extend(format_entries(analysis_data, entries_key))
    if forms:
        add_category("forms", "📋 Forms", "#4CAF50", forms)

    return {"nodes": nodes, "edges": edges}
