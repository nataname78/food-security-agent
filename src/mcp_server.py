"""
mcp_server.py — Custom MCP server for the Food Security Early Warning agent.

Exposes 3 tools over MCP (stdio transport):
    1. search_food_security_reports — hybrid search over the local corpus
    2. get_climate_indicator       — fetch a public rainfall/NDVI-proxy indicator for a region
    3. cross_reference_claim        — check how many independent corpus documents support a claim

Run standalone for local testing:
    python src/mcp_server.py

Run under the MCP inspector:
    npx @modelcontextprotocol/inspector python src/mcp_server.py
"""

from __future__ import annotations

import os
import sys
import warnings

# Must be set before sentence-transformers / huggingface_hub / transformers are
# imported anywhere in the process, and before the stdio MCP channel starts:
# any stray progress bar / warning printed to stdout while the MCP JSON-RPC
# protocol is live on stdout will corrupt the channel and hang the client
# indefinitely waiting for a well-formed response.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from retrieval import HybridRetriever

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")

mcp = FastMCP("food-security-early-warning")

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(CORPUS_DIR, use_dense=True, use_reranker=True)
    return _retriever


@mcp.tool()
def search_food_security_reports(query: str, top_k: int = 5) -> str:
    """Search the food-security report corpus for passages relevant to a query.

    Use when: you need evidence about a country/region's food security situation,
    IPC phase, drivers of insecurity, or early-warning methodology, before making
    a claim in your analysis.
    Do NOT use when: the query is about something outside food security / crop
    monitoring (e.g. general chit-chat, unrelated topics) — the tool will return
    irrelevant results and waste a retrieval call.
    Returns: a string with up to `top_k` ranked passages, each prefixed with its
    source filename, retrieved via hybrid (BM25 + dense) search with cross-encoder
    reranking.
    Example: search_food_security_reports("early warning indicators for crop failure")
    -> "[Source: methodology_early_warning_indicators.md] Early warning systems ..."
    """
    try:
        retriever = get_retriever()
        top_k = max(1, min(int(top_k), 10))
        results = retriever.search(query, top_k=top_k)
        if not results:
            return "No relevant passages found in the corpus for this query."
        return retriever.get_parent_context(results)
    except Exception as e:  # noqa: BLE001 — MCP tools must never raise uncaught exceptions
        return f"ERROR: search_food_security_reports failed ({type(e).__name__}: {e})"


@mcp.tool()
def get_climate_indicator(latitude: float, longitude: float, days_back: int = 30) -> str:
    """Fetch recent precipitation and temperature data for a location from NASA POWER
    (a free, public, no-API-key satellite/reanalysis climate dataset).

    Use when: you need an objective, current climate signal (rainfall, temperature)
    for a specific location to corroborate or contradict a food-security report claim.
    Do NOT use when: latitude/longitude are unknown or out of range ([-90,90], [-180,180]) —
    validate the location first.
    Returns: a short text summary of average daily precipitation (mm) and temperature (°C)
    over the requested window, or an ERROR string if the API is unreachable.
    Example: get_climate_indicator(13.5, 2.1, days_back=30)  # Niamey, Niger area
    -> "Avg precipitation: 2.1 mm/day, Avg temp: 31.4C over last 30 days"
    """
    try:
        import datetime

        import requests

        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return "ERROR: latitude/longitude out of valid range."
        days_back = max(1, min(int(days_back), 90))
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days_back)
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=PRECTOTCORR,T2M&community=AG"
            f"&longitude={longitude}&latitude={latitude}"
            f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]
        precip_vals = [v for v in data.get("PRECTOTCORR", {}).values() if v is not None and v > -900]
        temp_vals = [v for v in data.get("T2M", {}).values() if v is not None and v > -900]
        if not precip_vals or not temp_vals:
            return "ERROR: no valid climate data returned for this location/period."
        avg_precip = sum(precip_vals) / len(precip_vals)
        avg_temp = sum(temp_vals) / len(temp_vals)
        return (
            f"Avg precipitation: {avg_precip:.2f} mm/day, "
            f"Avg temp: {avg_temp:.1f}C over last {days_back} days "
            f"(lat={latitude}, lon={longitude}, source=NASA POWER)"
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR: get_climate_indicator failed ({type(e).__name__}: {e})"


@mcp.tool()
def cross_reference_claim(claim: str, top_k: int = 5) -> str:
    """Check how many distinct corpus documents contain evidence related to a claim,
    to support a confidence assessment (used by the critic agent and CONFIDENCE tagging).

    Use when: you are about to assert something in the final synthesis and want to
    know if it is corroborated by more than one independent source in the corpus.
    Do NOT use when: the claim is a pure logical/arithmetic deduction with no
    factual content to verify against the corpus.
    Returns: a string reporting the number of distinct source documents found and
    which ones, so the caller can gauge corroboration strength.
    Example: cross_reference_claim("El Nino is affecting Ethiopia's kiremt rains")
    -> "2 distinct sources found: ethiopia_2026_elnino.md, methodology_early_warning_indicators.md"
    """
    try:
        retriever = get_retriever()
        top_k = max(1, min(int(top_k), 10))
        results = retriever.search(claim, top_k=top_k)
        distinct_sources = sorted({r.chunk.source for r in results})
        if not distinct_sources:
            return "0 distinct sources found — this claim is not corroborated by the corpus."
        return f"{len(distinct_sources)} distinct source(s) found: {', '.join(distinct_sources)}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: cross_reference_claim failed ({type(e).__name__}: {e})"


if __name__ == "__main__":
    # Warm up the retriever (downloads/loads embedding + cross-encoder models)
    # BEFORE the stdio transport takes over stdout for the MCP JSON-RPC
    # protocol. Doing this lazily on the first tool call instead has caused
    # the client to hang on some setups, because any stray output during
    # model download corrupts the stdio channel's message framing.
    print("[mcp_server] warming up retriever (first run downloads models, may take a minute)...", file=sys.stderr)
    get_retriever()
    print("[mcp_server] ready.", file=sys.stderr)
    mcp.run(transport="stdio")