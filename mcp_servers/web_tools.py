"""
mcp_servers/web_tools.py — Web Search and Fetch tools.

Provides the agent with web search and fetch capabilities, 
returning strictly clean, raw data optimized for LLM context limits.
"""

import logging
import time
from typing import Dict, Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import trafilatura
from langchain_core.tools import tool

from core.text_utils import smart_truncate

logger = logging.getLogger("mcp.web_tools")

# ══════════════════════════════════════════════════════════════
# Tool Implementations
# ══════════════════════════════════════════════════════════════

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.
    
    Args:
        query: The search query.
    """
    
    max_retries = 3
    results = []
    
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            break 
        except Exception as e:
            logger.warning(f"DDGS attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return "Error: Search engine rate-limited or failed."
            time.sleep(2 ** attempt)
            
    if not results:
        return "No results found."
        
    # Return strict, clean key-value text blocks
    lines = []
    for res in results:
        title = res.get("title", "").strip()
        href = res.get("href", "").strip()
        body = res.get("body", "").strip()
        
        lines.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n---")
        
    return "\n".join(lines).strip()


@tool
def web_fetch(url: str) -> str:
    """
    Fetch a URL and return its raw text content.
    
    Args:
        url: The web page URL to fetch.
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        with httpx.Client(headers=headers, timeout=15.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                
                if not content_type.startswith("text/") and "application/xhtml+xml" not in content_type:
                    return f"Error: Unsupported content type '{content_type}'."
                
                response.read()
                html_content = response.content

        # 1. Primary Extraction: Trafilatura (Clean article extraction)
        text_content = trafilatura.extract(
            html_content,
            include_links=False, # Disabled to keep output clean
            include_formatting=False, # Disabled to avoid markdown clutter
            favor_precision=True
        )

        # 2. Fallback Extraction: BeautifulSoup Raw Text
        if not text_content:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Strip noisy structural elements
            for tag in soup(["script", "style", "nav", "footer", "aside", "noscript", "iframe", "header", "menu"]):
                tag.decompose()
                
            # Extract raw text separated by double newlines
            text_content = soup.get_text(separator='\n\n', strip=True)

        if not text_content:
             return "Error: Content could not be parsed or is empty."

        return smart_truncate(text_content, max_chars=15000, suffix="\n\n[TRUNCATED]").strip()

    except httpx.TimeoutException:
        logger.error(f"Timeout: {url}")
        return "Error: Connection timed out."
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        return f"Error: {type(e).__name__}"

# ══════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}