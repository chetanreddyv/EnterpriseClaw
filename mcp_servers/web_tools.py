"""
mcp_servers/web_tools.py — Web Search and Fetch tools.

This plugin provides the agent with the ability to search the web and
fetch specific webpage contents, returning clean, LLM-optimized markdown.
"""

import logging
import time
from typing import Dict, Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import trafilatura
from markdownify import markdownify as md

logger = logging.getLogger("mcp.web_tools")

# ══════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════

def _smart_truncate(text: str, max_chars: int = 15000) -> str:
    """Truncates text at the nearest paragraph/newline to avoid mid-word chops."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_newline = truncated.rfind('\n')
    
    # If a newline exists in the last 20% of the truncated text, cut there
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
        
    return truncated + "\n\n... [Content Truncated due to length limit]"

# ══════════════════════════════════════════════════════════════
# Tool Implementations
# ══════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.
    
    Args:
        query: The search query.
        max_results: The maximum number of results to return (default: 5).
        
    Returns:
        A string containing a markdown-formatted list of search results.
    """
    logger.info(f"🛠️ web_search(query='{query}', max_results={max_results})")
    
    max_retries = 3
    results = []
    
    # Exponential backoff for DDGS rate limits
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                # Consume the iterator safely
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            break # Break loop if successful
        except Exception as e:
            logger.warning(f"DDGS attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return f"Error executing search: Search engine rate-limited or failed after {max_retries} attempts. Try fetching a known URL instead."
            time.sleep(2 ** attempt) # Wait 1s, then 2s
            
    if not results:
        return f"No results found for query: '{query}'. Try a broader search."
        
    # Format beautifully for the LLM
    lines = [f"## Search Results for '{query}'\n"]
    for i, res in enumerate(results, start=1):
        title = res.get("title", "No Title")
        href = res.get("href", "No URL")
        body = res.get("body", "No Snippet")
        
        # Ensure snippet isn't excessively long
        if len(body) > 400:
            body = body[:397] + "..."
            
        lines.append(f"**{i}. [{title}]({href})**\n> {body}\n")
        
    out_str = "\n".join(lines)
    logger.info(f"✅ web_search returned {len(results)} results")
    return out_str


def web_fetch(url: str) -> str:
    """
    Fetch a URL and return its content as clean, readable Markdown.
    
    Args:
        url: The web page URL to fetch.
        
    Returns:
        A string containing the markdown extracted from the webpage.
    """
    logger.info(f"🛠️ web_fetch(url='{url}')")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        with httpx.Client(headers=headers, timeout=15.0, follow_redirects=True) as client:
            # Stream the response to check headers before downloading giant files
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                
                # Reject non-text/HTML files (like PDFs, zips, media)
                if not content_type.startswith("text/") and "application/xhtml+xml" not in content_type:
                    return f"Failed to fetch {url}: Unsupported content type '{content_type}'. This tool only reads HTML/text."
                
                # Read the actual content
                response.read()
                html_content = response.content

        # 1. Primary Extraction: Try Trafilatura (Best for articles/blogs)
        text_content = trafilatura.extract(
            html_content,
            include_links=True,
            include_formatting=True,
            favor_precision=True
        )

        # 2. Fallback Extraction: BeautifulSoup + Markdownify (Best for non-articles/lists/docs)
        if not text_content:
            logger.info("Trafilatura returned empty; falling back to BeautifulSoup + Markdownify.")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Strip extremely noisy elements
            for tag in soup(["script", "style", "nav", "footer", "aside", "noscript", "iframe"]):
                tag.decompose()
                
            # Convert the cleaned HTML to Markdown
            text_content = md(str(soup), strip=['img', 'a'], heading_style="ATX").strip()

        if not text_content:
             return f"Content of {url} could not be parsed or is empty."

        text_content = _smart_truncate(text_content, max_chars=15000)
        
        logger.info(f"✅ web_fetch extracted {len(text_content)} characters")
        return f"### Content of {url}:\n\n{text_content}"

    except httpx.TimeoutException:
        logger.error(f"❌ web_fetch Timeout: {url}")
        return f"Failed to fetch {url}: Connection timed out after 15 seconds."
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ web_fetch HTTP error: {e.response.status_code}")
        # Provide helpful guidance to the LLM
        if e.response.status_code in (401, 403):
            return f"Failed to fetch {url}: Access Denied (HTTP {e.response.status_code}). The site may be blocking bots."
        if e.response.status_code == 404:
            return f"Failed to fetch {url}: Page Not Found (HTTP 404)."
        return f"Failed to fetch {url}: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error(f"❌ web_fetch error: {str(e)}")
        return f"Failed to fetch {url}: {type(e).__name__} - {str(e)}"

# ══════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}
