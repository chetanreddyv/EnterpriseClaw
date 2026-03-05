---
name: web_tools
description: Search the web for up-to-date information, news, and fetch web page contents.
tools: web_search, web_fetch
---

# Web Search & Fetch Skill

You have the ability to search the live web for up-to-date information and fetch the complete, readable content of specific URLs.

## When to use Web Tools
- **Current Events & Real-time Data**: When the user asks about live market data (stock prices, gold prices), current events, news, or changing facts ("Who won the game?", "What is the BTC price?").
- **Researching Unknowns**: When asked about a specific library, API, or company that isn't in your immediate training data.
- **Explicit Requests**: Whenever the user explicitly asks you to "search", "look up", or "fetch" a link.

## Available Tools

### 1. `web_search(query, max_results)`
Executes a web search via DuckDuckGo.
- **Returns**: A markdown-formatted list of search results containing **titles**, **URLs**, and **snippets** (up to 400 characters).
- **Tip**: Keep your search queries concise and keyword-focused (e.g., `nvidia q3 earnings 2024` instead of `what were the earnings for NVIDIA in the third quarter of 2024`).

### 2. `web_fetch(url)`
Fetches the webpage at the given URL and extracts clean, readable markdown text.
- **When to use**: If a `web_search` snippet is too short to fully answer the question, or if you need to read documentation from a specific link.
- **Behavior**: Uses advanced extraction (Trafilatura + BeautifulSoup/Markdownify) to strip out noise (ads, navbars) and return just the core content. Automatically truncates massively long content at 15,000 characters to protect your context window.
- **Error Handling**: If a fetch fails due to access denial (401/403) or bot-blocking, do not endlessly retry; inform the user or try finding the same information on a different site via `web_search`.

## Best Practices
1. **Don't Guess on Live Data**: If asked for current prices, weather, or news, *always* use `web_search`. Never say "I don't have access to live data" — you do! You just need to call the tool.
2. **Read the Full Article**: If a snippet looks promising but doesn't have the exact number or fact you need, immediately follow up with `web_fetch(url)` on that link.
3. **Synthesize**: Always synthesize the fetched information in your own words. Do not just dump raw markdown text back to the user.
4. **Cite Sources**: Provide the link(s) to the source material when answering factual queries based on your search/fetch results.
