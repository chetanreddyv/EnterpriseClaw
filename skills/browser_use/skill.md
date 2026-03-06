---
name: browser_use
description: Control a real web browser to navigate, interact with, and extract data from websites.
tools: browser_navigate, browser_click, browser_type, browser_screenshot, browser_get_text, browser_execute_js
---

# Browser Use Skill

You can control a **real Chromium browser** to navigate websites, interact with page elements, fill forms, click buttons, and extract content. Each conversation gets its own isolated browser session with separate cookies.

## Available Tools

### Reading Tools (run immediately)

- **`browser_navigate(url)`** — Go to a URL. Returns the page title and visible text.
- **`browser_get_text()`** — Re-read the current page's visible text content.
- **`browser_screenshot()`** — Save a screenshot of the current page to disk.

### Interaction Tools (require human approval)

- **`browser_click(selector)`** — Click an element. Accepts CSS selectors (`#login-btn`) or visible text (`Sign In`).
- **`browser_type(selector, text)`** — Type text into an input field by CSS selector (`input[name=email]`), placeholder, or label.
- **`browser_execute_js(script)`** — Execute arbitrary JavaScript on the page.

## Best Practices

1. **Always navigate first.** Before clicking or typing, use `browser_navigate` to load the page.
2. **Read before you act.** Use `browser_get_text` to understand the page structure and find the right selectors before clicking.
3. **Use CSS selectors when possible.** They are more reliable than text matching. Look for `id`, `name`, or unique class attributes.
4. **Don't chain blind clicks.** After each click, check the page state with `browser_get_text` or `browser_screenshot` to confirm the result.
5. **Handle errors gracefully.** If a selector isn't found, read the page text to find the correct element.
6. **Prefer `browser_navigate` + `browser_get_text` over `web_fetch`** when you need to interact with a page (login, fill forms) — `web_fetch` just does a static HTTP GET.
