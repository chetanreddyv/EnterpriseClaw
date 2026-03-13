---
name: browser_use
description: Control a real web browser to navigate, interact with, and extract data from websites.
tools: browser_navigate, browser_click, browser_type, browser_screenshot, browser_get_text, browser_execute_js, browser_go_back, browser_scroll, browser_wait_for, browser_snapshot, browser_tab_management, browser_select_option, browser_press_key, browser_hover, browser_handle_dialog, browser_file_upload
---

### TRIGGER_EXAMPLES
- "open a browser"
- "navigate to google.com"
- "click the sign in button on the page"
- "fill out the search box with 'EnterpriseClaw'"
- "take a screenshot of the current page"
- "get a snapshot of the page elements"
- "scroll down on the page"
- "type 'hello' into the search bar and press enter"
### END_TRIGGER_EXAMPLES

# Browser Use Skill

You can control a **real Chromium browser** to navigate websites, interact with page elements, fill forms, click buttons, select dropdowns, upload files, and extract content. Each conversation gets its own isolated browser session with separate cookies.

## Available Tools

### Reading Tools (run immediately)

- **`browser_navigate(url)`** — Go to a URL. Returns the page title and visible text.
- **`browser_get_text()`** — Re-read the current page's visible text content.
- **`browser_screenshot()`** — Take a screenshot and return it as a visual image you can see.
- **`browser_snapshot()`** — Get the accessibility tree (structured elements with roles and names). Better than raw text for finding interactive elements.
- **`browser_go_back()`** — Go back to the previous page in history.
- **`browser_scroll(direction, amount)`** — Scroll up/down to see more content.
- **`browser_wait_for(seconds, text)`** — Wait for a time delay or for specific text to appear (useful for AJAX/SPA content).
- **`browser_tab_management(action, index)`** — List, open, switch, or close browser tabs.

### Interaction Tools 

- **`browser_click(selector, double_click)`** — Click an element. Accepts CSS selectors (`#login-btn`), visible text (`Sign In`), or role+name.
- **`browser_type(selector, text, submit)`** — Type text into an input field. Set `submit=True` to press Enter after typing.
- **`browser_select_option(selector, value)`** — Select an option from a `<select>` dropdown.
- **`browser_press_key(key)`** — Press a keyboard key (`Enter`, `Escape`, `Tab`, `ArrowDown`, `Control+a`).
- **`browser_hover(selector)`** — Hover over an element to trigger dropdown menus or tooltips.
- **`browser_execute_js(script)`** — Execute arbitrary JavaScript on the page.
- **`browser_handle_dialog(action, prompt_text)`** — Control how alert/confirm/prompt dialogs are handled.
- **`browser_file_upload(selector, file_paths)`** — Upload files to a file input element.

## Best Practices

1. **Always navigate first.** Before any interaction, use `browser_navigate` to load the page.
2. **Use `browser_snapshot` to find elements.** The accessibility tree shows you all interactive elements with their roles — much better than guessing CSS selectors.
3. **Use `browser_get_text` for reading content.** It gives you the full visible text on the page.
4. **Use `browser_screenshot` when you need to visually inspect** page layout, images, CAPTCHAs, or verify visual state.
5. **Use `submit=True` on `browser_type`** when filling search boxes or single-field forms instead of a separate `browser_press_key("Enter")`.
6. **Use `browser_select_option` for dropdowns** instead of trying to click through `<select>` options.
7. **Handle infinite scroll with `browser_scroll`** — scroll down, then `browser_get_text` to see the new content.
8. **Use `browser_go_back` if you navigate to a wrong page** — don't re-navigate from scratch.
9. **Wait for dynamic content with `browser_wait_for`** — SPAs and AJAX pages need time to load.
10. **Use `browser_tab_management` for multi-page workflows** — open links in new tabs when you need to reference multiple pages.

## ⚠️ CRITICAL: Autonomy & Resilience Rules

You are an **autonomous browser agent**. You MUST follow these rules during ANY browsing task:

### Never Give Up After One Failure
- If a URL returns 404 or an error, **DO NOT stop**. Navigate to the site's homepage and find the correct link manually.
- If a selector doesn't match, use `browser_snapshot()` to discover the actual element names, then retry.
- If a page loads strangely, take a `browser_screenshot()` to visually inspect it before concluding it's broken.
- If a login page doesn't exist at the expected URL, navigate to the homepage and LOOK for a "Sign In" or "Login" link.

### Think Like a Human Browser User
- A human doesn't type exact URLs for login pages — they go to the homepage and click "Sign In".
- A human doesn't give up when one link is broken — they look for alternatives.
- A human scrolls down to see more content, uses the search bar, clicks around menus.
- **You must do the same.** Use your reasoning to adapt to what you see on the page.

### Browsing is a Loop, Not a One-Shot Command
- After every action (navigate, click, type), **always inspect the result** using `browser_get_text()`, `browser_snapshot()`, or `browser_screenshot()`.
- Based on what you see, decide your NEXT action. Do not pre-plan a rigid sequence and give up if step 1 fails.
- Keep going until the task is complete or you've exhausted all reasonable approaches (at least 3-5 different strategies).

### URL Discovery
- **Never assume URLs.** If a skill mentions a URL like `example.com/login`, treat it as a *hint*, not a guarantee.
- If the hinted URL fails, try: (1) the homepage, (2) common alternatives (`/signin`, `/auth`, `/account`), (3) use `browser_get_text` or `browser_snapshot` on the homepage to find login/signup links.

### Error Recovery Strategies
1. **Page not found (404)?** → Go to homepage, look for navigation links
2. **Element not found?** → Use `browser_snapshot()` to see all interactive elements, try text matching
3. **Form submission failed?** → Use `browser_get_text()` to read error messages, fix inputs, retry
4. **CAPTCHA or unusual page?** → Use `browser_screenshot()` to visually inspect, report to user if truly blocked
5. **Redirect to unexpected page?** → Use `browser_get_text()` to understand where you are, adapt accordingly

