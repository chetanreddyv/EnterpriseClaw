---
name: job_application_automation
description: Automate job applications on job boards. Only triggered when user explicitly asks to apply to a job.
tools: browser_navigate, browser_click, browser_type, browser_screenshot, browser_get_text, browser_snapshot, browser_go_back, browser_scroll, browser_wait_for, browser_tab_management, browser_select_option, browser_press_key, browser_hover, browser_handle_dialog, browser_file_upload
---

# Job Application Automation Skill

> **⚠️ ACTIVATION RULE:** This skill should ONLY be used when the user explicitly asks to **apply** to jobs, submit applications, or automate job searching on job boards. Do NOT activate for general LinkedIn browsing, profile viewing, or other non-application tasks.

This skill guides the AI agent to search for jobs, evaluate them against user preferences, and automatically complete and submit job applications on behalf of the user using the `browser_use` tools.

**Note:** User profile data (resume, contact info, job preferences) is stored in the `user_profile` skill. Reference it when filling application forms.

## Standard Operating Procedure (SOP)

### Phase 1: Authentication
1.  **Navigate to Homepage:** Use `browser_navigate(url)` to go to the preferred job board's homepage (e.g., `https://jobright.ai`). Do NOT guess the login URL.
2.  **Find the Login Link:** Use `browser_snapshot()` to find a "Sign In", "Login", or "Log In" link/button on the homepage, then use `browser_click` on it.
    * *Fallback:* If no visible link, try common paths: `/login`, `/signin`, `/auth/login`. If none work, try Google: `site:jobright.ai login`.
3.  **Wait:** Use `browser_wait_for` to ensure the login form is fully loaded. Use `browser_snapshot()` to map the form fields.
4.  **Login:** * Fetch the username/password from the `.env` environment.
    * Use `browser_type(selector, email, submit=False)` for the email field.
    * Use `browser_type(selector, password, submit=True)` for the password field to log in.
    * *Fallback:* Use `browser_click` on the "Sign In" button if `submit=True` does not trigger the login.
5.  **Verify Login:** After login attempt, use `browser_get_text()` to confirm you are logged in (e.g., username visible, dashboard loaded). If still on login page, read error messages and retry.

### Phase 2: Search & Filter
1.  **Navigate to Search:** Use `browser_navigate(url)` to go directly to the search page, or use `browser_click` to access the search bar.
2.  **Input Criteria:** Use `browser_type` to input preferred Job Titles and Locations based on the user profile preferences.
3.  **Apply Filters:** Use `browser_click` or `browser_select_option` to apply filters (e.g., "Remote", "Past Week").
4.  **Read Results:** Use `browser_snapshot()` and `browser_get_text()` to identify listed jobs.

### Phase 3: Selection & Application
1.  **Select a Job:** Click on a relevant job listing using `browser_click(selector)`.
2.  **Evaluate:** Read the job description using `browser_get_text()`. If it heavily conflicts with user preferences (e.g., requires 10 years of experience when the user has 2), use `browser_go_back()` and select the next job.
3.  **Initiate Application:** Use `browser_click` on the "Apply Now" or "Easy Apply" button.
4.  **Fill Forms:**
    * Use `browser_snapshot()` to map out the application form fields.
    * Use `browser_type` to fill in text fields (Name, Email, Phone, LinkedIn) from user profile.
    * Use `browser_select_option` for dropdowns (e.g., Work Authorization, Race/Gender demographics if requested).
5.  **Upload Resume:** Use `browser_file_upload(selector, [Resume Path])` to attach the user's resume.

### Phase 4: Review & Submit
1.  **Verify:** Use `browser_get_text()` or `browser_snapshot()` to ensure all required fields are filled correctly and no error messages are displayed.
2.  **Submit:** Use `browser_click` on the "Submit Application" button.
3.  **Handle Dialogs:** If a confirmation popup appears, use `browser_handle_dialog('accept')`.
4.  **Next Job:** Once the success message is confirmed, use `browser_go_back()` or navigate back to the search results to repeat the process.

## Best Practices & Error Handling

* **Complex Applications:** If the "Apply" button redirects to an external site (like Workday) that requires a completely new account creation not covered in `.env`, skip it, log the URL for the user, and move to the next job. Use `browser_tab_management` to close the unwanted tab.
* **Dynamic Forms:** Many application forms are multi-page (SPAs). Use `browser_wait_for` after clicking "Next" before trying to use `browser_snapshot` on the new fields.
