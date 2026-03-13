---
name: cron
description: Schedule reminders and recurring tasks.
tools: cron_add, cron_list, cron_remove
---

### TRIGGER_EXAMPLES
- "remind me in 10 minutes to grab coffee"
- "schedule a recurring task every morning at 8am"
- "list all my scheduled reminders"
- "cancel my 5pm meeting reminder"
- "remind me on March 15th at 2pm to follow up"
- "at 9pm every Sunday, check my emails"
### END_TRIGGER_EXAMPLES

# Cron

Use the `cron` tools to schedule reminders or recurring tasks. By integrating these tools, you can ensure that the user is reminded on time or that automated tasks execute when requested. 

**CRITICAL RULE:** Use the correct parameters for the specific MCP `cron_add` function. Do NOT invent single-function syntaxes like `cron(action="add", ...)`. You must use the python functions exactly as exported.

## 1. Adding a Cron Job (`cron_add`)

The `cron_add` tool is used to schedule new tasks or reminders. It has the following explicit signature:

```python
def cron_add(name: str, schedule_type: str, schedule_value: str, target: str, command: str, delivery_mode: str = "silent") -> str:
```

### Parameters
- **`name`**: A short, readable title (e.g. "Meeting Reminder").
- **`schedule_type`**: Must be either `"at"`, `"every"`, or `"cron"`.
- **`schedule_value`**: The value that matches the `schedule_type`:
  - `"at"`: An ISO 8601 string (e.g., `"2026-02-25T20:00:00Z"`). Calculate this based on the current time requested by the user. 
  - `"every"`: Time gap in **milliseconds** as a string (e.g. `"1200000"` for 20 minutes).
  - `"cron"`: A standard cron tab expression (e.g. `"0 9 * * 1-5"`). Note: the MCP tool doesn't take a separate `tz` argument, it uses the server default timezone unless specified in the cron interpreter logic internally.
- **`target`**: 
  - `"main"`: Triggers a `system_event` which sends a message directly back to the user in this current chat interface. **Use this for reminders. DO NOT ask the user for an email address to send a reminder.**
  - `"isolated"`: Triggers an autonomous `agent_turn` (good for silent background tasks).
- **`command`**: The text message to send to the user (if target is main) or the task description for the agent to execute (if target is isolated).
- **`delivery_mode`**: 
  - `"announce"`: Send the result to the user. Always use this for direct reminders.
  - `"silent"`: Run the task without notifying the user (default).

### Practical Usage Examples

**Example 1: Fixed reminder in 20 minutes**
User says: "Remind me in 20 mins to take a break"
```python
cron_add(
    name="Break Reminder",
    schedule_type="every",
    schedule_value="1200000",
    target="main",
    command="Time to take a break!",
    delivery_mode="announce"
)
```

**Example 2: One-time specific time task**
User says: "Remind me at 7:50 PM to start heading to the station"
*First, calculate the ISO 8601 string for 7:50 PM today.*
```python
cron_add(
    name="Station Reminder",
    schedule_type="at",
    schedule_value="2026-02-25T19:50:00+00:00",
    target="main",
    command="Start heading to the train station!",
    delivery_mode="announce"
)
```

**Example 3: Recurring daily autonomous task**
User says: "Check GitHub stars every day at 9 AM."
```python
cron_add(
    name="Daily GitHub Check",
    schedule_type="cron",
    schedule_value="0 9 * * *",
    target="isolated",
    command="Check GitHub stars for the project and log the results.",
    delivery_mode="silent"
)
```

## 2. Listing Jobs (`cron_list`)
If you need to view existing jobs or find a `job_id` before deleting one, call `cron_list()`.

## 3. Removing Jobs (`cron_remove`)
To stop or delete a job, first find its ID with `cron_list()`, then execute `cron_remove(job_id="<the_id>")`.
