// Set threadId on load
let threadId = localStorage.getItem('ec_thread_id');
if (!threadId) {
    threadId = 'web_' + Math.random().toString(36).substring(2, 9);
    localStorage.setItem('ec_thread_id', threadId);
}
console.log("Thread ID:", threadId);

const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typing');
const welcomeMsg = document.getElementById('welcome');

const API_BASE = window.location.origin;
const seenEventIds = new Set();
let eventsPollInFlight = false;

function createMessageElement(content, isUser = false, msgName = null) {
    if (welcomeMsg) welcomeMsg.style.display = 'none';

    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${isUser ? 'user' : 'bot'}`;

    if (!isUser && msgName) {
        const nameDiv = document.createElement('div');
        nameDiv.className = 'msg-name';
        nameDiv.textContent = msgName;
        msgDiv.appendChild(nameDiv);
    }

    const contentDiv = document.createElement('div');
    if (isUser) {
        contentDiv.textContent = content; // raw text for user input
    } else {
        // Parse markdown for bot MSGs
        marked.setOptions({
            breaks: true,
            gfm: true
        });
        const html = marked.parse(content);
        contentDiv.innerHTML = DOMPurify.sanitize(html);
    }

    msgDiv.appendChild(contentDiv);
    return msgDiv;
}

function scrollToBottom() {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function createApprovalEventElement(event) {
    const args = event.args || {};
    const argsText = Object.entries(args).length
        ? Object.entries(args).map(([k, v]) => `- ${k}: \`${JSON.stringify(v)}\``).join('\n')
        : '- (No arguments)';

    const content = [
        '🔐 **Action Requires Approval**',
        '',
        `**Tool:** \`${event.tool_name || 'unknown'}\``,
        '',
        '**Arguments:**',
        argsText,
    ].join('\n');

    const wrapper = createMessageElement(content, false, 'HITL');
    const controls = document.createElement('div');
    controls.className = 'approval-controls';

    const approveBtn = document.createElement('button');
    approveBtn.className = 'approval-btn approve';
    approveBtn.textContent = 'Approve';

    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'approval-btn reject';
    rejectBtn.textContent = 'Reject';

    const editBtn = document.createElement('button');
    editBtn.className = 'approval-btn edit';
    editBtn.textContent = 'Edit';

    const setDisabled = (disabled) => {
        approveBtn.disabled = disabled;
        rejectBtn.disabled = disabled;
        editBtn.disabled = disabled;
    };

    approveBtn.onclick = async () => {
        setDisabled(true);
        await submitApproval('approve');
    };
    rejectBtn.onclick = async () => {
        setDisabled(true);
        await submitApproval('reject');
    };
    editBtn.onclick = async () => {
        setDisabled(true);
        messagesDiv.insertBefore(
            createMessageElement('✏️ Send your modifications as a normal message, then approve or reject.', false, 'System'),
            typingIndicator,
        );
        scrollToBottom();
    };

    async function submitApproval(action) {
        try {
            const res = await fetch(`${API_BASE}/api/v1/chat/${threadId}/resume`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
            if (!res.ok) {
                throw new Error(`Resume API responded with ${res.status}`);
            }
            typingIndicator.classList.add('show');
        } catch (e) {
            console.error('Approval submit failed:', e);
            messagesDiv.insertBefore(
                createMessageElement(`⚠️ Failed to submit ${action}: ${e.message}`, false, 'System'),
                typingIndicator,
            );
            scrollToBottom();
            setDisabled(false);
        }
    }

    controls.appendChild(approveBtn);
    controls.appendChild(rejectBtn);
    controls.appendChild(editBtn);
    wrapper.appendChild(controls);
    return wrapper;
}

async function pollEvents() {
    if (eventsPollInFlight) return;
    eventsPollInFlight = true;

    try {
        const res = await fetch(`${API_BASE}/api/v1/chat/${threadId}/events`);
        if (!res.ok) {
            throw new Error(`Events API responded with ${res.status}`);
        }

        const body = await res.json();
        const events = Array.isArray(body.events) ? body.events : [];
        if (!events.length) return;

        typingIndicator.classList.remove('show');

        for (const event of events) {
            if (!event || !event.id || seenEventIds.has(event.id)) continue;
            seenEventIds.add(event.id);

            if (event.type === 'message') {
                messagesDiv.insertBefore(
                    createMessageElement(String(event.content || ''), false, 'EnterpriseClaw'),
                    typingIndicator,
                );
                continue;
            }

            if (event.type === 'approval') {
                messagesDiv.insertBefore(createApprovalEventElement(event), typingIndicator);
            }
        }

        scrollToBottom();
    } catch (e) {
        console.error('Event polling failed:', e);
    } finally {
        eventsPollInFlight = false;
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    userInput.style.height = 'auto'; // reset height
    sendBtn.disabled = true;

    // Show user message instantly
    messagesDiv.insertBefore(createMessageElement(text, true), typingIndicator);
    scrollToBottom();

    // Show typing
    typingIndicator.classList.add('show');
    scrollToBottom();

    try {
        const res = await fetch(`${API_BASE}/api/v1/chat/${threadId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: text })
        });
        if (!res.ok) throw new Error("API responded with " + res.status);
    } catch (e) {
        console.error("Transmission error:", e);
        typingIndicator.classList.remove('show');
        const alert = createMessageElement(`⚠️ Error connecting to server: ${e.message}`, false, "System");
        alert.style.border = "1px solid var(--error)";
        messagesDiv.insertBefore(alert, typingIndicator);
        scrollToBottom();
    }
    
    // Typing indicator should be removed by the polling function when response arrives
    setTimeout(() => { sendBtn.disabled = false; userInput.focus(); }, 500);
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value.trim().length > 0) {
        sendBtn.disabled = false;
    } else {
        sendBtn.disabled = true;
    }
});

setInterval(pollEvents, 1200);
pollEvents();
