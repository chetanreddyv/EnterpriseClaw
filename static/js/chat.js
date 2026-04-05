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

// Sidebar toggle logic
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');

function toggleSidebar() {
    if (sidebar.classList.contains('-translate-x-full')) {
        sidebar.classList.remove('-translate-x-full');
        sidebarOverlay.classList.remove('hidden');
        // Small delay to allow display block to apply before opacity transition
        setTimeout(() => sidebarOverlay.classList.remove('opacity-0'), 10);
    } else {
        sidebar.classList.add('-translate-x-full');
        sidebarOverlay.classList.add('opacity-0');
        setTimeout(() => sidebarOverlay.classList.add('hidden'), 300);
    }
}

if (menuToggle) menuToggle.addEventListener('click', toggleSidebar);
if (sidebarOverlay) sidebarOverlay.addEventListener('click', toggleSidebar);


const API_BASE = window.location.origin;
const seenEventIds = new Set();
let eventsPollInFlight = false;

function getCurrentTimeStr() {
    return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function createMessageElement(content, isUser = false, msgName = null) {
    if (welcomeMsg) welcomeMsg.style.display = 'none';

    const wrapperDiv = document.createElement('div');
    const timeStr = getCurrentTimeStr();

    if (isUser) {
        wrapperDiv.className = 'flex flex-col gap-2 max-w-[90%] md:max-w-[80%] self-end group msg-animate w-full items-end';
        
        wrapperDiv.innerHTML = `
            <div class="flex items-center gap-2 px-1 self-end">
                <span class="font-label text-[10px] text-outline/50 uppercase tracking-widest">${timeStr}</span>
            </div>
            <div class="bg-surface-container-lowest border border-outline-variant/30 p-4 md:p-5 rounded-2xl rounded-tr-sm shadow-sm text-on-surface-variant leading-relaxed text-[15px] whitespace-pre-wrap w-fit text-left"></div>
        `;
        wrapperDiv.lastElementChild.textContent = content; // secure raw text
    } else {
        wrapperDiv.className = 'flex flex-col gap-2 max-w-[95%] md:max-w-[85%] self-start group msg-animate w-full';
        
        const headerHtml = msgName ? `
            <div class="flex items-center gap-2 px-1">
                <span class="font-headline italic text-[13px] text-primary">${msgName}</span>
                <span class="w-1 h-1 rounded-full bg-outline-variant/60"></span>
                <span class="font-label text-[10px] text-outline/50 uppercase tracking-widest">${timeStr}</span>
            </div>
        ` : '';

        wrapperDiv.innerHTML = `
            ${headerHtml}
            <div class="agent-bubble-glass p-5 md:p-6 rounded-2xl rounded-tl-sm border-l-2 border-primary shadow-[0_20px_40px_rgba(77,70,55,0.06)] text-on-surface leading-relaxed text-[15px] font-light prose-custom"></div>
        `;

        // Parse Markdown for bot MSGs
        marked.setOptions({ breaks: true, gfm: true });
        const html = marked.parse(content);
        wrapperDiv.lastElementChild.innerHTML = DOMPurify.sanitize(html);
    }

    return wrapperDiv;
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
        '---',
        '**🔐 Action Requires Approval**',
        '',
        `> **Tool Activity:** \`${event.tool_name || 'unknown'}\``,
        '',
        '**Payload:**',
        argsText,
        '---'
    ].join('\n');

    const wrapper = createMessageElement(content, false, 'Security Gate');
    const bubbleContent = wrapper.lastElementChild;
    const controls = document.createElement('div');
    controls.className = 'approval-controls';

    const approveBtn = document.createElement('button');
    approveBtn.className = 'approval-btn approve';
    approveBtn.innerHTML = '<span class="material-symbols-outlined text-[16px] align-text-bottom mr-1">check_circle</span> Approve Action';

    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'approval-btn reject';
    rejectBtn.innerHTML = '<span class="material-symbols-outlined text-[16px] align-text-bottom mr-1">cancel</span> Reject Action';

    const editBtn = document.createElement('button');
    editBtn.className = 'approval-btn edit';
    editBtn.innerHTML = '<span class="material-symbols-outlined text-[16px] align-text-bottom mr-1">edit_square</span> Modify Context';

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
        const alertMsg = createMessageElement('✏️ Send your modifications as a normal message, then approve or reject.', false, 'System Notice');
        messagesDiv.insertBefore(alertMsg, typingIndicator);
        scrollToBottom();
    };

    async function submitApproval(action) {
        try {
            const res = await fetch(`${API_BASE}/api/v1/chat/${threadId}/resume`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
            if (!res.ok) throw new Error(`Resume API responded with ${res.status}`);
            typingIndicator.classList.add('flex');
            typingIndicator.classList.remove('hidden');
        } catch (e) {
            console.error('Approval submit failed:', e);
            messagesDiv.insertBefore(createMessageElement(`⚠️ Failed to submit ${action}: ${e.message}`, false, 'System Error'), typingIndicator);
            scrollToBottom();
            setDisabled(false);
        }
    }

    controls.appendChild(approveBtn);
    controls.appendChild(rejectBtn);
    controls.appendChild(editBtn);
    bubbleContent.appendChild(controls);
    
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

        typingIndicator.classList.remove('flex');
        typingIndicator.classList.add('hidden');

        for (const event of events) {
            if (!event || !event.id || seenEventIds.has(event.id)) continue;
            seenEventIds.add(event.id);

            if (event.type === 'message') {
                messagesDiv.insertBefore(createMessageElement(String(event.content || ''), false, 'EnterpriseClaw'), typingIndicator);
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
    typingIndicator.classList.add('flex');
    typingIndicator.classList.remove('hidden');
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
        typingIndicator.classList.remove('flex');
        typingIndicator.classList.add('hidden');
        const alertMsg = createMessageElement(`⚠️ Error connecting to server: ${e.message}`, false, "System Error");
        messagesDiv.insertBefore(alertMsg, typingIndicator);
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
