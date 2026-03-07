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

// Base URL detection
const isProd = window.location.protocol === 'https:';
const wsProtocol = isProd ? 'wss:' : 'ws:';
const API_BASE = window.location.origin;

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
