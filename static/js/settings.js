// ============================================================
// EnterpriseClaw — Settings Panel Controller
// ============================================================

const settingsPanel = document.getElementById('settingsPanel');
const chatPanel = document.getElementById('chatPanel');
const settingsNavBtn = document.getElementById('settingsNavBtn');
const chatNavBtn = document.getElementById('chatNavBtn');
const settingsBody = document.getElementById('settingsBody');
const settingsToast = document.getElementById('settingsToast');

const SETTINGS_API = `${window.location.origin}/api/v1/settings`;

// ── Navigation ─────────────────────────────────────────────
function showSettings() {
    chatPanel.classList.add('hidden');
    settingsPanel.classList.remove('hidden');
    // Update nav active states
    chatNavBtn.classList.remove('bg-primary-container/10', 'text-primary', 'border-primary/10');
    chatNavBtn.classList.add('text-on-surface-variant', 'hover:bg-surface-container-high');
    settingsNavBtn.classList.add('bg-primary-container/10', 'text-primary', 'border-primary/10');
    settingsNavBtn.classList.remove('text-on-surface-variant', 'hover:bg-surface-container-high');
    loadSettings();
}

function showChat() {
    settingsPanel.classList.add('hidden');
    chatPanel.classList.remove('hidden');
    chatNavBtn.classList.add('bg-primary-container/10', 'text-primary', 'border-primary/10');
    chatNavBtn.classList.remove('text-on-surface-variant', 'hover:bg-surface-container-high');
    settingsNavBtn.classList.remove('bg-primary-container/10', 'text-primary', 'border-primary/10');
    settingsNavBtn.classList.add('text-on-surface-variant', 'hover:bg-surface-container-high');
}

settingsNavBtn.addEventListener('click', (e) => { e.preventDefault(); showSettings(); });
chatNavBtn.addEventListener('click', (e) => { e.preventDefault(); showChat(); });

// ── Toast Notification ─────────────────────────────────────
function showToast(message, type = 'success') {
    const icon = type === 'success' ? 'check_circle' : type === 'error' ? 'error' : 'info';
    const colors = {
        success: 'border-green-500/30 text-green-700',
        error: 'border-red-500/30 text-red-700',
        info: 'border-primary/30 text-primary',
    };
    settingsToast.innerHTML = `
        <div class="flex items-center gap-2 px-4 py-3 rounded-xl bg-surface-container-lowest/95 backdrop-blur-xl border ${colors[type]} shadow-lg font-label text-sm font-medium msg-animate">
            <span class="material-symbols-outlined text-base" style="font-variation-settings: 'FILL' 1;">${icon}</span>
            <span>${message}</span>
        </div>
    `;
    settingsToast.classList.remove('hidden');
    setTimeout(() => { settingsToast.classList.add('hidden'); }, 3000);
}

// ── Setting Definitions (mirrors MUTABLE_SETTINGS in app.py) ──
const SETTING_CATEGORIES = [
    {
        id: 'model',
        title: 'Model',
        icon: 'smart_toy',
        description: 'Configure the default LLM provider and model.',
        settings: [
            { key: 'default_model', label: 'Default Model', type: 'text', placeholder: 'e.g. openai/gpt-5.4-mini', description: 'Provider/model string used when no model is explicitly set.' },
        ]
    },
    {
        id: 'security',
        title: 'Security & HITL',
        icon: 'security',
        description: 'Control human-in-the-loop approval gates.',
        settings: [
            { key: 'hitl_enabled', label: 'HITL Enabled', type: 'toggle', description: 'When enabled, dangerous tool calls require explicit human approval before execution.' },
        ]
    },
    {
        id: 'worker',
        title: 'Worker Agent',
        icon: 'precision_manufacturing',
        description: 'Tune the Worker agent\'s execution limits.',
        settings: [
            { key: 'worker_max_steps', label: 'Max Steps', type: 'number', min: 1, max: 100, description: 'Maximum action steps a Worker can take before returning.' },
            { key: 'worker_max_observation_chars', label: 'Max Observation Chars', type: 'number', min: 1000, max: 200000, description: 'Character limit for environment observation payloads.' },
            { key: 'worker_max_skill_prompt_chars', label: 'Max Skill Prompt Chars', type: 'number', min: 1000, max: 100000, description: 'Character limit for skill prompt context injected into Worker.' },
        ]
    },
    {
        id: 'supervisor',
        title: 'Supervisor',
        icon: 'account_tree',
        description: 'Control the Supervisor\'s memory and truncation behavior.',
        settings: [
            { key: 'supervisor_token_budget', label: 'Token Budget', type: 'number', min: 1000, max: 50000, description: 'Token budget for message history trimming.' },
            { key: 'supervisor_content_truncation', label: 'Content Truncation', type: 'number', min: 500, max: 50000, description: 'Max characters for AI/Tool content before truncation.' },
        ]
    },
    {
        id: 'scheduler',
        title: 'Scheduler',
        icon: 'schedule',
        description: 'Configure the background task scheduler.',
        settings: [
            { key: 'scheduler_max_concurrent_jobs', label: 'Max Concurrent Jobs', type: 'number', min: 1, max: 32, description: 'Maximum cron jobs that can run simultaneously.' },
            { key: 'scheduler_heartbeat_seconds', label: 'Heartbeat Interval', type: 'number', min: 1, max: 300, description: 'Seconds between scheduler heartbeat checks.' },
            { key: 'scheduler_run_history_max', label: 'Run History Max', type: 'number', min: 10, max: 10000, description: 'Maximum run history records to keep per job.' },
        ]
    },
];

// ── Load & Render ──────────────────────────────────────────
let currentSettings = {};

async function loadSettings() {
    settingsBody.innerHTML = `
        <div class="flex items-center justify-center py-20">
            <div class="flex items-center gap-3 text-on-surface-variant/60 font-label text-sm">
                <div class="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin"></div>
                Loading configuration…
            </div>
        </div>
    `;

    try {
        const res = await fetch(SETTINGS_API);
        if (!res.ok) throw new Error(`API responded with ${res.status}`);
        const data = await res.json();
        currentSettings = data.settings || {};
        renderSettings();
    } catch (e) {
        settingsBody.innerHTML = `
            <div class="flex flex-col items-center justify-center py-20 gap-4">
                <span class="material-symbols-outlined text-4xl text-error/60">cloud_off</span>
                <p class="text-on-surface-variant/60 font-label text-sm">Failed to load settings</p>
                <button onclick="loadSettings()" class="text-primary font-label text-sm font-semibold hover:underline">Retry</button>
            </div>
        `;
    }
}

function renderSettings() {
    const html = SETTING_CATEGORIES.map(cat => `
        <div class="settings-category msg-animate">
            <div class="flex items-center gap-3 mb-1">
                <div class="w-9 h-9 rounded-xl bg-primary-container/15 flex items-center justify-center">
                    <span class="material-symbols-outlined text-primary text-lg">${cat.icon}</span>
                </div>
                <div>
                    <h3 class="font-headline font-bold text-base text-on-surface tracking-tight">${cat.title}</h3>
                    <p class="font-label text-[11px] text-on-surface-variant/60">${cat.description}</p>
                </div>
            </div>
            <div class="mt-4 flex flex-col gap-1">
                ${cat.settings.map(s => renderSettingRow(s)).join('')}
            </div>
        </div>
    `).join('');

    settingsBody.innerHTML = html;
}

function renderSettingRow(s) {
    const value = currentSettings[s.key];
    
    const detailsHtml = `
        <details class="group mt-1.5">
            <summary class="list-none flex items-center cursor-pointer font-label text-[11px] text-primary/70 hover:text-primary transition-colors select-none">
                <span class="material-symbols-outlined text-[14px] mr-1 transition-transform duration-200 group-open:rotate-90">chevron_right</span>
                About this setting
            </summary>
            <div class="mt-2 text-[11.5px] text-on-surface-variant/80 pl-4 pr-2 pb-1 leading-relaxed border-l-[1.5px] border-primary/20 ml-[6px]">
                ${s.description}
            </div>
        </details>
    `;

    if (s.type === 'toggle') {
        const checked = value === true || value === 'true';
        return `
            <div class="setting-row items-start py-3">
                <div class="flex-grow min-w-0 pt-0.5">
                    <div class="font-label text-sm font-semibold text-on-surface">${s.label}</div>
                    ${detailsHtml}
                </div>
                <label class="setting-toggle mt-1">
                    <input type="checkbox" ${checked ? 'checked' : ''} onchange="updateSetting('${s.key}', this.checked)" />
                    <span class="toggle-track"><span class="toggle-thumb"></span></span>
                </label>
            </div>
        `;
    }

    if (s.type === 'number') {
        return `
            <div class="setting-row items-start py-3">
                <div class="flex-grow min-w-0 pt-1">
                    <div class="font-label text-sm font-semibold text-on-surface">${s.label}</div>
                    ${detailsHtml}
                </div>
                <input type="number" value="${value ?? ''}" min="${s.min ?? ''}" max="${s.max ?? ''}"
                    class="form-input w-[100px] text-right font-semibold text-[13px] bg-white/70 border-[1.5px] border-primary/20 rounded-xl focus:border-primary focus:ring focus:ring-primary/20 transition-all shadow-sm"
                    data-key="${s.key}"
                    onchange="updateSetting('${s.key}', parseInt(this.value))"
                    onkeydown="if(event.key==='Enter'){this.blur();}" />
            </div>
        `;
    }

    // text input
    return `
        <div class="setting-row items-start py-3">
            <div class="flex-grow min-w-0 pt-1">
                <div class="font-label text-sm font-semibold text-on-surface">${s.label}</div>
                ${detailsHtml}
            </div>
            <input type="text" value="${value ?? ''}" placeholder="${s.placeholder || ''}"
                class="form-input w-[220px] font-mono text-[12px] bg-white/70 border-[1.5px] border-primary/20 rounded-xl focus:border-primary focus:ring focus:ring-primary/20 transition-all shadow-sm"
                data-key="${s.key}"
                onchange="updateSetting('${s.key}', this.value)"
                onkeydown="if(event.key==='Enter'){this.blur();}" />
        </div>
    `;
}

// ── Update Setting ─────────────────────────────────────────
async function updateSetting(key, value) {
    try {
        const res = await fetch(SETTINGS_API, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, value: String(value) }),
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Failed to update setting', 'error');
            return;
        }
        currentSettings[key] = data.value;
        showToast(`${key} updated successfully`);
    } catch (e) {
        showToast(`Network error: ${e.message}`, 'error');
    }
}

// ── Reset Settings ─────────────────────────────────────────
async function resetSettings() {
    try {
        const res = await fetch(`${SETTINGS_API}/reset`, { method: 'POST' });
        if (!res.ok) throw new Error(`API responded with ${res.status}`);
        showToast('Settings reloaded from .env', 'info');
        loadSettings();
    } catch (e) {
        showToast(`Failed to reset: ${e.message}`, 'error');
    }
}
