// ═══ ONBOARDING JS ═══
// Handles the 5-step interactive setup wizard

let state = {
  currentStep: 0,
  values: { gemini: '', telegram: '', chatId: '', botName: '' },
  validated: { gemini: false, telegram: false, chatId: false }
};

const pages = document.querySelectorAll('.page');
const steps = document.querySelectorAll('.steps li');

function goToStep(index) {
  // Hide all
  pages.forEach(p => p.classList.remove('active'));
  steps.forEach(s => s.classList.remove('active'));

  // Show target
  document.getElementById(`page-${index}`).classList.add('active');
  steps[index].classList.add('active');

  // Mark previous as completed
  for (let i = 0; i < index; i++) {
    steps[i].classList.add('completed');
  }

  state.currentStep = index;

  if (index === 5) {
    saveConfig();
  }
}

function syncNextButtons() {
  document.getElementById('btnNext1').disabled = !state.validated.gemini;
  document.getElementById('btnNext2').disabled = !state.validated.telegram;
  document.getElementById('btnNext3').disabled = !state.validated.chatId;
}

// ═══ API Validation Calls ═══

async function validateGemini() {
  const key = document.getElementById('geminiKey').value.trim();
  if (!key) return;
  const btn = document.getElementById('btnValidateGemini');
  const fb = document.getElementById('geminiResult');
  const field = document.getElementById('geminiKey');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Checking...';

  try {
    const resp = await fetch('/api/validate/gemini', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key})
    });
    const data = await resp.json();
    fb.className = 'feedback ' + (data.valid ? 'success' : 'error');
    fb.textContent = data.message;
    field.className = 'input-field ' + (data.valid ? 'valid' : 'invalid');
    state.validated.gemini = data.valid;
    if (data.valid) state.values.gemini = key;
    syncNextButtons();
  } catch (e) {
    fb.className = 'feedback error';
    fb.textContent = 'Network error — check your connection.';
  }
  btn.disabled = false;
  btn.textContent = 'Validate';
}

async function validateTelegram() {
  const key = document.getElementById('telegramToken').value.trim();
  if (!key) return;
  const btn = document.getElementById('btnValidateTelegram');
  const fb = document.getElementById('telegramResult');
  const field = document.getElementById('telegramToken');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Checking...';

  try {
    const resp = await fetch('/api/validate/telegram', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key})
    });
    const data = await resp.json();
    fb.className = 'feedback ' + (data.valid ? 'success' : 'error');
    fb.textContent = data.message;
    field.className = 'input-field ' + (data.valid ? 'valid' : 'invalid');
    state.validated.telegram = data.valid;
    if (data.valid) {
      state.values.telegram = key;
      state.values.botName = data.bot_name || '';
    }
    syncNextButtons();
  } catch (e) {
    fb.className = 'feedback error';
    fb.textContent = 'Network error — check your connection.';
  }
  btn.disabled = false;
  btn.textContent = 'Validate';
}

async function validateChatId() {
  const key = document.getElementById('chatIds').value.trim();
  if (!key) return;
  const btn = document.getElementById('btnValidateChat');
  const fb = document.getElementById('chatIdResult');
  const field = document.getElementById('chatIds');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Checking...';

  try {
    const resp = await fetch('/api/validate/chat-id', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key})
    });
    const data = await resp.json();
    fb.className = 'feedback ' + (data.valid ? 'success' : 'error');
    fb.textContent = data.message;
    field.className = 'input-field ' + (data.valid ? 'valid' : 'invalid');
    state.validated.chatId = data.valid;
    if (data.valid) state.values.chatId = key;
    syncNextButtons();
  } catch (e) {
    fb.className = 'feedback error';
    fb.textContent = 'Network error — check your connection.';
  }
  btn.disabled = false;
  btn.textContent = 'Validate';
}

// ═══ Save Config ═══
async function saveConfig() {
  // Update completion summary
  const geminiMask = state.values.gemini ? '••••' + state.values.gemini.slice(-6) : '—';
  document.getElementById('sumGemini').textContent = geminiMask + ' ✅';
  document.getElementById('sumBot').textContent = state.values.botName
    ? `${state.values.botName} ✅` : '✅';
  document.getElementById('sumUsers').textContent = state.values.chatId + ' ✅';
  const fbResult = document.getElementById('finalResult');

  // POST to backend
  try {
    const response = await fetch('/api/complete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        google_api_key: state.values.gemini,
        telegram_bot_token: state.values.telegram,
        allowed_chat_ids: state.values.chatId,
      })
    });
    const data = await response.json();
    
    if (data.success) {
      fbResult.className = 'feedback success';
      fbResult.textContent = 'Successfully wrote to .env! You can now start the application.';
      document.getElementById('finalAction').style.display = 'block';
    } else {
      throw new Error(data.message || 'Unknown error');
    }
  } catch (e) {
    console.error('Failed to save config:', e);
    fbResult.className = 'feedback error';
    fbResult.textContent = 'Error writing config: ' + e.message;
  }
}

function copyCmd() {
  navigator.clipboard.writeText('uv run python app.py');
  const btn = window.event.target;
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 1500);
}

// ═══ Enter key support ═══
document.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const step = state.currentStep;
  if (step === 1) validateGemini();
  else if (step === 2) validateTelegram();
  else if (step === 3) validateChatId();
});
