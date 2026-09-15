/* IHACDP consultation client. Transcript + record live only in this tab. */
const S = { messages: [], record: {}, busy: false, reported: false, fields: {} };

const $ = id => document.getElementById(id);

/* Follow new output only when the reader is already near the bottom. */
let stick = true;
const NEAR_BOTTOM = 140;
function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM;
}
function scrollToBottom(smooth) {
  if (!stick) return;
  thread.scrollTo({ top: thread.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
}
const thread = $('thread'), threadIn = $('thread-in'), input = $('input'), sendBtn = $('send');

fetch('/api/fields').then(r => r.json()).then(d => { S.fields = d.fields || {}; }).catch(() => {});

/* ---------- rendering ---------- */
function esc(s) { return (s || '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

function addTurn(role, text, id) {
  const empty = $('thread-empty');
  if (empty) empty.hidden = true;
  const col = $('chat-col');
  if (col) col.classList.remove('centred');
  const me = role === 'user';
  const el = document.createElement('div');
  el.className = 'turn' + (me ? ' me' : '');
  const USER_ICON = '<svg viewBox="0 0 256 256" aria-hidden="true" focusable="false">'
    + '<path d="M230.92,212c-15.23-26.33-38.7-45.21-66.09-54.16a72,72,0,1,0-73.66,0'
    + 'C63.78,166.78,40.31,185.66,25.08,212a8,8,0,1,0,13.85,8c18.84-32.56,52.14-52,89.07-52'
    + 's70.23,19.44,89.07,52a8,8,0,1,0,13.85-8ZM72,96a56,56,0,1,1,56,56A56.06,56.06,0,0,1,72,96Z"/></svg>';
  el.innerHTML = `${me ? '<div class="av pat">' + USER_ICON + '</div>'
                        : '<img class="av doc" src="/assets/logo.png" alt="">'}
    <div class="bubble"><div class="who">${me ? 'You' : 'IHACDP'}</div>
    <div class="msg"${id ? ` id="${id}"` : ''}>${text === null ? '<span class="dots"><i></i><i></i><i></i></span>' : esc(text)}</div></div>`;
  threadIn.appendChild(el);
  /* Sending always takes you to your own message. An incoming reply must not
     drag the view away from whatever the reader is looking at. */
  if (me) stick = true;
  scrollToBottom(true);
  return el.querySelector('.msg');
}

const LABEL = k => (k.charAt(0).toUpperCase() + k.slice(1)).replace(/_/g, ' ')
  .replace(/\bbp\b/i, 'BP').replace(/\bdx\b/i, 'diagnosed').replace(/\bbmi\b/i, 'BMI');

const GROUPS = [
  ['Demographics', ['age', 'sex', 'height_cm', 'weight_kg', 'bmi']],
  ['Vitals', ['systolic_bp', 'diastolic_bp', 'max_heart_rate']],
  ['History', ['smoker', 'heavy_alcohol', 'physical_activity', 'general_health', 'difficulty_walking',
    'hypertension_dx', 'diabetes_dx', 'high_cholesterol_dx', 'prior_stroke', 'prior_heart_disease',
    'chest_pain_type', 'exercise_angina', 'pedal_edema', 'anaemia', 'appetite']],
  ['Laboratory', ['total_cholesterol', 'fasting_glucose', 'random_glucose', 'serum_creatinine', 'blood_urea',
    'haemoglobin', 'packed_cell_volume', 'sodium', 'potassium', 'specific_gravity', 'urine_albumin',
    'total_bilirubin', 'direct_bilirubin', 'alt_sgpt', 'ast_sgot', 'alkaline_phosphatase',
    'total_protein', 'albumin', 'ag_ratio', 'st_depression']],
];

function fmt(k, v) {
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (k === 'chest_pain_type') return ['', 'Typical angina', 'Atypical', 'Non-anginal', 'Asymptomatic'][v] || v;
  if (k === 'general_health') return ['', 'Excellent', 'Good', 'Fair', 'Poor', 'Very poor'][v] || v;
  const u = (S.fields[k] || {}).unit || '';
  return (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : v) +
    (u && !u.includes('=') && u.length < 8 ? ' ' + u : '');
}

function renderFacts() {
  const box = $('facts');
  if (!box) return;
  const rec = S.record;
  const keys = Object.keys(rec).filter(k => rec[k] !== null && rec[k] !== undefined && !k.startsWith('wdbc_'));
  if (!keys.length) { box.innerHTML = ''; return; }

  const symptoms = keys.filter(k => k.startsWith('sx_'));
  const clinical = GROUPS.flatMap(([, fs]) => fs.filter(f => keys.includes(f)));
  const total = symptoms.length + clinical.length;
  if (!total) { box.innerHTML = ''; return; }

  const chips = symptoms.map(k => {
    const name = LABEL(k.slice(3));
    return `<span class="chip">${esc(name)} <b>${rec[k] ? 'yes' : 'no'}</b></span>`;
  }).concat(clinical.map(f =>
    `<span class="chip">${LABEL(f)} <b>${esc(String(fmt(f, rec[f])))}</b></span>`));

  box.innerHTML = `<div class="fu-head">Findings used (${total})</div><div class="chips">` +
    chips.join('') + '</div>';
}


function renderRisks(results, conditions) {
  const box = $('risks');
  let html = '';

  /* Everyday conditions first - this is what most consultations produce. */
  if (conditions && conditions.length) {
    html += conditions.map(c => {
      const pct = (c.probability * 100).toFixed(0);
      const drivers = (c.matched_symptoms || []).slice(0, 4)
        .map(m => `<div><span>${esc(m)}</span></div>`).join('');
      return `<div class="risk">
        <div class="risk-top"><span class="nm">${esc(c.condition)}</span>
          ${c.urgent ? '<span class="tag t-VeryHigh">Urgent</span>'
                     : `<span class="tag" style="background:var(--bg-3);color:var(--ink-3)">${esc(c.confidence || '')}</span>`}</div>
        <div class="pct">${pct}<span style="font-size:14px;color:var(--ink-4)">% match</span></div>
        <div class="bar"><span class="b-Moderate" style="width:${pct}%"></span></div>
        ${drivers ? `<div class="drv">${drivers}</div>` : ''}</div>`;
    }).join('');
  }

  /* Chronic models appear only when one actually ran. Listing five of them as
     "No data — needs serum_creatinine" at someone with a sprained ankle is
     noise, and those are internal field names besides. */
  const assessed = (results || []).filter(r => r.status === 'ok');
  if (assessed.length) html += `<div class="grp">Risk Models</div>`;
  box.innerHTML = html + assessed.map(r => {
    const b = r.risk_band.replace(/\s/g, '');
    const drivers = (r.contributions || []).filter(c => c.observed).slice(0, 3).map(c =>
      `<div><span>${esc(c.label || LABEL(c.feature))} = ${esc(String(c.value))}</span>
       <span class="${c.direction === 'increases' ? 'up' : 'dn'}">${c.direction === 'increases' ? '▲' : '▼'}</span></div>`).join('');
    return `<div class="risk">
      <div class="risk-top"><span class="nm">${esc(r.label)}</span><span class="tag t-${b}">${esc(r.risk_band)}</span></div>
      <div class="pct">${(r.probability * 100).toFixed(1)}<span style="font-size:14px;color:var(--ink-4)">%</span></div>
      <div class="sub">${esc(r.model)} · AUC ${r.auc} · coverage ${Math.round(r.coverage * 100)}%</div>
      <div class="bar"><span class="b-${b}" style="width:${(r.probability * 100).toFixed(1)}%"></span></div>
      ${drivers ? `<div class="drv">${drivers}</div>` : ''}</div>`;
  }).join('');
}

/* minimal markdown -> html (no external library, offline safe) */
function md(t) {
  const lines = esc(t).split('\n');
  let out = '', list = false;
  for (let ln of lines) {
    ln = ln.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
           .replace(/(?<!\*)\*(?!\s)([^*]+?)\*/g, '<em>$1</em>')
           .replace(/`([^`]+)`/g, '<code>$1</code>');
    const li = ln.match(/^\s*[-*•]\s+(.*)$/) || ln.match(/^\s*\d+\.\s+(.*)$/);
    if (li) { if (!list) { out += '<ul>'; list = true; } out += `<li>${li[1]}</li>`; continue; }
    if (list) { out += '</ul>'; list = false; }
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h) { out += `<h${Math.min(h[1].length + 1, 4)}>${h[2]}</h${Math.min(h[1].length + 1, 4)}>`; continue; }
    if (ln.trim()) out += `<p>${ln}</p>`;
  }
  return out + (list ? '</ul>' : '');
}

/* ---------- SSE over POST ---------- */
async function stream(url, body, onEvent) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split('\n\n'); buf = parts.pop();
    for (const p of parts) {
      const line = p.split('\n').find(l => l.startsWith('data: '));
      if (line) { try { onEvent(JSON.parse(line.slice(6))); } catch (e) {} }
    }
  }
}

/* ---------- actions ---------- */
function setBusy(b) {
  S.busy = b;
  sendBtn.disabled = b || !input.value.trim();
  input.disabled = b;
}

async function send() {
  const text = input.value.trim();
  if (!text || S.busy) return;
  input.value = ''; input.style.height = 'auto';
  addTurn('user', text);
  S.messages.push({ role: 'user', content: text });
  setBusy(true);
  const node = addTurn('assistant', null);
  let acc = '', first = true, autoAssess = false;

  try {
    await stream('/api/chat', { messages: S.messages, record: S.record }, ev => {
      if (ev.type === 'token') {
        if (first) { node.textContent = ''; first = false; }
        acc += ev.t; node.textContent = acc; scrollToBottom(false);
      } else if (ev.type === 'final') {
        node.textContent = ev.text || acc;
        S.messages.push({ role: 'assistant', content: ev.text || acc });
        S.record = ev.record || S.record;
        autoAssess = !!(ev.readiness && ev.readiness.should_assess);
      } else if (ev.type === 'error') {
        node.textContent = '⚠ ' + ev.message;
      }
    });
  } catch (e) {
    node.textContent = '⚠ Could not reach the local service. Is the server still running?';
  }
  setBusy(false);
  /* IHACDP decides when the history is complete - no button press required */
  if (autoAssess && !S.reported) {
    S.reported = true;
    showPreparing();
    setTimeout(assess, 700);
  } else {
    input.focus();
  }
}

function showPreparing() {
  const el = document.createElement('div');
  el.className = 'preparing';
  el.id = 'preparing';
  el.innerHTML = '<span class="dots"><i></i><i></i><i></i></span><span>Reviewing your findings…</span>';
  threadIn.appendChild(el);
  scrollToBottom(true);
}

async function assess() {
  if (S.busy) return;
  setBusy(true);
  const prep = $('preparing'); if (prep) prep.remove();
  const rep = $('report'), body = $('report-body');
  rep.hidden = false; body.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';
  let acc = '', first = true;
  try {
    await stream('/api/report', { record: S.record, messages: S.messages }, ev => {
      if (ev.type === 'assessment') {
        renderRisks(ev.data.results, ev.data.symptom_conditions);
        S.record = ev.data.record || S.record; renderFacts();
      } else if (ev.type === 'token') {
        if (first) { body.innerHTML = ''; first = false; }
        acc += ev.t; body.innerHTML = md(acc); scrollToBottom(false);
      } else if (ev.type === 'final') {
        body.innerHTML = md(ev.text || acc);
      } else if (ev.type === 'error') {
        body.innerHTML = `<p>⚠ ${esc(ev.message)}</p>`;
      }
    });
  } catch (e) {
    body.innerHTML = '<p>⚠ Assessment failed — the local service is unreachable.</p>';
  }
  setBusy(false);
}

/* ---------- wiring ---------- */
/* Any of these mean the reader took control of the viewport. Wheel and touch
   are listened to as well as scroll, so intent is caught even where scroll
   events are coalesced. */
['scroll', 'wheel', 'touchmove'].forEach(evt =>
  thread.addEventListener(evt, () => { stick = atBottom(thread); }, { passive: true }));
thread.addEventListener('keydown', e => {
  if (['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
    setTimeout(() => { stick = atBottom(thread); }, 0);
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 150) + 'px';
  sendBtn.disabled = !input.value.trim() || S.busy;
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

sendBtn.disabled = true;
input.focus();
