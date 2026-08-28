"use strict";

const state = {
  requests: [], selected: null, detail: null,
  view: localStorage.getItem("manual-vlm-view") || "operate",
  messagePage: 0, textPages: {}, composerMode: "tool",
  selectedTool: "", toolQuery: "", parameters: {}, reasoning: "",
  selectedAction: "", actionMessage: ""
};
const PAGE_MESSAGES = 4, PAGE_CHARS = 12000, PAGE_LINES = 100;
const qs = selector => document.querySelector(selector);
const esc = value => String(value == null ? "" : value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const pretty = value => JSON.stringify(value, null, 2);
const isEmpty = value => value == null || value === "" || (Array.isArray(value) && !value.length) || (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length);

function toast(message) {
  const node = qs("#toast"); node.textContent = message; node.classList.remove("hidden");
  setTimeout(() => node.classList.add("hidden"), 2200);
}
async function api(url, options) {
  const response = await fetch(url, options), value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || response.statusText);
  return value;
}
function splitText(raw) {
  const lines = String(raw == null ? "" : raw).split("\n"), pages = [];
  let page = [], chars = 0;
  lines.forEach(line => {
    if (page.length && (page.length >= PAGE_LINES || chars + line.length > PAGE_CHARS)) {
      pages.push(page.join("\n")); page = []; chars = 0;
    }
    if (line.length > PAGE_CHARS) {
      if (page.length) { pages.push(page.join("\n")); page = []; chars = 0; }
      for (let offset = 0; offset < line.length; offset += PAGE_CHARS) pages.push(line.slice(offset, offset + PAGE_CHARS));
    } else { page.push(line); chars += line.length + 1; }
  });
  if (page.length || !pages.length) pages.push(page.join("\n"));
  return pages;
}
function textHtml(raw, key) {
  const pages = splitText(raw), page = Math.min(state.textPages[key] || 0, pages.length - 1);
  const pager = pages.length > 1 ? `<div class="pager"><button class="btn text-page" data-key="${esc(key)}" data-delta="-1" ${page === 0 ? "disabled" : ""}>← 上一页</button><span>${page + 1} / ${pages.length}</span><button class="btn text-page" data-key="${esc(key)}" data-delta="1" ${page === pages.length - 1 ? "disabled" : ""}>下一页 →</button></div>` : "";
  return `<div class="text-part">${esc(pages[page])}</div>${pager}`;
}
function renderQueue() {
  const groups = [];
  state.requests.forEach(request => {
    let group = groups.find(item => item.id === request.session_id);
    if (!group) { group = {id: request.session_id, source: request.session_source, requests: []}; groups.push(group); }
    group.requests.push(request);
  });
  qs("#queue").innerHTML = groups.map(group => `<details class="session" open><summary title="${esc(group.id)}">${esc(group.id)} · ${group.requests.length} turns · ${esc(group.source)}</summary><div class="session-items">${group.requests.map(request => `<button class="request ${state.selected === request.id ? "selected" : ""}" data-id="${esc(request.id)}"><div class="request-top"><b>Turn ${request.session_turn}</b><span class="status-${esc(request.status)}">${esc(request.status)}</span></div><div class="request-task">${esc(request.task || request.request_label)}</div><div class="request-meta">${esc(request.request_label)} · ${request.message_count} msg · ${request.image_count} img${request.attempt > 1 ? ` · retry ${request.attempt}` : ""}</div></button>`).join("")}</div></details>`).join("") || '<div class="empty">No requests yet</div>';
  document.querySelectorAll(".request").forEach(node => node.onclick = () => selectRequest(node.dataset.id));
}
function renderSection(section, index) {
  if (section.type === "images") {
    const images = (section.items || []).map(item => `<div class="evidence"><img src="${esc(item.url)}" data-full="${esc(item.url)}" alt="${esc(item.label || "Image")}"><div class="caption"><b>${esc(item.label || "Image")}</b><span>${esc([item.role, item.freshness].filter(Boolean).join(" · "))}</span></div></div>`).join("");
    return `<section class="panel"><div class="panel-head">${esc(section.title || "Images")}</div><div class="panel-body"><div class="image-grid">${images}</div></div></section>`;
  }
  if (section.type === "text") {
    return `<section class="panel"><div class="panel-head">${esc(section.title || "Text")}</div><div class="panel-body text-value">${esc(section.value || "") || '<span class="parameter-description">No value.</span>'}</div></section>`;
  }
  return `<section class="panel"><div class="panel-head">${esc(section.title || "Data")}</div><div class="panel-body">${isEmpty(section.value) ? '<span class="parameter-description">No value.</span>' : `<pre class="json">${esc(pretty(section.value))}</pre>`}</div></section>`;
}
function renderAuditRecords(detail) {
  const records = detail.audit_records || [];
  const rows = records.slice().reverse().map((record, index) => {
    const successClass = record.success === true ? "success" : (record.success === false ? "failure" : "");
    const verdict = record.success === true ? "SUCCESS" : (record.success === false ? "FAILED" : (record.status || "UNKNOWN"));
    const turn = record.first_seen_turn === record.last_seen_turn ? `Turn ${record.first_seen_turn}` : `Turns ${record.first_seen_turn}–${record.last_seen_turn}`;
    return `<details class="record" ${index === 0 ? "open" : ""}><summary><span class="record-index">#${records.length - index}</span><span class="record-title">${esc(record.title || record.kind || "Record")}</span><span class="record-status ${successClass}">${esc(verdict)}</span><span class="record-turn">${esc(turn)}</span></summary><div class="record-body"><div class="record-block"><div class="form-label">Arguments</div><pre class="json">${esc(pretty(record.arguments || {}))}</pre></div><div class="record-block"><div class="form-label">Result · ${esc(record.status || "unknown")}</div><pre class="json">${esc(pretty(record.result || {}))}</pre></div></div></details>`;
  }).join("");
  return `<section class="panel record-ledger"><div class="panel-head">Tool Call 审计 · ${records.length} records</div><div>${rows || '<div class="panel-body parameter-description">当前 session 尚未出现已完成的 Tool Call；执行结果会在下一次 provider request 到达时记录。</div>'}</div></section>`;
}
function renderOperate(detail) {
  const view = detail.presentation?.view || {};
  const sections = view.sections || [], main = sections.filter(section => section.column !== "side"), side = sections.filter(section => section.column === "side");
  return `<div class="hero"><div><div class="eyebrow">${esc(view.eyebrow || detail.adapter?.label || "Request")}</div><h2>${esc(view.title || detail.model || "Manual response")}</h2></div><div class="badges">${(view.badges || []).filter(Boolean).map(value => `<span class="chip">${esc(value)}</span>`).join("")}</div></div>${(view.alerts || []).map(value => `<div class="alert">${esc(value)}</div>`).join("")}<div class="operate-grid"><div>${main.map(renderSection).join("") || '<section class="panel"><div class="panel-body parameter-description">此 adapter 没有提供额外操作视图；可切换到模型输入审计查看完整请求。</div></section>'}</div><div>${side.map(renderSection).join("")}</div></div>${renderAuditRecords(detail)}`;
}
function partHtml(part, partIndex, key) {
  if (typeof part === "string") return textHtml(part, key);
  if (!part || typeof part !== "object") return `<pre class="object-part">${esc(pretty(part))}</pre>`;
  if (part.type === "text") return `<div class="message-head"><span>Text part ${partIndex + 1}</span></div>${textHtml(part.text || "", key)}`;
  if (part.type === "image_url") {
    const url = typeof part.image_url === "object" ? part.image_url.url : part.image_url;
    return `<div class="message-head"><span>Image part ${partIndex + 1}</span><span>${esc(part.wire_url_kind || "")}</span></div><div class="image-part"><img src="${esc(url)}" data-full="${esc(url)}"></div>`;
  }
  return `<pre class="object-part">${esc(pretty(part))}</pre>`;
}
function messagePager(page, count) {
  if (count <= 1) return "";
  return `<div class="pager"><button class="btn main-page" data-delta="-1" ${page === 0 ? "disabled" : ""}>← 上一页</button><span>消息页 ${page + 1} / ${count}</span><button class="btn main-page" data-delta="1" ${page === count - 1 ? "disabled" : ""}>下一页 →</button></div>`;
}
function renderAudit(detail) {
  const audit = detail.wire_audit || {}, count = Math.max(1, Math.ceil(detail.messages.length / PAGE_MESSAGES));
  state.messagePage = Math.min(state.messagePage, count - 1);
  const start = state.messagePage * PAGE_MESSAGES, visible = detail.messages.slice(start, start + PAGE_MESSAGES), pager = messagePager(state.messagePage, count);
  const messages = visible.map((message, offset) => {
    const index = start + offset, content = Array.isArray(message.content) ? message.content.map((part, partIndex) => partHtml(part, partIndex, `m${index}p${partIndex}`)).join("") : textHtml(message.content ?? "", `m${index}`);
    return `<article class="message"><div class="message-head"><span>Message ${index + 1} · ${esc(message.role || "unknown")}</span><span>exact content</span></div>${content}</article>`;
  }).join("");
  return `<div class="audit"><div class="audit-banner"><b>完整输入校验</b><span>${esc(audit.note || "")}</span><code>${esc(audit.normalized_sha256 || "")}</code></div><div class="metrics"><div class="metric"><b>${audit.message_count || 0}</b><span>messages</span></div><div class="metric"><b>${audit.text_chars || 0}</b><span>text characters</span></div><div class="metric"><b>${audit.image_count || 0}</b><span>images</span></div><div class="metric"><b>${audit.normalized_body_bytes || 0}</b><span>normalized bytes</span></div></div>${pager}${messages}${pager}<details class="message"><summary class="message-head">Request options</summary><pre class="object-part">${esc(pretty(detail.request_options))}</pre></details>${detail.response_text ? `<article class="message"><div class="message-head">Manual response</div>${textHtml(detail.response_text, "response")}</article>` : ""}${detail.response_error ? `<article class="message"><div class="message-head">Cancelled</div>${textHtml(detail.response_error, "error")}</article>` : ""}</div>`;
}
function bindViewEvents() {
  document.querySelectorAll("img[data-full]").forEach(image => image.onclick = () => { qs("#modalImage").src = image.dataset.full; qs("#modal").classList.remove("hidden"); });
  document.querySelectorAll(".text-page").forEach(button => button.onclick = () => { state.textPages[button.dataset.key] = (state.textPages[button.dataset.key] || 0) + Number(button.dataset.delta); renderDetail(); });
  document.querySelectorAll(".main-page").forEach(button => button.onclick = () => { state.messagePage += Number(button.dataset.delta); renderDetail(); });
}
function toolForms() { return state.detail?.presentation?.composer?.tools || []; }
function selectedForm() { return toolForms().find(tool => tool.name === state.selectedTool); }
function displayValue(value) { return typeof value === "string" ? value : JSON.stringify(value, null, 2); }
function initialField(field) { return {mode: field.has_default ? "default" : (field.required === true ? "custom" : "omit"), value: field.has_default ? displayValue(field.default) : ""}; }
function selectTool(name) {
  state.selectedTool = name; state.parameters = {};
  (selectedForm()?.fields || []).forEach(field => state.parameters[field.name] = initialField(field));
  renderComposer();
}
function parseField(field, raw) {
  const value = String(raw ?? "").trim();
  if (field.value_type === "json") { try { return {value: JSON.parse(value)}; } catch (_) { return {error: `${field.name} 必须是合法 JSON`}; } }
  if (field.value_type === "integer") return /^-?\d+$/.test(value) ? {value: Number(value)} : {error: `${field.name} 必须是整数`};
  if (field.value_type === "number") return value !== "" && Number.isFinite(Number(value)) ? {value: Number(value)} : {error: `${field.name} 必须是数字`};
  if (field.value_type === "boolean") return ["true", "false"].includes(value.toLowerCase()) ? {value: value.toLowerCase() === "true"} : {error: `${field.name} 必须是 true 或 false`};
  return {value};
}
function modeOptions(field, current) {
  const options = [];
  if (field.required !== true) options.push(["omit", "省略（不发送）"]);
  options.push(["custom", "填写值"]);
  if (field.has_default) options.push(["default", "使用默认值"]);
  return options.map(([value, label]) => `<option value="${value}" ${current === value ? "selected" : ""}>${label}</option>`).join("");
}
function fieldInput(field, current) {
  if (current.mode === "omit") return '<div class="parameter-description">此参数不会被发送。</div>';
  if (current.mode === "default") return `<pre class="json">${esc(displayValue(field.default))}</pre>`;
  if (field.choices?.length) return `<select class="form-select" data-value="${esc(field.name)}"><option value="">请选择…</option>${field.choices.map(choice => `<option value="${esc(choice)}" ${String(current.value) === String(choice) ? "selected" : ""}>${esc(choice)}</option>`).join("")}</select>`;
  if (field.value_type === "boolean") return `<select class="form-select" data-value="${esc(field.name)}"><option value="">请选择…</option><option value="true" ${current.value === "true" ? "selected" : ""}>true</option><option value="false" ${current.value === "false" ? "selected" : ""}>false</option></select>`;
  if (field.value_type === "json" || field.name === "code") return `<textarea class="form-textarea" data-value="${esc(field.name)}">${esc(current.value)}</textarea>`;
  const type = ["integer", "number"].includes(field.value_type) ? "number" : "text", step = field.value_type === "integer" ? "1" : "any";
  return `<input class="form-input" type="${type}" step="${step}" data-value="${esc(field.name)}" value="${esc(current.value)}">`;
}
function parameterHtml(field) {
  const current = state.parameters[field.name] || initialField(field);
  return `<div class="parameter"><div><div class="parameter-key">${esc(field.name)}</div><div class="parameter-description">${esc(field.description || "No description")}</div><span class="chip">${field.required === true ? "必填" : "可选"}</span> <span class="chip">${esc(field.value_type)}</span></div><div><label class="form-label">处理方式</label><select class="form-select" data-mode="${esc(field.name)}">${modeOptions(field, current.mode)}</select></div><div><label class="form-label">Value</label>${fieldInput(field, current)}</div></div>`;
}
function renderToolComposer(composer) {
  const forms = composer.tools || [];
  if (!state.selectedTool && forms.length) {
    state.selectedTool = forms[0].name;
    (forms[0].fields || []).forEach(field => state.parameters[field.name] = initialField(field));
  }
  const form = selectedForm();
  const query = state.toolQuery.trim().toLowerCase();
  const shown = forms.filter(tool => !query || JSON.stringify(tool).toLowerCase().includes(query));
  return `<div class="tool-picker"><div class="tool-sidebar"><div class="tool-search-wrap"><input id="toolSearch" class="tool-search" value="${esc(state.toolQuery)}" placeholder="搜索工具名、描述或参数…" autocomplete="off"><span class="tool-count">${shown.length}/${forms.length}</span></div><div class="tool-list">${shown.map(tool => `<button class="tool-button ${tool.name === state.selectedTool ? "active" : ""}" data-tool="${esc(tool.name)}"><div class="tool-name">${esc(tool.name)}</div><div class="tool-description">${esc(tool.description || "")}</div></button>`).join("") || '<div class="parameter-description">没有匹配的工具。</div>'}</div></div><div class="tool-form">${form ? `<b>${esc(form.name)}</b>${(form.fields || []).map(parameterHtml).join("") || '<div class="parameter-description">此工具没有参数。</div>'}<div class="reasoning"><label class="form-label">Reasoning（可留空）</label><input id="reasoning" class="form-input" value="${esc(state.reasoning)}"></div>` : '<div class="parameter-description">请选择工具。</div>'}</div></div>`;
}
function renderActionComposer(composer) {
  const actions = composer.actions || [];
  if (!state.selectedAction && actions.length) state.selectedAction = actions[0].name;
  const action = actions.find(item => item.name === state.selectedAction);
  return `<div class="actions"><div><label class="form-label">Action</label><select id="actionSelect" class="form-select">${actions.map(item => `<option value="${esc(item.name)}" ${item.name === state.selectedAction ? "selected" : ""}>${esc(item.label || item.name)}</option>`).join("")}</select></div><div>${action?.message ? `<label class="form-label">Message</label><input id="actionMessage" class="form-input" value="${esc(state.actionMessage)}">` : '<div class="parameter-description">这个 action 不需要参数。</div>'}<div class="reasoning"><label class="form-label">Reasoning（可留空）</label><input id="reasoning" class="form-input" value="${esc(state.reasoning)}"></div></div></div>`;
}
function bindComposer() {
  document.querySelectorAll("[data-tool]").forEach(button => button.onclick = () => selectTool(button.dataset.tool));
  const search = qs("#toolSearch");
  if (search) search.oninput = () => {
    state.toolQuery = search.value;
    renderComposer();
    const next = qs("#toolSearch");
    if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
  };
  document.querySelectorAll("[data-mode]").forEach(select => select.onchange = () => { state.parameters[select.dataset.mode].mode = select.value; renderComposer(); });
  document.querySelectorAll("[data-value]").forEach(input => input.oninput = input.onchange = () => state.parameters[input.dataset.value].value = input.value);
  const reasoning = qs("#reasoning"); if (reasoning) reasoning.oninput = () => state.reasoning = reasoning.value;
  const action = qs("#actionSelect"); if (action) action.onchange = () => { state.selectedAction = action.value; state.actionMessage = ""; renderComposer(); };
  const message = qs("#actionMessage"); if (message) message.oninput = () => state.actionMessage = message.value;
}
function renderComposer() {
  const composer = state.detail?.presentation?.composer || {kind: "raw"}, body = qs("#composerBody"), raw = qs("#rawResponse"), tabs = qs("#composerTabs");
  qs("#mode").textContent = `${state.detail?.adapter?.label || "Generic"} · ${composer.label || "Response"}`;
  body.innerHTML = ""; raw.classList.add("hidden"); tabs.classList.add("hidden");
  if (composer.kind === "raw") {
    raw.classList.remove("hidden"); raw.placeholder = composer.placeholder || "Enter the complete assistant content.";
    return;
  }
  const choices = [["tool", "调用工具"], ["action", "其他响应"]];
  if (composer.allow_raw) choices.push(["raw", "原始响应"]);
  tabs.innerHTML = choices.map(([value, label]) => `<button class="tab ${state.composerMode === value ? "active" : ""}" data-compose="${value}">${label}</button>`).join("");
  tabs.classList.remove("hidden");
  document.querySelectorAll("[data-compose]").forEach(button => button.onclick = () => { state.composerMode = button.dataset.compose; renderComposer(); });
  if (state.composerMode === "raw") { raw.classList.remove("hidden"); raw.placeholder = "Enter the exact protocol response."; return; }
  body.innerHTML = state.composerMode === "tool" ? renderToolComposer(composer) : renderActionComposer(composer);
  bindComposer();
}
function buildSubmission() {
  const composer = state.detail?.presentation?.composer || {kind: "raw"};
  if (composer.kind === "raw" || state.composerMode === "raw") {
    const content = qs("#rawResponse").value.trim();
    return content ? {content} : {error: "请输入完整响应"};
  }
  if (state.composerMode === "action") {
    const action = (composer.actions || []).find(item => item.name === state.selectedAction);
    if (!action) return {error: "请选择响应类型"};
    if (action.message && !state.actionMessage.trim()) return {error: "请输入 message"};
    return {intent: {type: "action", name: action.name, arguments: action.message ? {message: state.actionMessage.trim()} : {}, reasoning: state.reasoning.trim()}};
  }
  const form = selectedForm(); if (!form) return {error: "请选择工具"};
  const values = {}, errors = [];
  (form.fields || []).forEach(field => {
    const current = state.parameters[field.name] || initialField(field);
    if (current.mode === "omit") { if (field.required === true) errors.push(`${field.name} 是必填参数`); return; }
    if (current.mode === "default") { if (field.has_default) values[field.name] = field.default; else errors.push(`${field.name} 没有 default`); return; }
    if (String(current.value || "").trim() === "") { if (field.required === true) errors.push(`${field.name} 是必填参数`); return; }
    const parsed = parseField(field, current.value); if (parsed.error) errors.push(parsed.error); else values[field.name] = parsed.value;
  });
  return errors.length ? {error: errors[0]} : {intent: {type: "tool_call", name: form.name, arguments: values, reasoning: state.reasoning.trim()}};
}
function renderDetail() {
  const detail = state.detail, viewer = qs("#viewer"), composer = qs("#composer");
  document.querySelectorAll(".view-tab").forEach(tab => tab.classList.toggle("active", tab.dataset.view === state.view));
  if (!detail) { viewer.innerHTML = '<div class="empty">选择一个请求。</div>'; composer.classList.add("hidden"); return; }
  const view = detail.presentation?.view || {};
  qs("#title").textContent = `${view.title || detail.model || detail.request_label} · Turn ${detail.session_turn}`;
  qs("#subtitle").textContent = `${detail.adapter?.label || "Generic"} · ${detail.request_label} · ${detail.status}`;
  qs("#adapterLabel").textContent = `${detail.adapter?.label || "Generic"} protocol adapter`;
  qs("#rawLink").href = `/api/requests/${detail.id}/raw`; qs("#rawLink").classList.remove("hidden");
  viewer.innerHTML = state.view === "audit" ? renderAudit(detail) : renderOperate(detail); bindViewEvents();
  if (detail.status === "pending") { composer.classList.remove("hidden"); renderComposer(); } else composer.classList.add("hidden");
}
async function selectRequest(id) {
  state.selected = id; state.messagePage = 0; state.textPages = {}; state.composerMode = "tool"; state.selectedTool = ""; state.toolQuery = ""; state.parameters = {}; state.reasoning = ""; state.selectedAction = ""; state.actionMessage = ""; qs("#rawResponse").value = "";
  renderQueue(); state.detail = await api(`/api/requests/${id}`); renderDetail();
}
async function poll() {
  try {
    const listing = await api("/api/requests"), before = state.requests.map(item => `${item.id}:${item.status}`).join();
    state.requests = listing.requests; renderQueue();
    if (!state.selected) { const first = state.requests.find(item => item.status === "pending") || state.requests[0]; if (first) await selectRequest(first.id); }
    else if (before !== state.requests.map(item => `${item.id}:${item.status}`).join()) {
      state.detail = await api(`/api/requests/${state.selected}`); renderDetail();
      const pending = state.requests.find(item => item.status === "pending"); if (state.detail.status !== "pending" && pending) await selectRequest(pending.id);
    }
  } catch (error) { console.error(error); } finally { setTimeout(poll, 750); }
}
document.querySelectorAll(".view-tab").forEach(tab => tab.onclick = () => { state.view = tab.dataset.view; localStorage.setItem("manual-vlm-view", state.view); renderDetail(); });
qs("#submit").onclick = async () => {
  try { const submission = buildSubmission(); if (submission.error) return toast(submission.error); await api(`/api/requests/${state.selected}/response`, {method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(submission)}); qs("#rawResponse").value = ""; toast("Response sent"); } catch (error) { toast(error.message); }
};
qs("#cancel").onclick = async () => { if (!confirm("Cancel this waiting request?")) return; try { await api(`/api/requests/${state.selected}/cancel`, {method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({reason: "Cancelled by the human operator."})}); toast("Request cancelled"); } catch (error) { toast(error.message); } };
qs("#composer").onkeydown = event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); qs("#submit").click(); } };
qs("#closeModal").onclick = () => qs("#modal").classList.add("hidden");
qs("#modal").onclick = event => { if (event.target.id === "modal" || event.target.classList.contains("modal-stage")) qs("#modal").classList.add("hidden"); };
renderDetail(); poll();
