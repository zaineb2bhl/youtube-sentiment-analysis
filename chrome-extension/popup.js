// Logique du popup : extraction -> appel API par batch -> affichage.
const $ = (id) => document.getElementById(id);
const LABELS_FR = { positive: "Positif", neutral: "Neutre", negative: "Négatif" };

const state = {
  results: [],       // [{text, author, isReply, sentiment, confidence, ...}]
  stats: null,
  filter: "all",
  title: "",
  apiTimeMs: 0,
};

const settings = {
  theme: "light",
  apiEnv: "prod",
  customUrl: "",
  maxComments: 100,
  autoScroll: true,
  includeReplies: false,
};

// ---------- Paramètres & thème ----------
function apiUrl() {
  if (settings.apiEnv === "local") return CONFIG.LOCAL_API_URL;
  if (settings.apiEnv === "custom" && settings.customUrl) return settings.customUrl.replace(/\/+$/, "");
  return CONFIG.PROD_API_URL;
}

async function loadSettings() {
  const saved = await chrome.storage.local.get("settings");
  Object.assign(settings, saved.settings || {});
  $("api-env").value = settings.apiEnv;
  $("api-url").value = settings.customUrl;
  $("api-url").classList.toggle("hidden", settings.apiEnv !== "custom");
  $("max-comments").value = settings.maxComments;
  $("auto-scroll").checked = settings.autoScroll;
  $("include-replies").checked = settings.includeReplies;
  applyTheme();
}

function saveSettings() {
  settings.apiEnv = $("api-env").value;
  settings.customUrl = $("api-url").value.trim();
  settings.maxComments = Math.min(1000, Math.max(10, parseInt($("max-comments").value, 10) || 100));
  settings.autoScroll = $("auto-scroll").checked;
  settings.includeReplies = $("include-replies").checked;
  $("api-url").classList.toggle("hidden", settings.apiEnv !== "custom");
  chrome.storage.local.set({ settings });
}

function applyTheme() {
  document.body.classList.toggle("dark", settings.theme === "dark");
  $("theme-toggle").textContent = settings.theme === "dark" ? "☀" : "🌙";
}

// ---------- Utilitaires ----------
function show(id, visible) { $(id).classList.toggle("hidden", !visible); }

function toast(msg) {
  $("toast").textContent = msg;
  show("toast", true);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => show("toast", false), 1800);
}

function showError(msg) {
  $("error").textContent = msg;
  show("error", true);
}

async function fetchWithTimeout(url, options = {}, timeout = CONFIG.REQUEST_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

// ---------- API ----------
async function checkHealth() {
  const el = $("api-status");
  el.className = "status unknown";
  el.textContent = "API…";
  try {
    const r = await fetchWithTimeout(`${apiUrl()}/health`, {}, 8000);
    const data = await r.json();
    const ok = r.ok && data.model_loaded;
    el.className = `status ${ok ? "ok" : "down"}`;
    el.textContent = ok ? `API en ligne (${settings.apiEnv})` : "API dégradée";
    return ok;
  } catch {
    el.className = "status down";
    el.textContent = `API injoignable (${settings.apiEnv})`;
    return false;
  }
}

async function predictAll(comments) {
  const texts = comments.map((c) => c.text);
  const predictions = [];
  let apiTime = 0;
  for (let i = 0; i < texts.length; i += CONFIG.BATCH_SIZE) {
    const batch = texts.slice(i, i + CONFIG.BATCH_SIZE);
    $("loading-text").textContent =
      `Analyse ${Math.min(i + batch.length, texts.length)}/${texts.length} commentaires…`;
    let r;
    try {
      r = await fetchWithTimeout(`${apiUrl()}/predict_batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments: batch }),
      });
    } catch (e) {
      throw new Error(e.name === "AbortError"
        ? "Délai dépassé : l'API ne répond pas (vérifiez que le Codespace est allumé, que le conteneur tourne et que le port 7860 est public)."
        : `Impossible de joindre l'API (${apiUrl()}).\nVérifiez qu'elle est démarrée et votre connexion.`);
    }
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(`Erreur API ${r.status} : ${JSON.stringify(detail.detail || detail)}`);
    }
    const data = await r.json();
    apiTime += data.processing_time_ms;
    predictions.push(...data.predictions);
  }
  state.apiTimeMs = apiTime;
  return comments.map((c, i) => ({ ...c, ...predictions[i] }));
}

function computeStats(results) {
  const counts = { positive: 0, neutral: 0, negative: 0 };
  let conf = 0, score = 0;
  results.forEach((r) => { counts[r.sentiment]++; conf += r.confidence; score += r.label; });
  const n = results.length || 1;
  const pct = Object.fromEntries(Object.entries(counts).map(([k, v]) => [k, (100 * v) / n]));
  const avg = score / n;
  return {
    total: results.length, counts, pct,
    avgConfidence: conf / n,
    overall: avg > 0.15 ? "positive" : avg < -0.15 ? "negative" : "neutral",
  };
}

// ---------- Extraction ----------
async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { action: "ping" });
  } catch {
    // Page ouverte avant l'installation de l'extension : on injecte le script.
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  }
}

// ---------- Rendu ----------
function renderDonut(pct) {
  const svg = $("donut");
  const colors = {
    positive: "var(--positive)", neutral: "var(--neutral)", negative: "var(--negative)",
  };
  const r = 15.915; // circonférence = 100
  let offset = 25;  // départ en haut
  let html = `<circle cx="21" cy="21" r="${r}" fill="none" stroke="var(--border)" stroke-width="5"/>`;
  for (const k of ["positive", "neutral", "negative"]) {
    const v = pct[k];
    if (v <= 0) continue;
    html += `<circle cx="21" cy="21" r="${r}" fill="none" stroke="${colors[k]}" stroke-width="5"
      stroke-dasharray="${v} ${100 - v}" stroke-dashoffset="${offset}"><title>${LABELS_FR[k]} ${v.toFixed(1)}%</title></circle>`;
    offset -= v;
  }
  const main = Object.entries(pct).sort((a, b) => b[1] - a[1])[0];
  html += `<text x="21" y="21" text-anchor="middle" font-size="6" font-weight="700">${Math.round(main[1])}%</text>
    <text x="21" y="27" text-anchor="middle" font-size="3.2">${LABELS_FR[main[0]]}</text>`;
  svg.innerHTML = html;
}

function renderSummary() {
  const s = state.stats;
  $("video-title").textContent = state.title;
  $("video-title").title = state.title;
  $("overall").textContent = LABELS_FR[s.overall];
  $("overall").style.color = `var(--${s.overall})`;
  for (const k of ["positive", "neutral", "negative"]) {
    $(`pct-${k}`).textContent = `${s.pct[k].toFixed(1)}% (${s.counts[k]})`;
  }
  $("meta").textContent =
    `${s.total} commentaires · confiance moy. ${(s.avgConfidence * 100).toFixed(0)}% · modèle ${state.apiTimeMs.toFixed(0)} ms · total ${state.totalMs} ms`;
  const spans = $("bar").children;
  spans[0].style.width = `${s.pct.positive}%`;
  spans[1].style.width = `${s.pct.neutral}%`;
  spans[2].style.width = `${s.pct.negative}%`;
  renderDonut(s.pct);
}

function visibleResults() {
  const q = $("search").value.trim().toLowerCase();
  let list = state.results.filter(
    (r) => (state.filter === "all" || r.sentiment === state.filter) &&
           (!q || r.text.toLowerCase().includes(q) || (r.author || "").toLowerCase().includes(q)));
  const sort = $("sort").value;
  if (sort === "conf-desc") list = [...list].sort((a, b) => b.confidence - a.confidence);
  if (sort === "conf-asc") list = [...list].sort((a, b) => a.confidence - b.confidence);
  return list;
}

function renderList() {
  const ul = $("comment-list");
  ul.replaceChildren();
  const list = visibleResults();
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Aucun commentaire pour ce filtre.";
    ul.appendChild(li);
    return;
  }
  // DocumentFragment : un seul reflow même avec des centaines de commentaires
  const frag = document.createDocumentFragment();
  for (const r of list) {
    const li = document.createElement("li");
    li.className = `${r.sentiment}${r.isReply ? " reply" : ""}`;
    const head = document.createElement("div");
    head.className = "c-head";
    const author = document.createElement("span");
    author.textContent = (r.isReply ? "↳ " : "") + (r.author || "Anonyme");
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `${LABELS_FR[r.sentiment]} · ${(r.confidence * 100).toFixed(0)}%`;
    head.append(author, badge);
    const text = document.createElement("div");
    text.className = "c-text";
    text.textContent = r.text; // textContent : pas d'injection HTML
    text.title = r.text;
    const conf = document.createElement("div");
    conf.className = "conf";
    const bar = document.createElement("span");
    bar.style.width = `${r.confidence * 100}%`;
    conf.appendChild(bar);
    li.append(head, text, conf);
    frag.appendChild(li);
  }
  ul.appendChild(frag);
}

// ---------- Export ----------
function resultsAsText() {
  const s = state.stats;
  const lines = [
    `Analyse de sentiment — ${state.title}`,
    `Total : ${s.total} | Positif ${s.pct.positive.toFixed(1)}% | Neutre ${s.pct.neutral.toFixed(1)}% | Négatif ${s.pct.negative.toFixed(1)}%`,
    `Sentiment global : ${LABELS_FR[s.overall]}`,
    "",
  ];
  visibleResults().forEach((r) =>
    lines.push(`[${LABELS_FR[r.sentiment]} ${(r.confidence * 100).toFixed(0)}%] ${r.author ? r.author + " : " : ""}${r.text.replace(/\s+/g, " ")}`));
  return lines.join("\n");
}

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportCsv() {
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [["author", "text", "sentiment", "label", "confidence", "is_reply"].join(",")];
  visibleResults().forEach((r) =>
    rows.push([r.author, r.text, r.sentiment, r.label, r.confidence, r.isReply].map(esc).join(",")));
  download("youtube_sentiment.csv", "﻿" + rows.join("\n"), "text/csv;charset=utf-8");
}

function exportJson() {
  download("youtube_sentiment.json",
    JSON.stringify({ title: state.title, stats: state.stats, results: visibleResults() }, null, 2),
    "application/json");
}

// ---------- Flux principal ----------
async function analyze() {
  show("error", false);
  show("results", false);
  $("analyze").disabled = true;
  show("loading", true);
  $("loading-text").textContent = "Extraction des commentaires…";
  const t0 = performance.now();
  try {
    const tab = await getActiveTab();
    if (!tab?.url || !/^https:\/\/www\.youtube\.com\/(watch|shorts)/.test(tab.url)) {
      throw new Error("Ouvrez une vidéo YouTube (youtube.com/watch?v=…) puis relancez l'analyse.");
    }
    await ensureContentScript(tab.id);
    if (settings.autoScroll) $("loading-text").textContent = "Chargement des commentaires (défilement)…";
    const res = await chrome.tabs.sendMessage(tab.id, {
      action: "getComments", max: settings.maxComments,
      autoScroll: settings.autoScroll, includeReplies: settings.includeReplies,
    });
    if (!res?.ok) throw new Error(res?.error || "Échec de l'extraction des commentaires.");
    if (!res.comments.length) {
      throw new Error("Aucun commentaire trouvé.\nFaites défiler jusqu'à la section commentaires (ou activez le défilement automatique) puis réessayez.");
    }
    state.title = res.title;
    state.results = await predictAll(res.comments);
    state.stats = computeStats(state.results);
    state.totalMs = Math.round(performance.now() - t0);
    renderSummary();
    renderList();
    show("results", true);
    chrome.storage.local.set({ lastResults: { url: tab.url, ...state } });
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    show("loading", false);
    $("analyze").disabled = false;
  }
}

async function restoreLast() {
  const tab = await getActiveTab();
  const { lastResults } = await chrome.storage.local.get("lastResults");
  if (lastResults && tab?.url === lastResults.url) {
    Object.assign(state, lastResults, { filter: "all" });
    renderSummary();
    renderList();
    show("results", true);
  }
}

// ---------- Événements ----------
document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  checkHealth();
  restoreLast();

  $("analyze").addEventListener("click", analyze);
  $("theme-toggle").addEventListener("click", () => {
    settings.theme = settings.theme === "dark" ? "light" : "dark";
    applyTheme();
    chrome.storage.local.set({ settings });
  });
  $("settings-toggle").addEventListener("click", () => $("settings").classList.toggle("hidden"));
  ["api-env", "api-url"].forEach((id) =>
    $(id).addEventListener("change", () => { saveSettings(); checkHealth(); }));
  ["max-comments", "auto-scroll", "include-replies"].forEach((id) =>
    $(id).addEventListener("change", saveSettings));

  document.querySelectorAll(".filter").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      renderList();
    }));
  $("search").addEventListener("input", renderList);
  $("sort").addEventListener("change", renderList);

  $("copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(resultsAsText());
      toast("Résultats copiés ✔");
    } catch {
      toast("Copie impossible");
    }
  });
  $("export-csv").addEventListener("click", exportCsv);
  $("export-json").addEventListener("click", exportJson);
});
