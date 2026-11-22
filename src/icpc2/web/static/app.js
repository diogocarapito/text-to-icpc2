// All text comes from src/icpc2/web/content/{pt,en}.md (rendered into #i18n by the server).
const I18N = JSON.parse(document.getElementById("i18n").textContent);
const $ = (s) => document.querySelector(s);
let lang = "pt", current = null, codes = [], health = null;
const t = (k, fmt = {}) =>
  (I18N[lang]?.[k] ?? I18N.pt[k] ?? k).replace(/\{(\w+)\}/g, (_, n) => fmt[n] ?? "");

function setLang(l) {
  lang = l;
  document.documentElement.lang = l;
  document.querySelectorAll(".lang button").forEach((b) => b.classList.toggle("on", b.dataset.lang === l));
  document.querySelectorAll("[data-i18n]").forEach((e) => (e.textContent = t(e.dataset.i18n)));
  document.querySelectorAll("[data-i18n-ph]").forEach((e) => (e.placeholder = t(e.dataset.i18nPh)));
  document.title = t("title");
  renderModel();
  $("#theme").title = t("theme"); $("#theme").setAttribute("aria-label", t("theme"));
  try { localStorage.setItem("lang", l); } catch {}
  if (current) render(current);
}
document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
try { const l = localStorage.getItem("lang"); if (l) setLang(l); } catch {}

// Light/dark toggle: starts from the system setting, then remembers the choice.
$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = isDark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});

async function sendFeedback(kind, code) {
  if (!current) return;
  await fetch("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: current.request_id, kind, code }),
  });
  $("#thanks").hidden = false;
}

function render(data) {
  const ol = $("#results");
  ol.innerHTML = "";
  data.predictions.forEach((p) => {
    const title = lang === "en" && p.title_en ? p.title_en : p.title_pt;
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="row">
        <span class="code">${p.code}</span>
        <span class="title"></span>
        <span class="score">${(p.score * 100).toFixed(1)}%</span>
      </div>
      <div class="bar"><span style="width:${Math.max(2, p.score * 100)}%"></span></div>
      ${p.inclusion.length ? `<div class="incl"></div>` : ""}
      <div class="actions">
        <button class="copy">${t("copy")}</button>
        <button class="ok">✓ ${t("correct")}</button>
        <button class="no">✗ ${t("wrong")}</button>
      </div>`;
    li.querySelector(".title").textContent = title;
    if (p.inclusion.length) li.querySelector(".incl").textContent = p.inclusion.join("; ");
    li.querySelector(".copy").onclick = async (ev) => {
      try { await navigator.clipboard.writeText(p.code); } catch {}
      ev.target.textContent = t("copied");
      sendFeedback("copy", p.code);
    };
    li.querySelector(".ok").onclick = () => { markChosen(li); sendFeedback("correct", p.code); };
    li.querySelector(".no").onclick = () => { li.classList.add("bad"); sendFeedback("wrong", p.code); showFix(); };
    ol.appendChild(li);
  });
  const ms = data.latency_ms != null ? ` · ${data.latency_ms} ms` : "";
  $("#meta").textContent = (data.model_version ? t("model", { version: data.model_version }) : t("code_lookup")) + ms;
}

function markChosen(li) {
  document.querySelectorAll("#results li").forEach((x) => x.classList.remove("chosen"));
  li.classList.add("chosen");
}

async function showFix() {
  $("#fix").hidden = false;
  if (!codes.length) {
    codes = await (await fetch("/api/codes")).json();
    $("#codes").innerHTML = codes.map((c) => `<option value="${c.code} — ${(lang === "en" && c.title_en) || c.title_pt}">`).join("");
  }
  $("#fixq").focus();
}

$("#fixgo").onclick = async () => {
  const code = ($("#fixq").value.match(/^\s*([A-Za-z]\d\d)/) || [])[1];
  if (!code) return;
  await sendFeedback("corrected", code.toUpperCase());
  $("#fix").hidden = true;
  $("#fixq").value = "";
};

$("#f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("#q").value.trim();
  if (!text) return;
  $("#thanks").hidden = true; $("#fix").hidden = true;
  const btn = $("#f button"); btn.disabled = true;
  try {
    const r = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    current = data;
    render(data);
    $("#out").hidden = false;
  } catch (e) {
    $("#out").hidden = false; $("#results").innerHTML = ""; $("#meta").textContent = e.message;
  } finally { btn.disabled = false; }
});

function renderModel() {
  if (health) $("#model").textContent = health.model ? t("model", { version: health.model }) : t("model_loading");
}
fetch("/health").then((r) => r.json()).then((h) => { health = h; renderModel(); });
