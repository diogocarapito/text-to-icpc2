// Admin dashboard: theme toggle + two column charts (per day stacked web/API, per hour).
const DATA = JSON.parse(document.getElementById("data").textContent);
const SVG = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");

document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = isDark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});

function el(name, attrs = {}, parent) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceMax(v) {
  if (v <= 4) return 4;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

// rounded top (4px), square at the baseline
function colPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function showTip(ev, rows, title) {
  tip.replaceChildren();
  const h = document.createElement("div"); h.className = "tip-h"; h.textContent = title; tip.appendChild(h);
  for (const [name, value, cls] of rows) {
    const r = document.createElement("div"); r.className = "tip-r";
    if (cls) { const k = document.createElement("i"); k.className = `key ${cls}`; r.appendChild(k); }
    const b = document.createElement("b"); b.textContent = value; r.appendChild(b);
    const s = document.createElement("span"); s.textContent = name; r.appendChild(s);
    tip.appendChild(r);
  }
  tip.hidden = false;
  const box = ev.currentTarget.getBoundingClientRect();
  const x = ev.clientX ?? box.left + box.width / 2, y = ev.clientY ?? box.top;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = `${Math.min(window.innerWidth - tw - 8, Math.max(8, x + 12))}px`;
  tip.style.top = `${Math.max(8, y - th - 12)}px`;
}
const hideTip = () => { tip.hidden = true; };

/**
 * Column chart. series: [{key, name, cls}] stacked bottom → top.
 * labelFmt(label) → axis text; titleFmt(label) → tooltip heading.
 */
function columns(node, rows, series, { labelFmt = (l) => l, titleFmt = (l) => l, height = 180 } = {}) {
  const draw = () => {
    node.replaceChildren();
    const W = node.clientWidth || 600, H = height, padL = 32, padB = 22, padT = 8;
    const cw = W - padL - 4, ch = H - padB - padT;
    const totals = rows.map((r) => series.reduce((s, k) => s + (r[k.key] || 0), 0));
    const max = niceMax(Math.max(1, ...totals));
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, node);
    // y grid + ticks
    for (let i = 0; i <= 4; i++) {
      const v = (max / 4) * i, y = padT + ch - (v / max) * ch;
      el("line", { x1: padL, x2: W - 4, y1: y, y2: y, class: i ? "grid" : "axis" }, svg);
      const t = el("text", { x: padL - 6, y: y + 3, class: "tick", "text-anchor": "end" }, svg);
      t.textContent = Number.isInteger(v) ? v : v.toFixed(1);
    }
    const band = cw / rows.length, bw = Math.max(2, Math.min(24, band * 0.7));
    const every = Math.ceil(rows.length / Math.max(1, Math.floor(cw / 44)));
    rows.forEach((r, i) => {
      const x = padL + i * band + (band - bw) / 2;
      let y0 = padT + ch;
      const parts = series.map((s) => ({ s, v: r[s.key] || 0 })).filter((p) => p.v > 0);
      parts.forEach((p, j) => {
        const h = (p.v / max) * ch, gap = j ? 2 : 0;  // 2px surface gap between stacked parts
        const top = j === parts.length - 1;
        const y = y0 - h;
        if (h - gap > 0) {
          el("path", { d: top ? colPath(x, y, bw, h - gap, 4) : `M${x},${y}h${bw}v${h - gap}h${-bw}Z`,
                       class: `bar ${p.s.cls}` }, svg);
        }
        y0 = y;
      });
      if (i % every === 0) {
        const t = el("text", { x: padL + i * band + band / 2, y: H - 6, class: "tick", "text-anchor": "middle" }, svg);
        t.textContent = labelFmt(r.label);
      }
      const hit = el("rect", { x: padL + i * band, y: padT, width: band, height: ch, class: "hit", tabindex: 0 }, svg);
      hit.setAttribute("aria-label", `${titleFmt(r.label)}: ${series.map((s) => `${s.name} ${r[s.key] || 0}`).join(", ")}`);
      const show = (ev) => {
        hit.classList.add("on");
        const lines = series.slice().reverse().map((s) => [s.name, r[s.key] || 0, s.cls]);
        if (series.length > 1) lines.push(["total", totals[i], null]);
        showTip(ev, lines, titleFmt(r.label));
      };
      hit.addEventListener("pointermove", show);
      hit.addEventListener("focus", show);
      hit.addEventListener("pointerleave", () => { hit.classList.remove("on"); hideTip(); });
      hit.addEventListener("blur", () => { hit.classList.remove("on"); hideTip(); });
    });
  };
  draw();
  new ResizeObserver(() => draw()).observe(node);
}

const fmtDay = (iso) => { const [, m, d] = iso.split("-"); return `${d}/${m}`; };
columns(document.getElementById("chart-days"), DATA.per_day,
  [{ key: "web", name: "Página web", cls: "s1" }, { key: "api", name: "API", cls: "s2" }],
  { labelFmt: fmtDay, titleFmt: (iso) => fmtDay(iso) + "/" + iso.slice(0, 4) });
columns(document.getElementById("chart-hours"), DATA.hours.map((n, h) => ({ label: h, n })),
  [{ key: "n", name: "pedidos", cls: "s0" }],
  { labelFmt: (h) => `${h}h`, titleFmt: (h) => `${h}:00–${h}:59`, height: 160 });
