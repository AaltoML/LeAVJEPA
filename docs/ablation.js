"use strict";
(() => {
  const root = document.getElementById("ablation-chart");
  if (!root) return;
  // VGGSound top-1 (%) with frozen linear and attentive probes, as plotted by
  // ml-thesis/papers/iclr2027/plot_component_ablation.py (values rounded as in the paper figure).
  const views = [
    { name: "No local views", short: "None", sub: "K = 0", lin: 10.3, att: 23.2 },
    { name: "Joint local", short: "Joint", sub: "audio + video", subShort: "A + V", lin: 13.7, att: 24.7 },
    { name: "Masked local", short: "Masked", sub: "A + V, masked", subShort: "A + V", lin: 23.1, att: 34.1 },
    { name: "Random dropout", short: "Random", sub: "p = 0.5", lin: 38.1, att: 41.3 },
    { name: "Modality-specific", short: "Specific", sub: "LeAVJEPA", lin: 43.9, att: 47.1, ours: true },
  ];
  const losses = [
    { name: "Invariance only", short: "Invariance", sub: "λ = 0", lin: 1.2, att: 1.1 },
    { name: "SIGReg only", short: "SIGReg", sub: "λ = 1", lin: 0.3, att: 1.2 },
    { name: "Both", short: "Both", sub: "λ = 0.05", lin: 43.9, att: 47.1, ours: true },
  ];
  // Audio-only pretraining baselines (AUDIO_ONLY in the plotting script).
  const audioOnly = { lin: 14.07, att: 22.5 };
  const max = 50, ticks = [0, 10, 20, 30, 40, 50];
  const fmt = v => v.toFixed(1);
  const text = (x, y, value, attrs = "") => `<text x="${x}" y="${y}" ${attrs}>${value}</text>`;

  const bars = (cx, d, Y, bw, fs) => {
    const w = Math.min(18, bw * .3), g = Math.max(1, Math.min(4, bw * .3 - w)), y0 = Y(0);
    const bar = (x, v, cls) => { const h = Math.max(1.5, y0 - Y(v)); return `<rect x="${x}" y="${y0 - h}" width="${w}" height="${h}" class="${cls}"/>`; };
    return bar(cx - w - g / 2, d.lin, "ab-lin") + bar(cx + g / 2, d.att, "ab-att") +
      text(cx - w / 2 - g / 2, Y(d.lin) - 5, fmt(d.lin), `text-anchor="middle" font-size="${fs}" class="ab-t2"`) +
      text(cx + w / 2 + g / 2, Y(d.att) - 5, fmt(d.att), `text-anchor="middle" font-size="${fs}" class="ab-t1"`);
  };

  const render = () => {
    // Narrow screens stack the two panels so each gets the full width for its x labels.
    const W = Math.max(300, root.clientWidth), narrow = W < 560;
    const m = { t: 26, b: narrow ? 44 : 50, l: 30, r: 4 }, gap = 28;
    let panels, H;
    if (narrow) {
      const pw = W - m.l - m.r, h1 = 220, h2 = 130, top2 = m.t + h1 + m.b + 34;
      panels = [{ x0: m.l, w: pw, top: m.t, h: h1, data: views, line: true, title: "Local-view design" },
                { x0: m.l, w: pw, top: top2, h: h2, data: losses, title: "Loss terms" }];
      H = top2 + h2 + m.b;
    } else {
      const pw = W - m.l - m.r - gap, w1 = pw * .72, h = 320 - m.t - m.b;
      panels = [{ x0: m.l, w: w1, top: m.t, h, data: views, line: true, title: "Local-view design" },
                { x0: m.l + w1 + gap, w: pw - w1, top: m.t, h, data: losses, title: "Loss terms" }];
      H = 320;
    }
    let s = "";
    panels.forEach((p, pi) => {
      const Y = v => p.top + p.h * (1 - v / max), yb = p.top + p.h;
      const bw = p.w / p.data.length, X = i => p.x0 + bw * (i + .5), fs = narrow ? 10.5 : 12, short = narrow || bw < 95;
      s += text(p.x0, p.top - 15, p.title, 'font-size="12" font-weight="700" class="ab-t2"');
      if (narrow || pi === 0) ticks.forEach(t => s += text(m.l - 7, Y(t) + 4, t, 'text-anchor="end" font-size="11" class="ab-t3"'));
      ticks.forEach(t => s += `<line x1="${p.x0}" x2="${p.x0 + p.w}" y1="${Y(t)}" y2="${Y(t)}" class="${t ? "ab-grid" : "ab-axis"}"/>`);
      p.data.forEach((d, i) => {
        s += text(X(i), yb + 18, short ? d.short : d.name, `text-anchor="middle" font-size="${fs}" font-weight="${d.ours ? 700 : 400}" class="ab-t1"`);
        s += text(X(i), yb + 33, bw < 72 && d.subShort ? d.subShort : d.sub, `text-anchor="middle" font-size="${fs - 1}" class="ab-t3"`);
      });
      if (!p.line) { s += p.data.map((d, i) => bars(X(i), d, Y, bw, narrow ? 9.5 : 11)).join(""); return; }
      ["lin", "att"].forEach(k => {
        s += `<line x1="${p.x0}" x2="${p.x0 + p.w}" y1="${Y(audioOnly[k])}" y2="${Y(audioOnly[k])}" class="ab-base"/>`;
        s += `<path d="M${p.data.map((d, i) => `${X(i)},${Y(d[k])}`).join("L")}" class="ab-${k}-line"/>`;
      });
      const vs = narrow ? 10.5 : 11.5;
      p.data.forEach((d, i) => {
        s += `<circle cx="${X(i)}" cy="${Y(d.lin)}" r="4.5" class="ab-lin ab-mark"/><circle cx="${X(i)}" cy="${Y(d.att)}" r="4.5" class="ab-att ab-mark"/>`;
        s += text(X(i), Y(d.att) - 10, fmt(d.att), `text-anchor="middle" font-size="${vs}" class="ab-t1 ab-halo"`);
        s += text(X(i), Y(d.lin) + 18, fmt(d.lin), `text-anchor="middle" font-size="${vs}" class="ab-t2 ab-halo"`);
      });
      const lx = p.x0 + 10, ly = p.top + 8, lf = narrow ? 10.5 : 12, row = narrow ? 15 : 17;
      s += `<rect x="${lx - 6}" y="${ly - 5}" width="${narrow ? 128 : 150}" height="${row * 2 + 4}" class="ab-legend-bg"/>`;
      [["Attentive probe", `<rect x="${lx}" y="${ly}" width="10" height="10" rx="2" class="ab-att"/>`],
       ["Linear probe", `<rect x="${lx}" y="${ly + row}" width="10" height="10" rx="2" class="ab-lin"/>`]]
        .forEach(([label, key], i) => s += key + text(lx + 18, ly + row * i + 9.5, label, `font-size="${lf}" class="ab-t2 ab-halo"`));
      s += text(p.x0 + p.w - 4, Y(audioOnly.lin) + 15, "Audio-only pretraining", `text-anchor="end" font-size="${narrow ? 10 : 11}" class="ab-t3 ab-halo"`);
    });
    root.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="VGGSound top-1 accuracy. Local-view design, attentive and linear probe: no local views 23.2 and 10.3; joint local views 24.7 and 13.7; masked local views 34.1 and 23.1; random modality dropout 41.3 and 38.1; modality-specific views (LeAVJEPA) 47.1 and 43.9. Loss terms: invariance only 1.1 and 1.2; SIGReg only 1.2 and 0.3; both 47.1 and 43.9.">${s}</svg>`;
  };

  let width = 0;
  new ResizeObserver(() => { if (root.clientWidth !== width) { width = root.clientWidth; render(); } }).observe(root);
  render();
})();
