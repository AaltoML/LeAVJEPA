"use strict";
(() => {
  const root = document.getElementById("embedding-plot");
  const data = window.LEAV_TSNE;
  if (!root || !data) return;
  const n = data.labels.length;
  const colors = ["#7760b5", "#c79038", "#428e83", "#c96675", "#5387b1"];
  const modes = ["Audio", "Video", "Joint"], short = ["A", "V", "AV"];
  const family = i => Math.floor(data.labels[i % n] / 6);
  const className = i => data.classes[data.labels[i % n]].replace(/^playing /, "").replace(/^./, c => c.toUpperCase());
  const visible = [false, false, true];
  let selected = -1, points = [], width = 0, height = 0;
  let k = 1, tx = 0, ty = 0;
  const maxZoom = 12;

  root.innerHTML = `
    <div class="em-stage">
      <canvas role="img" aria-label="t-SNE scatterplot of LeAVJEPA embeddings for 1,474 VGGSound clips, coloured by semantic family."></canvas>
      <div class="em-tip" hidden></div>
      <div class="em-zoom" role="group" aria-label="Zoom">
        <button type="button" data-zoom="in" aria-label="Zoom in">+</button>
        <button type="button" data-zoom="out" aria-label="Zoom out">−</button>
        <button type="button" data-zoom="reset" aria-label="Reset zoom" disabled>⟲</button>
      </div>
    </div>
    <div class="em-side">
      <div class="em-group">
        <div class="em-heading" id="em-modes-heading">Input</div>
        <div class="em-modes" role="group" aria-labelledby="em-modes-heading">${modes.map((m, i) =>
          `<button type="button" id="em-mode-${i}" data-mode="${i}" aria-pressed="${visible[i]}"><span aria-hidden="true">${"◆▲●"[i]}</span> ${m}</button>`).join("")}</div>
      </div>
      <div class="em-group">
        <div class="em-heading">Semantic family</div>
        <div class="em-legend">${data.families.map((f, i) => `<span><i style="background:${colors[i]}"></i>${f}</span>`).join("")}</div>
      </div>
    </div>`;
  const canvas = root.querySelector("canvas"), ctx = canvas.getContext("2d"), tip = root.querySelector(".em-tip");

  // Fixed bounds over all modalities, so toggling never moves points.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < 3 * n; i++) {
    const x = data.xy[2 * i], y = data.xy[2 * i + 1];
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const project = i => {
    const pad = 14, s = Math.min((width - 2 * pad) / (maxX - minX), (height - 2 * pad) / (maxY - minY));
    const x = width / 2 + (data.xy[2 * i] - (minX + maxX) / 2) * s, y = height / 2 - (data.xy[2 * i + 1] - (minY + maxY) / 2) * s;
    return [x * k + tx, y * k + ty];
  };
  // Deterministic shuffle so no class or modality is always drawn on top.
  const order = Array.from({ length: 3 * n }, (_, i) => i);
  for (let i = order.length - 1, seed = 20260930; i > 0; i--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const j = seed % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const mark = (x, y, mode, r) => {
    ctx.beginPath();
    if (mode === 2) ctx.arc(x, y, r, 0, Math.PI * 2);
    else if (mode === 1) { ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r, y + r * .8); ctx.lineTo(x - r, y + r * .8); ctx.closePath(); }
    else { ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath(); }
  };

  const draw = () => {
    ctx.clearRect(0, 0, width, height);
    points = [];
    const r = 3 * k ** .35;
    ctx.globalAlpha = .7;
    for (const i of order) {
      const mode = Math.floor(i / n);
      if (!visible[mode]) continue;
      const [x, y] = project(i);
      if (x < -r || y < -r || x > width + r || y > height + r) continue;
      points.push({ i, x, y });
      ctx.fillStyle = colors[family(i)];
      mark(x, y, mode, r);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (selected < 0) return;
    const views = [0, 1, 2].map(m => project(m * n + selected));
    ctx.strokeStyle = "#18181b"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); views.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "11px Helvetica, Arial, sans-serif";
    views.forEach(([x, y], m) => {
      ctx.fillStyle = "#fff"; mark(x, y, m, 7); ctx.fill();
      ctx.fillStyle = colors[family(selected)]; ctx.strokeStyle = "#18181b"; mark(x, y, m, 4.5); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#18181b"; ctx.fillText(short[m], x + 9, y - 7);
    });
  };

  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (w === width && h === height) return;
    width = w; height = h; k = 1; tx = ty = 0; zoomed();
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  };
  const nearest = event => {
    const r = canvas.getBoundingClientRect(), x = event.clientX - r.left, y = event.clientY - r.top;
    let hit = null, best = 100;
    for (const p of points) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < best) { best = d; hit = p; } }
    return hit;
  };

  // Zoom about a point in canvas pixels; the view never leaves the plot bounds.
  const zoomed = () => {
    root.querySelector('[data-zoom="reset"]').disabled = k === 1;
    root.querySelector('[data-zoom="in"]').disabled = k >= maxZoom;
    root.querySelector('[data-zoom="out"]').disabled = k === 1;
    canvas.classList.toggle("em-zoomed", k > 1);
  };
  const clampView = () => {
    tx = Math.min(0, Math.max(width * (1 - k), tx));
    ty = Math.min(0, Math.max(height * (1 - k), ty));
  };
  const zoomAt = (factor, cx, cy) => {
    const next = Math.min(maxZoom, Math.max(1, k * factor));
    tx = cx - (cx - tx) * next / k; ty = cy - (cy - ty) * next / k; k = next;
    clampView(); zoomed(); tip.hidden = true; draw();
  };
  const local = event => { const r = canvas.getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top]; };
  canvas.addEventListener("wheel", event => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    zoomAt(Math.exp(-event.deltaY * .01), ...local(event));
  }, { passive: false });
  root.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
    const z = button.dataset.zoom;
    if (z === "reset") { k = 1; tx = ty = 0; zoomed(); draw(); }
    else zoomAt(z === "in" ? 1.6 : 1 / 1.6, width / 2, height / 2);
  }));
  const pointers = new Map();
  let drag = null, pinch = null, moved = false;
  canvas.addEventListener("pointerdown", event => {
    pointers.set(event.pointerId, local(event));
    moved = false;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), k };
      drag = null;
    } else if (k > 1) {
      drag = { x: event.clientX, y: event.clientY, tx, ty };
      canvas.setPointerCapture(event.pointerId);
    }
  });
  const release = event => {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinch = null;
    if (!pointers.size) drag = null;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointermove", event => {
    if (pointers.has(event.pointerId)) pointers.set(event.pointerId, local(event));
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      moved = true;
      zoomAt(pinch.k * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch.d / k, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      return;
    }
    if (drag) {
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (!moved && Math.hypot(dx, dy) < 4) return;
      moved = true;
      tx = drag.tx + dx; ty = drag.ty + dy;
      clampView(); tip.hidden = true; draw();
      return;
    }
    const hit = nearest(event);
    canvas.style.cursor = hit ? "pointer" : "default";
    if (!hit) { tip.hidden = true; return; }
    tip.textContent = `${className(hit.i)} · ${modes[Math.floor(hit.i / n)]}`;
    tip.hidden = false;
    tip.style.left = Math.min(hit.x + 10, width - tip.offsetWidth) + "px";
    tip.style.top = Math.max(hit.y - 30, 0) + "px";
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", event => {
    if (moved) { moved = false; return; }
    const hit = nearest(event);
    selected = hit ? hit.i % n : -1;
    draw();
  });
  root.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
    const m = +button.dataset.mode;
    visible[m] = !visible[m];
    button.setAttribute("aria-pressed", String(visible[m]));
    tip.hidden = true;
    draw();
  }));
  new ResizeObserver(resize).observe(canvas);
  resize();
})();
