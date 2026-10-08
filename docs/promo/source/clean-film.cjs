const fs = require("fs"),
  path = require("path"),
  vm = require("vm");
const { spawn } = require("child_process"),
  { once } = require("events");
const { createCanvas, ImageData, GlobalFonts } = require("@napi-rs/canvas");
const ROOT = path.resolve(__dirname, "../.."),
  TMP = "/private/tmp/leavjepa-promo";
const CHART_SECONDS = 4;
const MINIMAL_SECONDS = 5;
const SIGREG_HOLD_AT = 23,
  SIGREG_HOLD_SECONDS = 1;
const W = 1080,
  H = 1080,
  FPS = 30,
  TITLE_SECONDS = 3,
  DURATION =
    30 + TITLE_SECONDS + CHART_SECONDS + MINIMAL_SECONDS + SIGREG_HOLD_SECONDS;
const filmTimestamp = (t) =>
  TITLE_SECONDS + t + (t >= SIGREG_HOLD_AT ? SIGREG_HOLD_SECONDS : 0);
GlobalFonts.registerFromPath(
  "/Library/Fonts/SF-Pro-Display-Medium.otf",
  "Medium",
);
GlobalFonts.registerFromPath(
  "/Library/Fonts/SF-Pro-Display-Regular.otf",
  "Regular",
);
GlobalFonts.registerFromPath(
  "/Library/Fonts/SF-Pro-Display-Semibold.otf",
  "Semibold",
);
const cv = createCanvas(W, H),
  c = cv.getContext("2d");
const col = {
  black: "#050505",
  white: "#f3f2f4",
  muted: "#8c878f",
  line: "#454047",
  purple: "#a552bb",
  orange: "#f78212",
  faint: "#1c191e",
};
const clamp = (v) => Math.min(1, Math.max(0, v));
const sm = (v) => {
  v = clamp(v);
  return v * v * (3 - 2 * v);
};
const step = (t, a, b) => sm((t - a) / (b - a));
const lerp = (a, b, t) => a + (b - a) * t;
const rgba = (h, a) => {
  const v = parseInt(h.slice(1), 16);
  return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${clamp(a)})`;
};
const fade = (t, a, b, edge = 0.35) =>
  step(t, a, a + edge) * (1 - step(t, b - edge, b));
function group(alpha, fn) {
  if (alpha < 0.001) return;
  c.save();
  c.globalAlpha *= alpha;
  fn();
  c.restore();
}
function text(
  s,
  x,
  y,
  size = 32,
  color = col.white,
  weight = "Regular",
  align = "left",
) {
  c.save();
  c.fillStyle = color;
  c.font = `${size}px ${weight}`;
  c.textAlign = align;
  c.textBaseline = "top";
  c.fillText(s, x, y);
  c.restore();
}
function middleText(
  s,
  x,
  y,
  size = 26,
  color = col.muted,
  weight = "Regular",
  align = "left",
) {
  c.save();
  c.fillStyle = color;
  c.font = `${size}px ${weight}`;
  c.textAlign = align;
  c.textBaseline = "alphabetic";
  const m = c.measureText(s);
  c.fillText(
    s,
    x,
    y + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2,
  );
  c.restore();
}
function line(x1, y1, x2, y2, color = col.line, width = 1) {
  c.strokeStyle = color;
  c.lineWidth = width;
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(x2, y2);
  c.stroke();
}
function dot(x, y, r, color) {
  c.fillStyle = color;
  c.beginPath();
  c.arc(x, y, r, 0, Math.PI * 2);
  c.fill();
}
function rect(x, y, w, h, fill = null, stroke = null, r = 0) {
  c.beginPath();
  c.roundRect(x, y, w, h, r);
  if (fill) {
    c.fillStyle = fill;
    c.fill();
  }
  if (stroke) {
    c.strokeStyle = stroke;
    c.lineWidth = 1;
    c.stroke();
  }
}
function caption(s, t, a, b, y = 146, size = 48) {
  group(fade(t, a, b, 0.38), () =>
    text(s, 540, y, size, col.white, "Medium", "center"),
  );
}
const srcs = [
  ["guitar", "AJw-x30L46E_000080"],
  ["helicopter", "9nsLTLNvOSw_000150"],
].map(([name, id]) => {
  const win = {};
  vm.runInNewContext(
    fs.readFileSync(path.join(ROOT, "assets/attention", id, "maps.js"), "utf8"),
    { window: win },
  );
  const d = win.LEAV_ATTENTION_DATA[id],
    b = Buffer.from(d.video, "base64");
  const a = fs.readFileSync(path.join(TMP, name + "-full.f32"));
  return {
    name,
    id,
    d,
    weights: new Float32Array(b.buffer, b.byteOffset, b.length / 4),
    audio: new Float32Array(a.buffer, a.byteOffset, a.length / 4),
    fd: fs.openSync(path.join(TMP, name + "-v2.rgba"), "r"),
    cache: new Map(),
    heatCache: new Map(),
  };
});
function video(src, t) {
  const k = Math.max(
    0,
    Math.min(191, Math.floor((((t % 7.68) + 7.68) % 7.68) * 25)),
  );
  if (!src.cache.has(k)) {
    const raw = Buffer.alloc(672 * 672 * 4);
    fs.readSync(src.fd, raw, 0, raw.length, k * raw.length);
    const v = createCanvas(672, 672);
    v.getContext("2d").putImageData(
      new ImageData(
        new Uint8ClampedArray(raw.buffer, raw.byteOffset, raw.length),
        672,
        672,
      ),
      0,
      0,
    );
    src.cache.set(k, v);
    if (src.cache.size > 14) src.cache.delete(src.cache.keys().next().value);
  }
  return src.cache.get(k);
}
function heat(src, t) {
  const tt = (((t % 7.68) + 7.68) % 7.68) + src.d.sourceOffset;
  let k = 0;
  for (let i = 1; i < src.d.times.length; i++)
    if (Math.abs(src.d.times[i] - tt) < Math.abs(src.d.times[k] - tt)) k = i;
  if (!src.heatCache.has(k)) {
    const p = new Uint8ClampedArray(42 * 42 * 4);
    for (let i = 0; i < 42 * 42; i++) {
      let idx = Math.min(
          255,
          Math.max(
            0,
            Math.floor((src.weights[k * 1764 + i] / src.d.maximum) * 255),
          ),
        ),
        rgb = src.d.palette[idx];
      p[i * 4] = rgb[0];
      p[i * 4 + 1] = rgb[1];
      p[i * 4 + 2] = rgb[2];
      p[i * 4 + 3] = 255;
    }
    const h = createCanvas(42, 42);
    h.getContext("2d").putImageData(new ImageData(p, 42, 42), 0, 0);
    src.heatCache.set(k, h);
  }
  return src.heatCache.get(k);
}
function melCanvas(src) {
  const sr = 16000,
    nfft = 400,
    hop = 160,
    bands = 128,
    bins = 201;
  const samples = Float64Array.from(
    { length: Math.floor(src.audio.length / 3) },
    (_, i) => src.audio[i * 3],
  );
  const cols = Math.floor((samples.length - nfft) / hop) + 1;
  const co = new Float64Array(bins * nfft),
    si = new Float64Array(bins * nfft),
    wind = new Float64Array(nfft);
  for (let j = 0; j < nfft; j++)
    wind[j] = 0.54 - 0.46 * Math.cos((2 * Math.PI * j) / nfft);
  for (let k = 0; k < bins; k++)
    for (let j = 0; j < nfft; j++) {
      co[k * nfft + j] = Math.cos((2 * Math.PI * k * j) / nfft);
      si[k * nfft + j] = Math.sin((2 * Math.PI * k * j) / nfft);
    }
  const mel = (f) => 2595 * Math.log10(1 + f / 700),
    hz = (m) => 700 * (10 ** (m / 2595) - 1);
  const edges = Array.from({ length: bands + 2 }, (_, i) =>
    hz((i / (bands + 1)) * mel(8000)),
  );
  const filters = Array.from({ length: bands }, (_, b) =>
    Float64Array.from({ length: bins }, (_, k) => {
      const f = (k * sr) / nfft;
      return Math.max(
        0,
        Math.min(
          (f - edges[b]) / (edges[b + 1] - edges[b]),
          (edges[b + 2] - f) / (edges[b + 2] - edges[b + 1]),
        ),
      );
    }),
  );
  const power = new Float64Array(bins),
    frame = new Float64Array(nfft),
    values = new Float64Array(cols * bands);
  let max = -1000;
  for (let x = 0; x < cols; x++) {
    for (let j = 0; j < nfft; j++) frame[j] = samples[x * hop + j] * wind[j];
    for (let k = 0; k < bins; k++) {
      let re = 0,
        im = 0,
        off = k * nfft;
      for (let j = 0; j < nfft; j++) {
        re += frame[j] * co[off + j];
        im -= frame[j] * si[off + j];
      }
      power[k] = re * re + im * im;
    }
    for (let y = 0; y < bands; y++) {
      let v = 0;
      for (let k = 0; k < bins; k++) v += power[k] * filters[y][k];
      v = 10 * Math.log10(v + 1e-12);
      values[y * cols + x] = v;
      max = Math.max(max, v);
    }
  }
  const pix = new Uint8ClampedArray(cols * bands * 4);
  for (let y = 0; y < bands; y++)
    for (let x = 0; x < cols; x++) {
      let q = clamp((values[y * cols + x] - max + 80) / 80),
        rgb = src.d.palette[Math.floor(q * 255)],
        off = ((bands - 1 - y) * cols + x) * 4;
      pix[off] = rgb[0];
      pix[off + 1] = rgb[1];
      pix[off + 2] = rgb[2];
      pix[off + 3] = 255;
    }
  const m = createCanvas(cols, bands);
  m.getContext("2d").putImageData(new ImageData(pix, cols, bands), 0, 0);
  return m;
}
const mel = melCanvas(srcs[0]);
function wave(src, t, x, y, w, h, color = col.orange) {
  const n = 128,
    center = Math.floor((((t % 7.68) + 7.68) % 7.68) * 48000),
    span = 19200;
  c.strokeStyle = color;
  c.lineWidth = 1.5;
  c.lineCap = "round";
  c.beginPath();
  for (let j = 0; j < n; j++) {
    let rms = 0;
    for (let k = 0; k < 36; k++) {
      let i =
        (center + Math.floor((j / n - 0.5) * span) + k * 3 + src.audio.length) %
        src.audio.length;
      rms += src.audio[i] * src.audio[i];
    }
    const a = Math.min(1, Math.sqrt(rms / 36) * 6.5),
      v = Math.max(1, a * h);
    const xx = x + (j * w) / (n - 1);
    c.moveTo(xx, y - v / 2);
    c.lineTo(xx, y + v / 2);
  }
  c.stroke();
  c.lineCap = "butt";
}
function cursor(x, y, down = 1) {
  c.save();
  c.translate(x, y);
  c.scale(0.84, 0.84);
  c.shadowColor = "#0008";
  c.shadowBlur = 3;
  c.beginPath();
  c.moveTo(0, 24);
  c.lineTo(-5, 12);
  c.quadraticCurveTo(-6, 7, -2, 6);
  c.quadraticCurveTo(1, 5, 4, 11);
  c.lineTo(6, 15);
  c.lineTo(6, -10);
  c.quadraticCurveTo(6, -15, 10, -15);
  c.quadraticCurveTo(14, -15, 14, -10);
  c.lineTo(14, 3);
  c.quadraticCurveTo(20, -2, 23, 4);
  c.quadraticCurveTo(29, 0, 32, 7);
  c.quadraticCurveTo(39, 4, 40, 12);
  c.lineTo(40, 24);
  c.quadraticCurveTo(37, 35, 32, 41);
  c.lineTo(11, 41);
  c.quadraticCurveTo(9, 35, 0, 24);
  c.fillStyle = col.white;
  c.fill();
  c.strokeStyle = "#242026";
  c.lineWidth = 1.7;
  c.stroke();
  c.restore();
}
function comparison(
  src,
  time,
  x,
  y,
  s,
  split,
  handleAlpha = 1,
  cursorAlpha = 0,
  smallLabels = true,
) {
  c.save();
  c.beginPath();
  c.roundRect(x, y, s, s, 14);
  c.clip();
  c.drawImage(video(src, time), x, y, s, s);
  c.save();
  c.beginPath();
  c.rect(x + s * split, y, s * (1 - split), s);
  c.clip();
  c.imageSmoothingEnabled = false;
  c.drawImage(heat(src, time), x, y, s, s);
  c.restore();
  if (smallLabels) {
    const a = clamp((s - 430) / 200);
    group(a, () => {
      if (split > 0.18) {
        rect(x + 20, y + 20, 76, 30, "#0008", null, 5);
        text("Video", x + 30, y + 25, 19);
      }
      if (split < 0.72) {
        rect(x + s - 172, y + 20, 152, 30, "#0008", null, 5);
        text("Audio attention", x + s - 162, y + 25, 19);
      }
    });
  }
  c.restore();
  group(handleAlpha, () => {
    const xx = x + s * split,
      yy = y + s * 0.53;
    line(xx, y + 1, xx, y + s - 1, "#ffffffe6", 1.5);
    c.save();
    c.shadowColor = "#0009";
    c.shadowBlur = 12;
    dot(xx, yy, 20, col.white);
    c.restore();
    line(xx - 5, yy - 5, xx - 10, yy, col.black, 1.5);
    line(xx - 10, yy, xx - 5, yy + 5, col.black, 1.5);
    line(xx + 5, yy - 5, xx + 10, yy, col.black, 1.5);
    line(xx + 10, yy, xx + 5, yy + 5, col.black, 1.5);
    group(cursorAlpha, () => cursor(xx + 1, yy + 9));
  });
}
function splitIntro(t) {
  if (t < 1.3) return 0.99;
  if (t < 3.3) return lerp(0.99, 0.24, step(t, 1.3, 3.3));
  if (t < 5.4) return lerp(0.24, 0.68, step(t, 3.3, 5.4));
  return lerp(0.68, 1, step(t, 5.8, 6.65));
}
const sourcePos = {
  v: { x: 170, y: 370, w: 300, h: 300 },
  a: { x: 610, y: 445, w: 300, h: 150 },
};
function sourceSequence(t) {
  const shrink = step(t, 6.4, 8.3),
    patch = step(t, 8.7, 9.8),
    gone = step(t, 10.25, 11.3);
  const travel = step(t, 10.25, 11.3);
  const x = lerp(lerp(150, sourcePos.v.x, shrink), 70, travel),
    y = lerp(lerp(250, sourcePos.v.y, shrink), 864, travel),
    s = lerp(lerp(780, 300, shrink), 124, travel);
  group(1 - gone, () => {
    group(shrink * 0.8 * (1 - travel), () => {
      for (let i = 2; i >= 1; i--) {
        c.save();
        c.globalAlpha *= 1 - i * 0.22;
        c.drawImage(video(srcs[0], t - i * 0.16), x + i * 12, y - i * 13, s, s);
        c.restore();
      }
    });
    if (t < 8.7)
      comparison(
        srcs[0],
        t,
        x,
        y,
        s,
        splitIntro(t),
        1 - step(t, 6.05, 6.75),
        fade(t, 1.15, 5.75, 0.25),
        t < 6.35,
      );
    else {
      const im = video(srcs[0], t);
      for (let j = 0; j < 6; j++)
        for (let i = 0; i < 6; i++) {
          const gap = patch * 4.5,
            px = x + (i * s) / 6 + gap / 2,
            py = y + (j * s) / 6 + gap / 2;
          c.drawImage(
            im,
            i * 112,
            j * 112,
            112,
            112,
            px,
            py,
            s / 6 - gap,
            s / 6 - gap,
          );
        }
      group(1 - patch, () => {
        for (let i = 1; i < 6; i++) {
          line(x + (i * s) / 6, y, x + (i * s) / 6, y + s, "#ffffff65");
          line(x, y + (i * s) / 6, x + s, y + (i * s) / 6, "#ffffff65");
        }
      });
    }
    group(shrink * (1 - travel), () => {
      text("Video", x + s / 2, y + s + 33, 29, col.white, "Regular", "center");
      const ax = sourcePos.a.x,
        ay = sourcePos.a.y;
      const g = step(t, 8.05, 9.25);
      group(1 - g, () => wave(srcs[0], t, ax, ay + 75, 300, 116));
      group(g, () => {
        for (let j = 0; j < 4; j++)
          for (let i = 0; i < 8; i++) {
            const gap = patch * 3;
            c.drawImage(
              mel,
              (i * mel.width) / 8,
              (j * mel.height) / 4,
              mel.width / 8,
              mel.height / 4,
              ax + i * 37.5 + gap / 2,
              ay + j * 37.5 + gap / 2,
              37.5 - gap,
              37.5 - gap,
            );
          }
      });
      text(
        "Audio",
        ax + 150,
        ay + sourcePos.a.h + 108,
        29,
        col.white,
        "Regular",
        "center",
      );
    });
  });
}
function miniSource(t, clipTime = t) {
  const a =
    step(t, 10.8, 11.4) *
    (1 -
      step(
        t,
        22.65 + CHART_SECONDS + MINIMAL_SECONDS,
        23.5 + CHART_SECONDS + MINIMAL_SECONDS,
      ));
  group(a, () => {
    const x = 70,
      y = 864,
      s = 124;
    c.save();
    c.beginPath();
    c.roundRect(x, y, s, s, 6);
    c.clip();
    c.drawImage(video(srcs[0], clipTime), x, y, s, s);
    c.restore();
    wave(srcs[0], clipTime, 218, 926, 192, 39, rgba(col.orange, 0.7));
  });
}
function tokenRowLayout(t, row) {
  const compact = step(t, 13.65, 15.05),
    drop = step(t, 11.9, 13.35);
  const y = lerp(373 + row * 102, 390 + row * 77, compact),
    size = lerp(36, 18, compact);
  const xForToken = (i) => {
    const audio = i >= 6,
      remaining = row >= 2 && ((row === 2 && audio) || (row === 3 && !audio));
    let x = 298 + i * 42;
    if (remaining) x = lerp(x, 424 + (i % 6) * 42, drop);
    const compactX =
      244 + i * 21 + (remaining ? (audio ? -3 : 3) * 21 * drop : 0);
    return lerp(x, compactX, compact);
  };
  const labelStart =
    row === 2
      ? lerp(xForToken(0), xForToken(6), sm((drop - 0.75) / 0.25))
      : xForToken(0);
  return {
    compact,
    y,
    size,
    centerY: y + size / 2,
    xForToken,
    right: xForToken(row === 3 ? 5 : 11) + size,
    labelRight: labelStart - (row < 2 ? 34 : 18),
    labelSize: lerp(26, 24, compact),
  };
}
function tokenRows(t) {
  const appear = step(t, 10, 11.4),
    compact = step(t, 13.65, 15.05),
    gone = step(t, 16.3, 17.65);
  if (appear <= 0 || gone >= 1) return;
  const drop = step(t, 11.9, 13.35),
    im = video(srcs[0], t);
  group(1 - gone, () => {
    for (let row = 0; row < 4; row++) {
      const layout = tokenRowLayout(t, row),
        yy = layout.y;
      group(appear, () => {
        if (row === 0) {
          const next = tokenRowLayout(t, 1);
          middleText(
            "Joint",
            layout.labelRight,
            (layout.centerY + next.centerY) / 2,
            layout.labelSize,
            col.muted,
            "Regular",
            "right",
          );
          const bx = layout.xForToken(0) - 20,
            top = layout.centerY,
            bottom = next.centerY;
          line(bx + 6, top, bx, top, col.line);
          line(bx, top, bx, bottom, col.line);
          line(bx, bottom, bx + 6, bottom, col.line);
        } else if (row >= 2)
          middleText(
            row === 2 ? "Audio" : "Video",
            layout.labelRight,
            layout.centerY,
            layout.labelSize,
            col.muted,
            "Regular",
            "right",
          );
      });
      for (let i = 0; i < 12; i++) {
        const audio = i >= 6,
          dropped = (row === 2 && !audio) || (row === 3 && audio),
          base = audio ? sourcePos.a : sourcePos.v;
        const sx = base.x + ((((i + row * 3) % 6) + 0.5) * base.w) / 6,
          sy = base.y + ((((row + i) % 4) + 0.5) * base.h) / 4;
        const px = lerp(sx, layout.xForToken(i), appear),
          py = lerp(sy, yy, appear) + (dropped ? -48 * drop : 0),
          size = lerp(44, layout.size, appear);
        const a = (dropped ? 1 - drop : 1) * appear;
        group(a, () => {
          const img = audio ? mel : im;
          const ix = ((i % 6) * img.width) / 6,
            iy = ((row % 4) * img.height) / 4;
          c.drawImage(
            img,
            ix,
            iy,
            img.width / 6,
            img.height / 4,
            px,
            py,
            size,
            size,
          );
          const tint = audio ? col.orange : col.purple;
          rect(
            px,
            py,
            size,
            size,
            rgba(tint, step(t, 10.4, 11.5) * (audio ? 0.42 : 0.48)),
            rgba(tint, 0.72),
            2,
          );
        });
      }
    }
  });
}
function bezierPoint(a, b, d, e, u) {
  const v = 1 - u;
  return [
    v * v * v * a[0] +
      3 * v * v * u * b[0] +
      3 * v * u * u * d[0] +
      u * u * u * e[0],
    v * v * v * a[1] +
      3 * v * v * u * b[1] +
      3 * v * u * u * d[1] +
      u * u * u * e[1],
  ];
}
const viewStart = [
    [365, 414],
    [751, 422],
    [355, 739],
    [769, 733],
  ],
  viewEnd = [
    [520, 551],
    [558, 568],
    [528, 594],
    [566, 603],
  ];
function encoder(t) {
  const a = step(t, 14.4, 15.15),
    exit = step(t, 17.05, 18.3),
    modelX = 600,
    cx = modelX + 129;
  group(a * (1 - exit), () => {
    const dx = -exit * 70,
      dy = -exit * 30;
    group(step(t, 14.65, 15.05) * (1 - step(t, 16.3, 17.65)), () => {
      for (let r = 0; r < 4; r++) {
        const row = tokenRowLayout(t, r),
          start = [row.right, row.centerY],
          end = [modelX + dx, 522 + dy];
        const control1 = [start[0] + (end[0] - start[0]) * 0.55, start[1]],
          control2 = [end[0] - 35, end[1]];
        c.strokeStyle = rgba(col.line, 0.92);
        c.lineWidth = 1.25;
        c.beginPath();
        c.moveTo(...start);
        c.bezierCurveTo(...control1, ...control2, ...end);
        c.stroke();
        group(1 - step(t, 16.2, 17.0), () => {
          for (let k = 0; k < 2; k++) {
            const q = ((t - 14.3) * 0.8 + k * 0.5 + r * 0.12) % 1;
            if (q < 0) continue;
            const [x, y] = bezierPoint(start, control1, control2, end, q);
            const color =
              r === 2
                ? col.orange
                : r === 3
                  ? col.purple
                  : k
                    ? col.orange
                    : col.purple;
            dot(x, y, 3.2, color);
          }
        });
      }
    });
    c.save();
    c.translate(dx, dy);
    rect(modelX, 401, 258, 254, "#0c0b0d", "#6c6670", 9);
    text("Shared ViT", cx, 423, 33, col.white, "Medium", "center");
    rect(modelX + 23, 480, 212, 58, "#141116", "#3a343e", 5);
    text("Attention", cx, 494, 29, col.white, "Regular", "center");
    line(cx, 539, cx, 561, col.line, 1.2);
    rect(modelX + 23, 561, 212, 58, "#141116", "#3a343e", 5);
    text("MLP", cx, 575, 29, col.white, "Regular", "center");
    const p = ((t - 14.3) * 0.65) % 1;
    if (p >= 0) {
      const yy = lerp(478, 621, p);
      group(0.45 * Math.sin(p * Math.PI), () =>
        line(
          modelX + 24,
          yy,
          modelX + 234,
          yy,
          p < 0.5 ? col.purple : col.orange,
          1.4,
        ),
      );
    }
    line(cx, 655, cx, 675, col.line, 1.3);
    line(cx, 707, cx, 734, col.line, 1.3);
    rect(cx - 32, 675, 64, 32, "#111014", "#6c6670", 5);
    middleText("CLS", cx, 691, 22, col.white, "Regular", "center");
    rect(modelX, 734, 258, 57, "#0c0b0d", "#6c6670", 6);
    text("Projector", cx, 747, 31, col.white, "Regular", "center");
    line(cx, 791, cx, 817, col.line, 1.2);
    c.restore();
  });
}
function embeddingPoints(t) {
  const birth = step(t, 16.45, 18.45),
    align = step(t, 18.65, 20.45),
    exit = step(t, 20.45, 21.2);
  if (birth <= 0) return;
  group(1 - exit, () => {
    const cols = [col.white, col.white, col.orange, col.purple];
    for (let i = 0; i < 4; i++) {
      const fly = step(t, 16.45 + i * 0.16, 18.05 + i * 0.16);
      const p = bezierPoint(
        [729, 817],
        [850 - i * 95, 831 - i * 20],
        [viewStart[i][0] + 70, viewStart[i][1] + 30],
        viewStart[i],
        fly,
      );
      const x = lerp(p[0], viewEnd[i][0], align),
        y = lerp(p[1], viewEnd[i][1], align);
      group(step(t, 18.05, 18.5) * 0.45, () =>
        line(x, y, 540, 577, rgba(cols[i], 0.42), 1),
      );
      dot(x, y, i < 2 ? 8 : 10, cols[i]);
      if (i < 2) {
        c.strokeStyle = rgba(i ? col.orange : col.purple, 0.85);
        c.lineWidth = 2;
        c.beginPath();
        c.arc(x, y, 13, 0, Math.PI * 2);
        c.stroke();
      }
    }
    group(step(t, 18.2, 18.55), () => {
      c.strokeStyle = "#777179";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(540, 569);
      c.lineTo(548, 577);
      c.lineTo(540, 585);
      c.lineTo(532, 577);
      c.closePath();
      c.stroke();
    });
    group(fade(t, 18.4, 20.8), () =>
      text("Alignment", 540, 811, 29, col.muted, "Regular", "center"),
    );
  });
}
const batch = Array.from({ length: 24 }, (_, i) => {
  const radius = Math.sqrt(-2 * Math.log((i + 0.5) / 24)),
    angle = i * 2.39996323 + 0.35;
  const startAngle = i * 2.39996323,
    startRadius = 55 + Math.sqrt(i / 24) * 105;
  return {
    x: radius * Math.cos(angle) * 132,
    y: radius * Math.sin(angle) * 132,
    sx: Math.cos(startAngle) * startRadius,
    sy: Math.sin(startAngle) * startRadius * 0.72,
    viewAngle: angle * 0.37 + 0.25,
    delay: (i % 4) * 0.055,
  };
});
function spreadPosition(p, t) {
  const pull = step(t, 20.55 + p.delay * 0.5, 21.23 + p.delay * 0.5);
  const struggle = step(t, 20.95, 21.23) * (1 - step(t, 21.52, 21.82));
  const squeeze =
    lerp(1, 0.065, pull) * (1 + struggle * 0.24 * Math.sin((t - 21.2) * 15));
  const cx = p.sx * squeeze + struggle * 2.4 * Math.sin(t * 19 + p.viewAngle);
  const cy = p.sy * squeeze + struggle * 1.8 * Math.cos(t * 17 + p.viewAngle);
  const u = step(t, 21.65 + p.delay, 22.7 + p.delay),
    bend = Math.sin(u * Math.PI) * 11;
  return {
    x: 540 + lerp(cx, p.x, u) - Math.sin(p.viewAngle) * bend,
    y: 556 + lerp(cy, p.y, u) + Math.cos(p.viewAngle) * bend,
    u,
  };
}
function spreadRepresentations(t) {
  const a = step(t, 20.15, 20.65),
    spread = step(t, 21.65, 22.85),
    gone = step(t, 23.05, 24.0);
  group(a * (1 - gone), () => {
    group(step(t, 21.48, 21.85), () => {
      for (const radius of [132, 264]) {
        c.strokeStyle = rgba(col.white, radius === 132 ? 0.28 : 0.22);
        c.lineWidth = 1.25;
        c.beginPath();
        c.arc(540, 556, lerp(45, radius, spread), 0, Math.PI * 2);
        c.stroke();
      }
      const rx = lerp(45, 264, spread),
        ry = rx;
      for (let i = 0; i < 4; i++) {
        const theta = (i * Math.PI) / 4,
          dx = Math.cos(theta) * rx,
          dy = Math.sin(theta) * ry;
        line(
          540 - dx,
          556 - dy,
          540 + dx,
          556 + dy,
          rgba(col.white, i % 2 ? 0.12 : 0.18),
          1,
        );
      }
    });
    for (let i = 0; i < batch.length; i++) {
      const p = batch[i],
        pos = spreadPosition(p, t),
        before = spreadPosition(p, t - 0.09);
      const offsets = [
        [-5.5, -1],
        [3, -4],
        [4, 4],
      ].map(([x, y], k) => {
        const co = Math.cos(p.viewAngle),
          si = Math.sin(p.viewAngle);
        return {
          x: x * co - y * si,
          y: x * si + y * co,
          color: [col.white, col.orange, col.purple][k],
        };
      });
      const visibility = step(t, 20.15 + p.delay, 20.6 + p.delay);
      group(visibility, () => {
        const motion = clamp(
          Math.hypot(pos.x - before.x, pos.y - before.y) / 12,
        );
        for (let k = 0; k < 3; k++) {
          const v = offsets[k];
          line(
            before.x + v.x,
            before.y + v.y,
            pos.x + v.x,
            pos.y + v.y,
            rgba(v.color, motion * 0.32),
            1.7,
          );
        }
        for (let k = 0; k < 3; k++) {
          const v = offsets[k];
          dot(pos.x + v.x, pos.y + v.y, k === 0 ? 5.5 : 6, v.color);
        }
      });
    }
  });
}
function minimalTraining(t) {
  if (t < 23.6 || t >= 23.75 + MINIMAL_SECONDS) return;
  const disappear =
    1 - step(t, 23.05 + MINIMAL_SECONDS, 23.75 + MINIMAL_SECONDS);
  const positive = step(t, 25.55, 26.05);
  group(disappear, () => {
    group(step(t, 23.82, 24.2), () =>
      text("During pretraining", 128, 205, 28, col.muted),
    );
    const labels = [
      "contrastive negatives",
      "EMA teacher",
      "decoder",
      "predictor",
    ];
    for (let i = 0; i < labels.length; i++) {
      const reveal = step(t, 23.95 + i * 0.32, 24.4 + i * 0.32),
        x = 128 + (1 - reveal) * 22,
        y = 282 + i * 101;
      group(reveal * (1 - positive * 0.35), () => {
        c.save();
        c.font = "62px Medium";
        const noWidth = c.measureText("No").width;
        c.restore();
        const accent = c.createLinearGradient(x, 0, x + noWidth, 0);
        accent.addColorStop(0, col.purple);
        accent.addColorStop(1, col.orange);
        text("No", x, y, 62, accent, "Medium");
        text(labels[i], x + noWidth + 20, y, 62, col.white, "Regular");
      });
    }
    group(positive, () => {
      const x = 128,
        y = 716 + (1 - positive) * 12;
      text("Latent-space alignment", x, y, 52, col.white, "Regular");
      text("+ regularisation", x, y + 61, 52, col.white, "Regular");
    });
  });
}
const viewAblations = [
  { label: ["Joint", "views"], value: 24.7, color: "#625d67" },
  { label: ["Masking"], value: 34.1, color: "#79727e" },
  { label: ["Random", "dropout"], value: 41.3, color: "#958e9b" },
  { label: ["Modality", "dropout"], value: 47.1, color: null },
];
function ablationChart(t) {
  if (t < 23.6 || t >= 27.75) return;
  const visibility = step(t, 23.6, 24.0) * (1 - step(t, 27.05, 27.75));
  group(visibility, () => {
    group(step(t, 23.78, 24.1), () => {
      text(
        "The view makes the difference.",
        540,
        146,
        48,
        col.white,
        "Medium",
        "center",
      );
    });
    const baseline = 730,
      height = 390,
      left = 184,
      right = 996;
    group(step(t, 23.8, 24.15), () => {
      c.save();
      c.translate(92, baseline - height / 2);
      c.rotate(-Math.PI / 2);
      middleText(
        "Top-1 accuracy (%)",
        0,
        0,
        27,
        col.muted,
        "Regular",
        "center",
      );
      c.restore();
      for (const tick of [0, 25, 50]) {
        const y = baseline - (height * tick) / 50;
        line(left, y, right, y, rgba(col.white, tick === 0 ? 0.28 : 0.09), 1);
        middleText(
          String(tick),
          left - 20,
          y,
          23,
          col.muted,
          "Regular",
          "right",
        );
      }
    });
    for (let i = 0; i < viewAblations.length; i++) {
      const d = viewAblations[i],
        x = left + ((i + 0.5) * (right - left)) / viewAblations.length,
        start = 23.95 + i * 0.3;
      const grow = step(t, start, start + 0.75),
        h = ((height * d.value) / 50) * grow;
      let fill = d.color;
      if (!fill) {
        fill = c.createLinearGradient(x - 64, 0, x + 64, 0);
        fill.addColorStop(0, col.purple);
        fill.addColorStop(1, col.orange);
      }
      if (h > 0) rect(x - 64, baseline - h, 128, h, fill, null, 4);
      group(step(t, start + 0.48, start + 0.85), () =>
        text(
          d.value.toFixed(1) + "%",
          x,
          baseline - h - 66,
          50,
          col.white,
          "Medium",
          "center",
        ),
      );
      group(step(t, start - 0.12, start + 0.2), () => {
        d.label.forEach((label, j) =>
          text(
            label,
            x,
            d.label.length === 1 ? 790 : 772 + j * 37,
            29,
            d.color ? "#b9b3bd" : col.white,
            "Regular",
            "center",
          ),
        );
      });
    }
  });
}
function returnClip(t) {
  const grow = step(t, 23.15, 24.35),
    ending = step(t, 27, 28.2),
    a = step(t, 23.1, 23.9);
  if (a <= 0) return;
  let s = lerp(250, 780, grow);
  s = lerp(s, 690, ending);
  let x = (1080 - s) / 2,
    y = lerp(440, 250, grow);
  y = lerp(y, 190, ending);
  const tm = Math.max(0, t - 23.15);
  let split = 0.84;
  if (t > 23.95 && t < 25.8) split = lerp(0.84, 0.1, step(t, 23.95, 25.8));
  else if (t >= 25.8 && t < 26.1) split = 0.1;
  else if (t >= 26.1 && t < 27.35)
    split = lerp(0.1, 0.52, step(t, 26.1, 27.35));
  else if (t >= 27.35) split = lerp(0.52, 0, step(t, 27.35, 28.55));
  const handle = step(t, 23.6, 24.35) * (1 - step(t, 28.55, 28.9));
  group(a, () =>
    comparison(
      srcs[1],
      tm,
      x,
      y,
      s,
      split,
      handle,
      fade(t, 24.2, 28.8, 0.25),
      ending < 0.4,
    ),
  );
  group(fade(t, 24.4, 27.0), () =>
    text(
      "No localization supervision.",
      540,
      212,
      26,
      col.muted,
      "Regular",
      "center",
    ),
  );
  group(step(t, 27.1, 28.1), () => {
    text("Paper + code", 540, 914, 25, col.muted, "Regular", "center");
    text(
      "aaltoml.github.io/LeAVJEPA",
      540,
      960,
      32,
      col.white,
      "Regular",
      "center",
    );
  });
}
function renderFilm(elapsed) {
  const t =
    elapsed -
    Math.min(SIGREG_HOLD_SECONDS, Math.max(0, elapsed - SIGREG_HOLD_AT));
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.globalAlpha = 1;
  c.fillStyle = col.black;
  c.fillRect(0, 0, W, H);
  caption("Hear it.", t, -0.45, 1.95);
  caption("See where it comes from.", t, 2.05, 6.25);
  caption("Start with the same event.", t, 6.7, 10.35);
  caption("Drop a modality.", t, 10.55, 14.0);
  caption("Share one transformer.", t, 14.15, 17.85);
  caption("Bring the views together.", t, 18.0, 20.9);
  caption("Keep the representations spread out.", t, 21.65, 23.75);
  group(fade(t, 21.65, 23.75, 0.38), () =>
    text("With SIGReg", 540, 212, 28, col.muted, "Regular", "center"),
  );
  caption(
    "The correlation emerges.",
    t - CHART_SECONDS - MINIMAL_SECONDS,
    24.0,
    27.15,
  );
  if (t < 11.3) sourceSequence(t);
  miniSource(t, elapsed);
  c.save();
  const zoom = 1 + 0.12 * step(t, 11.35, 14);
  c.translate(540, 575);
  c.scale(zoom, zoom);
  c.translate(-540, -575);
  tokenRows(t);
  encoder(t);
  embeddingPoints(t);
  spreadRepresentations(t);
  c.restore();
  minimalTraining(t);
  ablationChart(t - MINIMAL_SECONDS);
  returnClip(t - CHART_SECONDS - MINIMAL_SECONDS);
}
function titleCard() {
  rect(0, 0, W, H, col.black);
  c.save();
  c.font = "156px Semibold";
  let x = 76;
  for (const part of ["Le", "AV", "JEPA"]) {
    const width = c.measureText(part).width;
    let color = col.white;
    if (part === "AV") {
      color = c.createLinearGradient(x, 0, x + width, 0);
      color.addColorStop(0, col.purple);
      color.addColorStop(1, col.orange);
    }
    text(part, x, 264, 156, color, "Semibold");
    x += width;
  }
  c.restore();
  const accent = c.createLinearGradient(80, 0, 318, 0);
  accent.addColorStop(0, col.purple);
  accent.addColorStop(1, col.orange);
  rect(80, 461, 238, 2, accent);
  text("A minimalist architecture", 80, 519, 58, col.white, "Regular");
  text("for audio-visual", 80, 591, 58, col.white, "Regular");
  text("self-supervised learning.", 80, 663, 58, col.white, "Regular");
  text(
    "ELLIS Institute Finland · Aalto University",
    80,
    832,
    27,
    col.muted,
    "Regular",
  );
}
function render(t) {
  const filmTime = t - TITLE_SECONDS;
  if (filmTime >= 0) renderFilm(filmTime);
  else {
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    c.fillStyle = col.black;
    c.fillRect(0, 0, W, H);
  }
  if (t < TITLE_SECONDS + 0.35) {
    const cover = 1 - step(t, TITLE_SECONDS, TITLE_SECONDS + 0.35);
    group(cover, () => rect(0, 0, W, H, col.black));
    const titleAlpha = 1 - step(t, TITLE_SECONDS - 0.2, TITLE_SECONDS + 0.25);
    group(titleAlpha, titleCard);
  }
  group(step(t, DURATION - 0.75, DURATION - 0.25), titleCard);
}
async function main() {
  if (process.argv.includes("--stills")) {
    const times = [
      1,
      ...[
        0.6,
        3.4,
        5.3,
        7.5,
        9.7,
        11.6,
        13.3,
        15.6,
        17.2,
        18.5,
        20.2,
        22.4,
        26.4,
        25.75 + MINIMAL_SECONDS,
        24.6 + CHART_SECONDS + MINIMAL_SECONDS,
        26.3 + CHART_SECONDS + MINIMAL_SECONDS,
        28.7 + CHART_SECONDS + MINIMAL_SECONDS,
      ].map(filmTimestamp),
    ];
    const sheet = createCanvas(1080, Math.ceil(times.length / 3) * 360),
      ctx = sheet.getContext("2d");
    for (let i = 0; i < times.length; i++) {
      render(times[i]);
      fs.writeFileSync(
        path.join(TMP, `clean-${i}.png`),
        cv.toBuffer("image/png"),
      );
      ctx.drawImage(cv, (i % 3) * 360, Math.floor(i / 3) * 360, 360, 360);
    }
    fs.writeFileSync(
      path.join(TMP, "clean-contact.png"),
      sheet.toBuffer("image/png"),
    );
    console.log("Clean-film previews rendered.");
    return;
  }
  const out = path.join(TMP, "clean-picture.mp4");
  const ff = spawn(
    "ffmpeg",
    [
      "-hide_banner",
      "-loglevel",
      "warning",
      "-y",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgba",
      "-s",
      "1080x1080",
      "-r",
      "30",
      "-i",
      "pipe:0",
      "-an",
      "-c:v",
      "libx264",
      "-crf",
      "17",
      "-preset",
      "medium",
      "-pix_fmt",
      "yuv420p",
      "-profile:v",
      "high",
      "-level",
      "4.2",
      "-color_primaries",
      "bt709",
      "-color_trc",
      "bt709",
      "-colorspace",
      "bt709",
      "-movflags",
      "+faststart",
      out,
    ],
    { stdio: ["pipe", "inherit", "inherit"] },
  );
  const done = once(ff, "close");
  const frames = DURATION * FPS;
  for (let i = 0; i < frames; i++) {
    render(i / FPS);
    if (!ff.stdin.write(cv.data())) await once(ff.stdin, "drain");
    if (i % 150 === 0) console.log(`Rendered ${i}/${frames} frames`);
  }
  ff.stdin.end();
  const [code] = await done;
  if (code) throw Error("FFmpeg failed");
  console.log("Picture complete: " + out);
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
