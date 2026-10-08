const fs = require("fs");
const CHART_SECONDS = 4;
const MINIMAL_SECONDS = 5,
  SIGREG_HOLD_SECONDS = 1,
  ENDING_SHIFT = CHART_SECONDS + MINIMAL_SECONDS + SIGREG_HOLD_SECONDS;
const TMP = "/private/tmp/leavjepa-promo",
  SR = 48000,
  TITLE_SECONDS = 3,
  DUR = 30 + TITLE_SECONDS + ENDING_SHIFT,
  N = SR * DUR;
const read = (n) => {
  const b = fs.readFileSync(`${TMP}/${n}-full.f32`);
  return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
};
const guitar = read("guitar"),
  helicopter = read("helicopter");
const clamp = (x) => Math.min(1, Math.max(0, x));
const smooth = (x, a, b) => {
  x = clamp((x - a) / (b - a));
  return x * x * (3 - 2 * x);
};
function rms(x) {
  let s = 0;
  for (const a of x) s += a * a;
  return Math.sqrt(s / x.length);
}
const gg = 0.112 / rms(guitar),
  hg = 0.103 / rms(helicopter),
  samples = new Float64Array(N);
let peak = 0;
for (let i = 0; i < N; i++) {
  const t = i / SR - TITLE_SECONDS;
  let val = 0;
  const blend = smooth(t, 23.15 + ENDING_SHIFT, 26.65 + ENDING_SHIFT);
  if (t >= 0 && t < 26.65 + ENDING_SHIFT) {
    const phase = t % 7.68,
      j = Math.min(guitar.length - 1, Math.floor(phase * SR));
    const seam =
      Math.min(1, phase / 0.012) * Math.min(1, (7.68 - phase) / 0.012);
    const methodLevel = 1 - 0.36 * smooth(t, 9.8, 12);
    val +=
      guitar[j] *
      gg *
      seam *
      methodLevel *
      Math.cos((blend * Math.PI) / 2) *
      smooth(t, 0, 0.15);
  }
  if (t >= 23.15 + ENDING_SHIFT) {
    const j = Math.min(
      helicopter.length - 1,
      Math.floor((t - 23.15 - ENDING_SHIFT) * SR),
    );
    val +=
      helicopter[j] *
      hg *
      Math.sin((blend * Math.PI) / 2) *
      (1 - smooth(t, 29.15 + ENDING_SHIFT, 29.75 + ENDING_SHIFT));
  }
  samples[i] = val;
  peak = Math.max(peak, Math.abs(val));
}
const gain = Math.min(1, 0.79 / peak),
  b = Buffer.alloc(44 + N * 4);
b.write("RIFF", 0);
b.writeUInt32LE(36 + N * 4, 4);
b.write("WAVEfmt ", 8);
b.writeUInt32LE(16, 16);
b.writeUInt16LE(1, 20);
b.writeUInt16LE(2, 22);
b.writeUInt32LE(SR, 24);
b.writeUInt32LE(SR * 4, 28);
b.writeUInt16LE(4, 32);
b.writeUInt16LE(16, 34);
b.write("data", 36);
b.writeUInt32LE(N * 4, 40);
for (let i = 0; i < N; i++) {
  const v = Math.round(samples[i] * gain * 32767);
  b.writeInt16LE(v, 44 + i * 4);
  b.writeInt16LE(v, 46 + i * 4);
}
fs.writeFileSync(`${TMP}/clip-only.wav`, b);
console.log(
  JSON.stringify({
    seconds: DUR,
    title_seconds: TITLE_SECONDS,
    crossfade_seconds: 3.5,
    sources: ["Acoustic guitar", "Helicopter 2"],
    music_added: false,
    effects_added: false,
    peak_dbfs: 20 * Math.log10(peak * gain),
  }),
);
