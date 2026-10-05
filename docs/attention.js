"use strict";
(() => {
  const root = document.getElementById("attention-viewer");
  if (!root) return;
  const get = (name) => document.getElementById(`attention-${name}`);
  const examples = window.LEAV_ATTENTION_EXAMPLES || [];
  const video = get("video"), canvas = get("map"), context = canvas.getContext("2d");
  const tile = document.createElement("canvas"), tileContext = tile.getContext("2d");
  const cached = new Map();
  let clip, weights, version = 0, frame = -1, ready = false;
  video.muted = true;

  function formatTime(seconds) {
    return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
  }
  function draw(mediaTime = video.currentTime) {
    if (!clip || !ready) return;
    const time = mediaTime + clip.sourceOffset;
    let next = 0;
    for (let i = 1; i < clip.frames; i++)
      if (Math.abs(clip.times[i] - time) < Math.abs(clip.times[next] - time)) next = i;
    if (next !== frame) {
      frame = next;
      const size = clip.grid ** 2, pixels = tileContext.createImageData(clip.grid, clip.grid);
      for (let i = 0; i < size; i++) {
        const value = weights[frame * size + i];
        const color = clip.palette[Math.max(0, Math.min(255, Math.floor(value / clip.maximum * 255)))];
        pixels.data.set(color, i * 4);
        pixels.data[i * 4 + 3] = 255;
      }
      tileContext.putImageData(pixels, 0, 0);
      context.imageSmoothingEnabled = false;
      context.drawImage(tile, 0, 0, canvas.width, canvas.height);
      canvas.dataset.frame = String(frame);
    }
    get("time").value = String(frame);
    get("clock").textContent = `${formatTime(mediaTime)} / ${formatTime(clip.duration)}`;
  }
  function controls() {
    get("play").textContent = video.paused ? "Play" : "Pause";
    get("sound").textContent = video.muted ? "Sound on" : "Sound off";
    get("sound").setAttribute("aria-pressed", String(!video.muted));
  }
  async function choose(id) {
    const request = ++version, example = examples.find((item) => item.id === id);
    video.pause();
    video.removeAttribute("src");
    video.load();
    clip = null; ready = false; frame = -1;
    context.clearRect(0, 0, canvas.width, canvas.height);
    root.setAttribute("aria-busy", "true");
    get("play").disabled = get("time").disabled = true;
    get("status").textContent = `Loading ${example.label.toLowerCase()}…`;
    try {
      if (!cached.has(id)) {
        await new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = example.data;
          script.onload = () => { script.remove(); resolve(); };
          script.onerror = () => { script.remove(); reject(new Error("Attention data could not be loaded. Please reload the page.")); };
          document.head.append(script);
        });
        const data = window.LEAV_ATTENTION_DATA[id];
        const values = new Float32Array(Uint8Array.from(atob(data.video), (c) => c.charCodeAt(0)).buffer);
        if (values.length !== data.frames * data.grid ** 2) throw new Error("Incomplete attention data.");
        cached.set(id, { data, values });
      }
      if (request !== version) return;
      ({ data: clip, values: weights } = cached.get(id));
      tile.width = tile.height = clip.grid;
      get("time").max = String(clip.frames - 1);
      get("time").value = "0";
      video.setAttribute("aria-label", `${example.label}: original video with optional sound`);
      canvas.setAttribute("aria-label", `${example.label}: audio-to-video attention`);
      video.poster = example.poster;
      video.src = example.source;
    } catch (error) {
      if (request !== version) return;
      root.setAttribute("aria-busy", "false");
      get("status").textContent = error.message;
    }
  }
  video.addEventListener("loadeddata", () => {
    if (!clip) return;
    ready = true;
    root.setAttribute("aria-busy", "false");
    get("play").disabled = get("time").disabled = false;
    get("status").textContent = clip.nativeHW && Math.min(...clip.nativeHW) < 672
      ? `Source footage: ${clip.nativeHW[1]} × ${clip.nativeHW[0]}.` : "";
    draw();
  });
  video.addEventListener("error", () => {
    if (clip) get("status").textContent = "This video could not be loaded. Please reload the page.";
  });
  video.addEventListener("seeked", () => draw());
  for (const event of ["play", "pause", "ended", "volumechange"]) video.addEventListener(event, controls);
  if (video.requestVideoFrameCallback) {
    const update = (_, metadata) => { draw(metadata.mediaTime); video.requestVideoFrameCallback(update); };
    video.requestVideoFrameCallback(update);
  } else {
    const update = () => { if (!video.paused) draw(); requestAnimationFrame(update); };
    requestAnimationFrame(update);
  }
  get("play").addEventListener("click", () => {
    if (video.paused) video.play().catch(() => { get("status").textContent = "Playback could not start. Try pressing Play again."; });
    else video.pause();
  });
  get("sound").addEventListener("click", () => { video.muted = !video.muted; controls(); });
  get("time").addEventListener("input", () => {
    if (!clip || !ready) return;
    video.pause();
    video.currentTime = Math.max(0, clip.times[Number(get("time").value)] - clip.sourceOffset);
  });
  get("reveal").addEventListener("input", () => {
    get("stage").style.setProperty("--reveal", `${get("reveal").value}%`);
  });
  for (const example of examples) get("example").add(new Option(example.label, example.id));
  get("example").addEventListener("change", () => choose(get("example").value));
  if (examples.length) {
    get("example").value = examples.find((e) => e.label === "Oboe")?.id || examples[0].id;
    choose(get("example").value);
  } else get("status").textContent = "Examples are unavailable.";
  document.addEventListener("visibilitychange", () => { if (document.hidden) video.pause(); });
})();
