"use strict";
(() => {
  const players = [...document.querySelectorAll(".attention-player")];
  if (!players.length) return;
  const examples = window.LEAV_ATTENTION_EXAMPLES || [];
  const loaded = new Map();
  const instances = [];

  function load(example) {
    if (!loaded.has(example.id)) {
      loaded.set(example.id, new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = example.data;
        script.onload = () => {
          script.remove();
          const data = window.LEAV_ATTENTION_DATA[example.id];
          const values = new Float32Array(Uint8Array.from(atob(data.video), (c) => c.charCodeAt(0)).buffer);
          if (values.length !== data.frames * data.grid ** 2) reject(new Error("Incomplete attention data."));
          else resolve({ data, values });
        };
        script.onerror = () => { script.remove(); reject(new Error("Attention data could not be loaded. Please reload the page.")); };
        document.head.append(script);
      }).catch((error) => { loaded.delete(example.id); throw error; }));
    }
    return loaded.get(example.id);
  }

  function setup(root) {
    const get = (role) => root.querySelector(`[data-role="${role}"]`);
    const video = get("video"), canvas = get("map"), context = canvas.getContext("2d");
    const tile = document.createElement("canvas"), tileContext = tile.getContext("2d");
    let clip, weights, version = 0, frame = -1, ready = false, visible = false;
    video.muted = true;

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
    }
    function controls() {
      get("sound").setAttribute("aria-label", video.muted ? "Turn sound on" : "Turn sound off");
      get("sound").setAttribute("aria-pressed", String(!video.muted));
    }
    function start() {
      if (ready && visible && !document.hidden) video.play().catch(() => {});
    }
    async function choose(id) {
      const request = ++version, example = examples.find((item) => item.id === id);
      video.pause();
      video.removeAttribute("src");
      video.load();
      clip = null; ready = false; frame = -1;
      context.clearRect(0, 0, canvas.width, canvas.height);
      root.setAttribute("aria-busy", "true");
      get("status").textContent = `Loading ${example.label.toLowerCase()}…`;
      try {
        const result = await load(example);
        if (request !== version) return;
        ({ data: clip, values: weights } = result);
        tile.width = tile.height = clip.grid;
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
      get("status").textContent = clip.nativeHW && Math.min(...clip.nativeHW) < 672
        ? `Source footage: ${clip.nativeHW[1]} × ${clip.nativeHW[0]}.` : "";
      draw();
      start();
    });
    video.addEventListener("error", () => {
      if (clip) get("status").textContent = "This video could not be loaded. Please reload the page.";
    });
    video.addEventListener("seeked", () => draw());
    video.addEventListener("volumechange", controls);
    if (video.requestVideoFrameCallback) {
      const update = (_, metadata) => { draw(metadata.mediaTime); video.requestVideoFrameCallback(update); };
      video.requestVideoFrameCallback(update);
    } else {
      const update = () => { if (!video.paused) draw(); requestAnimationFrame(update); };
      requestAnimationFrame(update);
    }
    get("sound").addEventListener("click", () => {
      video.muted = !video.muted;
      if (!video.muted) for (const other of instances) if (other.video !== video) other.video.muted = true;
      controls();
    });
    const reveal = () => get("stage").style.setProperty("--reveal", `${get("reveal").value}%`);
    get("reveal").addEventListener("input", reveal);
    reveal();
    const example = examples.find((e) => e.label === root.dataset.example);
    if (example) choose(example.id);
    else get("status").textContent = "This example is unavailable.";

    return {
      video,
      show(isVisible) { visible = isVisible; if (visible) start(); else video.pause(); },
      resume: start,
    };
  }

  for (const root of players) instances.push(setup(root));
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) instances[players.indexOf(entry.target)].show(entry.isIntersecting);
    });
    for (const root of players) observer.observe(root);
  } else for (const instance of instances) instance.show(true);
  document.addEventListener("visibilitychange", () => {
    for (const instance of instances) document.hidden ? instance.video.pause() : instance.resume();
  });
})();
