const API_BASE = "http://localhost:8000";
const CAPTURE_INTERVAL_MS = 1000; // capture one frame per second

let selectedMinutes = 30;
let frames = [];
let captureTimer = null;
let countdownTimer = null;
let sessionSeconds = 0;
let currentSongId = "";

// ── Views ──────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ── Duration input ─────────────────────────────────────
document.getElementById("duration-input").addEventListener("change", (e) => {
  selectedMinutes = Math.max(1, Math.min(180, parseInt(e.target.value) || 30));
  e.target.value = selectedMinutes;
});

// ── Start session ──────────────────────────────────────
document.getElementById("start-btn").addEventListener("click", async () => {
  selectedMinutes = Math.max(1, Math.min(180, parseInt(document.getElementById("duration-input").value) || 30));
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById("webcam").srcObject = stream;
    startSession(stream);
    showView("view-session");
  } catch {
    alert("Camera access is required. Please allow camera permissions.");
  }
});

function startSession(stream) {
  frames = [];
  sessionSeconds = selectedMinutes * 60;
  updateTimerDisplay(sessionSeconds);

  const video = document.getElementById("webcam");
  const canvas = document.createElement("canvas");

  // Capture a frame every second
  captureTimer = setInterval(() => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const b64 = canvas.toDataURL("image/jpeg", 0.7);
    frames.push(b64);
    document.getElementById("frame-count").textContent = `${frames.length} frames captured`;
  }, CAPTURE_INTERVAL_MS);

  // Countdown
  countdownTimer = setInterval(() => {
    sessionSeconds--;
    updateTimerDisplay(sessionSeconds);
    if (sessionSeconds <= 0) endSession(stream);
  }, 1000);

  document.getElementById("stop-btn").onclick = () => endSession(stream);
}

function updateTimerDisplay(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  document.getElementById("timer").textContent = `${m}:${s}`;
}

function endSession(stream) {
  clearInterval(captureTimer);
  clearInterval(countdownTimer);
  stream.getTracks().forEach((t) => t.stop());
  analyzeFrames();
}

// ── Analyze ────────────────────────────────────────────
async function analyzeFrames() {
  showView("view-loading");

  try {
    const res = await fetch(`${API_BASE}/face`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, duration: selectedMinutes * 60 }),
    });

    if (!res.ok) throw new Error(`API error ${res.status}`);
    const result = await res.json();
    await showResult(result);
  } catch (err) {
    alert("Something went wrong: " + err.message);
    restart();
  }
}

// ── Result ─────────────────────────────────────────────
async function showResult(result) {
  const { top_emotions, song } = result;
  currentSongId = song;
  showView("view-result");

  // Render emotion chips
  const chipsEl = document.getElementById("emotion-chips");
  chipsEl.innerHTML = "";
  (top_emotions || []).forEach((emo, i) => {
    const chip = document.createElement("span");
    chip.className = "emotion-chip";
    if (i === 0) chip.classList.add("primary");
    chip.textContent = emo;
    chipsEl.appendChild(chip);
  });

  const { artist, title } = parseSongId(song);
  document.getElementById("song-title").textContent = formatWords(title);
  document.getElementById("song-artist").textContent = formatWords(artist);

  fetchAlbumArt(artist, title);
  fetchLyrics(artist, title);
}

function parseSongId(id) {
  // IDs are like "john_lennon_imagine" or "alan_parsons_project_so_far_away"
  // Use iTunes to resolve; locally just split on first underscore run heuristically.
  const parts = id.split("_");
  // Guess: artist = first 2 tokens, title = rest (good enough for iTunes search)
  const artist = parts.slice(0, 2).join(" ");
  const title = parts.slice(2).join(" ");
  return { artist, title };
}

function formatWords(str) {
  return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

async function fetchAlbumArt(artist, title) {
  const query = encodeURIComponent(`${artist} ${title}`);
  try {
    const res = await fetch(
      `https://itunes.apple.com/search?term=${query}&entity=song&limit=1`
    );
    const data = await res.json();
    const result = data.results?.[0];
    if (result) {
      const artUrl = result.artworkUrl100.replace("100x100", "600x600");
      document.getElementById("album-cover").src = artUrl;
      document.getElementById("album-bg").style.backgroundImage = `url(${artUrl})`;

      // Override artist/title with iTunes accurate data
      document.getElementById("song-title").textContent = result.trackName;
      document.getElementById("song-artist").textContent = result.artistName;

      // Load the 30s preview into the audio element for in-app playback
      if (result.previewUrl) loadAudio(result.previewUrl);
    }
  } catch {
    // leave placeholder
  }
}

// ── In-app audio playback ──────────────────────────────
function loadAudio(url) {
  const audio = document.getElementById("audio-player");
  audio.src = url;
  audio.load();

  audio.addEventListener("loadedmetadata", () => {
    document.getElementById("time-total").textContent = formatTime(audio.duration);
  });

  audio.addEventListener("timeupdate", () => {
    const pct = (audio.currentTime / audio.duration) * 100;
    document.getElementById("progress-fill").style.width = `${pct}%`;
    document.getElementById("time-current").textContent = formatTime(audio.currentTime);
    updateLyricSync(audio.currentTime, audio.duration);
  });

  audio.addEventListener("ended", () => {
    document.getElementById("play-btn").textContent = "▶";
  });

  // Seek on progress bar click
  document.getElementById("progress-bar").onclick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    audio.currentTime = pct * audio.duration;
  };
}

function togglePlay() {
  const audio = document.getElementById("audio-player");
  const btn = document.getElementById("play-btn");
  if (!audio.src) return;
  if (audio.paused) {
    audio.play();
    btn.textContent = "❚❚";
  } else {
    audio.pause();
    btn.textContent = "▶";
  }
}

function formatTime(s) {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

let lyricLines = [];

async function fetchLyrics(artist, title) {
  const lyricsEl = document.getElementById("lyrics");
  lyricLines = [];

  try {
    const res = await fetch(
      `https://api.lyrics.ovh/v1/${encodeURIComponent(artist)}/${encodeURIComponent(title)}`
    );
    const data = await res.json();
    if (data.lyrics) {
      lyricsEl.classList.remove("not-found");
      // Split into non-empty lines and render each as a span
      const lines = data.lyrics
        .trim()
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0);

      lyricsEl.innerHTML = "";
      lines.forEach((line, i) => {
        const el = document.createElement("div");
        el.className = "lyric-line";
        el.textContent = line;
        el.dataset.index = i;
        lyricsEl.appendChild(el);
      });
      lyricLines = Array.from(lyricsEl.querySelectorAll(".lyric-line"));
    } else {
      throw new Error("no lyrics");
    }
  } catch {
    lyricsEl.classList.add("not-found");
    lyricsEl.textContent = "Lyrics not available";
  }
}

function updateLyricSync(currentTime, duration) {
  if (!lyricLines.length || !duration || !isFinite(duration)) return;
  const pct = currentTime / duration;
  const activeIdx = Math.min(
    lyricLines.length - 1,
    Math.floor(pct * lyricLines.length)
  );

  lyricLines.forEach((el, i) => {
    el.classList.toggle("active", i === activeIdx);
    el.classList.toggle("passed", i < activeIdx);
  });

  // Auto-scroll active line into view center
  const active = lyricLines[activeIdx];
  if (active) {
    active.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function openOnYouTube() {
  const title = document.getElementById("song-title").textContent;
  const artist = document.getElementById("song-artist").textContent;
  const q = encodeURIComponent(`${title} ${artist}`);
  window.open(`https://www.youtube.com/results?search_query=${q}`, "_blank");
}

function restart() {
  frames = [];
  showView("view-setup");
}
