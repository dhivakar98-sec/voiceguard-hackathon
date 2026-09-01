/* VoiceGuard frontend — vanilla JS, same origin as the API (no CORS). */
"use strict";

const $ = (id) => document.getElementById(id);

const drop = $("drop");
const fileInput = $("fileInput");
const fileRow = $("fileRow");
const analyzeBtn = $("analyzeBtn");
const spinner = analyzeBtn.querySelector(".spinner");
const btnLabel = analyzeBtn.querySelector(".label");
const errorCard = $("errorCard");
const resultCard = $("resultCard");

let selectedFile = null;
let busy = false;

const ACCEPTED = [
  ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".opus",
  ".aiff", ".aif", ".au", ".aac", ".mp4", ".wma", ".webm",
];

/* ----------------------------------------------------------------- engine */
async function loadEngine(attempt = 0) {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    const dot = $("engineDot");
    const text = $("engineText");

    if (h.status === "loading") {
      dot.className = "dot";
      text.textContent = "loading detector…";
      if (attempt < 40) setTimeout(() => loadEngine(attempt + 1), 1500);
      return;
    }
    if (h.backend === "ml") {
      dot.className = "dot ml";
      text.textContent = `ML model · ${shortModel(h.model)}`;
      $("engine").title = `Pretrained ML detector: ${h.model} (${h.device})`;
    } else {
      dot.className = "dot";
      text.textContent = "heuristic engine";
      $("engine").title =
        "Lightweight numpy heuristic detector — install backend/requirements-ml.txt " +
        "for the pretrained ML model." +
        (h.ml_load_error ? `\n\nReason: ${h.ml_load_error}` : "");
    }
  } catch {
    $("engineDot").className = "dot off";
    $("engineText").textContent = "backend offline";
  }
}

const shortModel = (m) => (m || "").split("/").pop().slice(0, 30);

/* ---------------------------------------------------------------- samples */
async function loadSamples() {
  let found = [];
  try {
    const res = await fetch("/api/samples");
    found = (await res.json()).samples || [];
  } catch {
    return; // backend not up yet — the upload box still works
  }
  if (!found.length) return;

  const box = $("sampleLinks");
  for (const name of found) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name.replace(/\.[^.]+$/, "").replace(/_/g, " ");
    b.onclick = async () => {
      const blob = await (await fetch(`/sample_audio/${name}`)).blob();
      setFile(new File([blob], name, { type: "audio/wav" }));
    };
    box.appendChild(b);
  }
  $("samples").hidden = false;
}

/* --------------------------------------------------------- file selection */
function setFile(file) {
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!ACCEPTED.includes(ext)) {
    showError(`'${ext}' is not a supported audio type. Use ${ACCEPTED.slice(0, 6).join(", ")} …`);
    return;
  }
  selectedFile = file;
  $("fileName").textContent = file.name;
  $("fileSize").textContent = `${(file.size / 1024).toFixed(0)} KB`;
  $("player").src = URL.createObjectURL(file);
  fileRow.hidden = false;
  analyzeBtn.disabled = false;
  errorCard.hidden = true;
  resultCard.hidden = true;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileRow.hidden = true;
  analyzeBtn.disabled = true;
  $("player").removeAttribute("src");
}

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
$("clearBtn").addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) setFile(f);
});

/* ---------------------------------------------------------------- analyze */
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile || busy) return;
  busy = true;
  spinner.hidden = false;
  btnLabel.textContent = "Deeply analyzing audio…";
  analyzeBtn.disabled = true;
  errorCard.hidden = true;
  resultCard.hidden = true;

  try {
    const body = new FormData();
    body.append("file", selectedFile, selectedFile.name);
    const res = await fetch("/api/analyze", { method: "POST", body });
    const data = await res.json().catch(() => ({ detail: "The server returned an invalid response." }));
    if (!res.ok) throw new Error(data.detail || `Request failed (HTTP ${res.status}).`);
    render(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the server still running?");
  } finally {
    busy = false;
    spinner.hidden = true;
    btnLabel.textContent = "Analyze audio";
    analyzeBtn.disabled = !selectedFile;
  }
});

function showError(msg) {
  $("errorText").textContent = msg;
  errorCard.hidden = false;
  resultCard.hidden = true;
  errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ----------------------------------------------------------------- render */
function render(d) {
  const fake = d.verdict === "FAKE";

  const badge = $("verdictBadge");
  badge.className = "verdict " + (fake ? "fake" : "human");
  $("verdictEmoji").textContent = fake ? "🚨" : "✅";
  $("verdictText").textContent = fake ? "FAKE · AI-CLONED" : "HUMAN";

  $("confValue").textContent = d.confidence;
  const bar = $("confBar");
  bar.className = "bar-fill " + (fake ? "fake" : "human");
  bar.style.width = "0%";
  requestAnimationFrame(() => { bar.style.width = `${d.confidence}%`; });
  $("confSub").textContent = fake
    ? `The detector is ${d.confidence}% confident this voice was synthesised.`
    : `The detector is ${d.confidence}% confident this is a genuine human voice.`;

  $("factProb").textContent = d.spoof_probability.toFixed(3);
  $("factPeak").textContent = d.max_segment_probability.toFixed(3);
  $("factThr").textContent = `> ${d.threshold.toFixed(2)} = fake`;
  $("factDur").textContent = `${d.analysed_sec.toFixed(1)}s of ${d.duration_sec.toFixed(1)}s`;
  $("factBackend").textContent = d.backend === "ml" ? shortModel(d.model) : "heuristic";
  $("factMs").textContent = `${d.processing_ms} ms`;

  // per-clip warnings (non-speech input, very short clip, ...)
  const warnBox = $("warnBox");
  const warnings = $("warnings");
  warnings.innerHTML = "";
  if (d.warnings && d.warnings.length) {
    d.warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      warnings.appendChild(li);
    });
    warnBox.hidden = false;
  } else {
    warnBox.hidden = true;
  }

  // reasons (heuristic backend only)
  const reasonsBox = $("reasonsBox");
  const reasons = $("reasons");
  reasons.innerHTML = "";
  if (d.reasons && d.reasons.length) {
    d.reasons.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      reasons.appendChild(li);
    });
    reasonsBox.hidden = false;
  } else {
    reasonsBox.hidden = true;
  }

  // spectrogram
  if (d.spectrogram_png_base64) {
    $("spec").src = `data:image/png;base64,${d.spectrogram_png_base64}`;
    $("specBox").hidden = false;
  } else {
    $("specBox").hidden = true;
  }

  // per-segment scores
  const segs = $("segments");
  segs.innerHTML = "";
  if (d.segments && d.segments.length > 1) {
    d.segments.forEach((s) => {
      const row = document.createElement("div");
      row.className = "seg";
      const pct = Math.round(s.spoof_probability * 100);
      row.innerHTML =
        `<span class="seg-time">${s.start.toFixed(1)}s – ${s.end.toFixed(1)}s</span>` +
        `<span class="seg-bar"><i style="width:${pct}%;background:${
          s.spoof_probability > 0.5 ? "var(--red)" : "var(--green)"
        }"></i></span>` +
        `<span class="seg-val">${pct}%</span>`;
      segs.appendChild(row);
    });
    $("segBox").hidden = false;
  } else {
    $("segBox").hidden = true;
  }

  $("note").textContent = d.note;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

loadEngine();
loadSamples();
