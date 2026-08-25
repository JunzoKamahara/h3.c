"use strict";
const $ = (id) => document.getElementById(id);

let pollTimer = null;
let startedAt = 0;

/* --------------------------------------------------------- image slots
   Same upload/preview/drag-drop behavior for both --first-frame and
   --last-frame; `prefix` selects the "first-*"/"last-*" element ids and
   the i18n key namespace (both currently share firstframe.choose/clear/
   hint/unsupported/uploadFailed - only the legend and drop-label text
   differ between the two). */
function makeImageSlot(prefix, i18nPrefix) {
  const slot = { imageId: null };
  const dropzone = $(`${prefix}-dropzone`);
  const label = $(`${prefix}-dropzone-label`);
  const input = $(`${prefix}-image-input`);
  const clearBtn = $(`${prefix}-clear-image`);
  const previewBox = $(`${prefix}-preview-box`);
  const previewImg = $(`${prefix}-preview-image`);
  const errorEl = $(`${prefix}-image-error`);

  async function upload(file) {
    errorEl.classList.add("hidden");
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      errorEl.textContent = t("firstframe.unsupported");
      errorEl.classList.remove("hidden");
      return;
    }
    const res = await fetch("/api/upload-image", {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    });
    const data = await res.json();
    if (!res.ok) {
      errorEl.textContent = data.error || t("firstframe.uploadFailed");
      errorEl.classList.remove("hidden");
      return;
    }
    slot.imageId = data.image_id;
    label.textContent = file.name;
    clearBtn.classList.remove("hidden");
    await refreshPreview();
  }

  async function refreshPreview() {
    if (!slot.imageId) return;
    const profile = document.querySelector('input[name="profile"]:checked').value;
    previewImg.src = `/api/assets/${slot.imageId}/preview?profile=${profile}&t=${Date.now()}`;
    previewBox.classList.remove("hidden");
  }
  slot.refreshPreview = refreshPreview;

  $(`${prefix}-choose-image`).addEventListener("click", () => input.click());
  input.addEventListener("change", (event) => {
    if (event.target.files.length) upload(event.target.files[0]);
  });
  clearBtn.addEventListener("click", () => {
    slot.imageId = null;
    input.value = "";
    label.textContent = t(`${i18nPrefix}.drop`);
    clearBtn.classList.add("hidden");
    previewBox.classList.add("hidden");
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
    if (event.dataTransfer.files.length) upload(event.dataTransfer.files[0]);
  });
  return slot;
}

const firstFrameSlot = makeImageSlot("first", "firstframe");
const lastFrameSlot = makeImageSlot("last", "lastframe");
document.querySelectorAll('input[name="profile"]').forEach((radio) =>
  radio.addEventListener("change", () => {
    firstFrameSlot.refreshPreview();
    lastFrameSlot.refreshPreview();
  }));

function showSection(name) {
  ["form", "progress", "result", "error-box"].forEach((id) =>
    $(id).classList.toggle("hidden", id !== name));
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const config = await res.json();
  const banner = $("config-banner");
  const problems = [];
  if (!config.model_dir_ok) problems.push(t("banner.modelDir", { path: config.model_dir }));
  if (!config.attention_cache_ok) problems.push(t("banner.cache"));
  if (problems.length) {
    banner.textContent = problems.join(" / ");
    banner.classList.remove("hidden");
  }
  $("layers").max = config.layers_max;
  $("layers").min = config.layers_min;
}

$("layers").addEventListener("input", () => {
  $("layers-value").textContent = $("layers").value;
});

$("generate").addEventListener("click", async () => {
  $("form-error").classList.add("hidden");
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("form-error").textContent = t("alert.promptRequired");
    $("form-error").classList.remove("hidden");
    return;
  }
  const body = {
    prompt,
    profile: document.querySelector('input[name="profile"]:checked').value,
    seconds: Number(document.querySelector('input[name="seconds"]:checked').value),
    layers: Number($("layers").value),
    reuse: Number($("reuse").value),
    steps: Number($("steps").value),
    seed: $("seed").value.trim() || null,
    image_id: firstFrameSlot.imageId,
    last_image_id: lastFrameSlot.imageId,
  };
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    $("form-error").textContent = data.error || t("alert.rejected");
    $("form-error").classList.remove("hidden");
    return;
  }
  startPolling(data.id);
});

function startPolling(jobId) {
  showSection("progress");
  startedAt = Date.now();
  $("phase").textContent = t("progress.queued");
  $("bar").value = 0;
  $("log").textContent = "";
  pollTimer = setInterval(() => pollJob(jobId), 1000);
  pollJob(jobId);
}

async function pollJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();
  const elapsedS = Math.floor((Date.now() - startedAt) / 1000);
  const time = `${Math.floor(elapsedS / 60)}:${String(elapsedS % 60).padStart(2, "0")}`;
  $("elapsed").textContent = t("progress.elapsed", { time });
  if (job.phase) {
    $("phase").textContent = job.total ? `${job.phase} (${job.completed}/${job.total})` : job.phase;
    $("bar").max = job.total || 1;
    $("bar").value = job.completed || 0;
  }
  $("log").textContent = job.log_tail.join("\n");
  $("log").scrollTop = $("log").scrollHeight;

  if (job.state === "done") {
    clearInterval(pollTimer);
    showSection("result");
    const src = `/api/jobs/${jobId}/video`;
    $("player").src = src;
    $("download").href = src;
  } else if (job.state === "error") {
    clearInterval(pollTimer);
    showSection("error-box");
    $("error-text").textContent = job.error || t("error.default");
  }
}

$("cancel-back").addEventListener("click", () => {
  if (pollTimer) clearInterval(pollTimer);
  showSection("form");
});
$("again").addEventListener("click", () => showSection("form"));
$("error-back").addEventListener("click", () => showSection("form"));

applyStaticTranslations();
loadConfig();
