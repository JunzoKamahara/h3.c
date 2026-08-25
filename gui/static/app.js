"use strict";
const $ = (id) => document.getElementById(id);

let pollTimer = null;
let startedAt = 0;
let imageId = null;

/* ------------------------------------------------------------ first frame */
async function uploadImage(file) {
  $("image-error").classList.add("hidden");
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    $("image-error").textContent = "JPEG / PNG / WebP のみ対応しています。";
    $("image-error").classList.remove("hidden");
    return;
  }
  const res = await fetch("/api/upload-image", {
    method: "POST",
    headers: { "Content-Type": file.type },
    body: file,
  });
  const data = await res.json();
  if (!res.ok) {
    $("image-error").textContent = data.error || "画像のアップロードに失敗しました。";
    $("image-error").classList.remove("hidden");
    return;
  }
  imageId = data.image_id;
  $("dropzone-label").textContent = file.name;
  $("clear-image").classList.remove("hidden");
  await refreshPreview();
}

async function refreshPreview() {
  if (!imageId) return;
  const profile = document.querySelector('input[name="profile"]:checked').value;
  $("preview-image").src = `/api/assets/${imageId}/preview?profile=${profile}&t=${Date.now()}`;
  $("preview-box").classList.remove("hidden");
}

$("choose-image").addEventListener("click", () => $("image-input").click());
$("image-input").addEventListener("change", (event) => {
  if (event.target.files.length) uploadImage(event.target.files[0]);
});
$("clear-image").addEventListener("click", () => {
  imageId = null;
  $("image-input").value = "";
  $("dropzone-label").textContent = "ここに画像をドロップ、またはクリックして選択";
  $("clear-image").classList.add("hidden");
  $("preview-box").classList.add("hidden");
});
const dropzone = $("dropzone");
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag");
  if (event.dataTransfer.files.length) uploadImage(event.dataTransfer.files[0]);
});
document.querySelectorAll('input[name="profile"]').forEach((radio) =>
  radio.addEventListener("change", refreshPreview));

function showSection(name) {
  ["form", "progress", "result", "error-box"].forEach((id) =>
    $(id).classList.toggle("hidden", id !== name));
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const config = await res.json();
  const banner = $("config-banner");
  const problems = [];
  if (!config.model_dir_ok) problems.push(`モデルディレクトリが見つかりません: ${config.model_dir}`);
  if (!config.attention_cache_ok) problems.push("int8キャッシュが見つかりません(未生成なら低速なレジデントロードにフォールバックします)");
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
    $("form-error").textContent = "プロンプトを入力してください。";
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
    image_id: imageId,
  };
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    $("form-error").textContent = data.error || "リクエストが拒否されました。";
    $("form-error").classList.remove("hidden");
    return;
  }
  startPolling(data.id);
});

function startPolling(jobId) {
  showSection("progress");
  startedAt = Date.now();
  $("phase").textContent = "待機中...";
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
  $("elapsed").textContent = `経過: ${Math.floor(elapsedS / 60)}:${String(elapsedS % 60).padStart(2, "0")}`;
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
    $("error-text").textContent = job.error || "生成に失敗しました。";
  }
}

$("cancel-back").addEventListener("click", () => {
  if (pollTimer) clearInterval(pollTimer);
  showSection("form");
});
$("again").addEventListener("click", () => showSection("form"));
$("error-back").addEventListener("click", () => showSection("form"));

loadConfig();
