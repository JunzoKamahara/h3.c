"use strict";
const $ = (id) => document.getElementById(id);

let pollTimer = null;
let startedAt = 0;

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
