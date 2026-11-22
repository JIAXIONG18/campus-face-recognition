const $ = (id) => document.getElementById(id);
const state = { stream: null, threshold: 0.45, files: [] };

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 2200);
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${tab.dataset.view}`));
    if (tab.dataset.view === "records") refreshRecords();
    if (tab.dataset.view === "enroll") refreshUsers();
    if (tab.dataset.view === "settings") loadSettings();
  });
});

async function api(path, options = {}) {
  const response = await fetch(path, options);
  return response.json();
}

function scene() {
  return document.querySelector('input[name="scene"]:checked').value;
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 } });
    $("video").srcObject = state.stream;
    $("placeholder").classList.add("hidden");
    $("btn-capture").disabled = false;
    $("btn-stop").disabled = false;
    $("btn-camera").disabled = true;
  } catch (err) {
    toast("无法访问摄像头：" + err.message);
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  $("placeholder").classList.remove("hidden");
  $("btn-capture").disabled = true;
  $("btn-stop").disabled = true;
  $("btn-camera").disabled = false;
}

async function submitBlob(blob) {
  const form = new FormData();
  form.append("image", blob, "capture.jpg");
  form.append("scene", scene());
  $("elapsed").textContent = "识别中…";
  const data = await api("/api/recognize", { method: "POST", body: form });
  $("elapsed").textContent = data.elapsed_ms ? `${data.elapsed_ms} ms` : "";
  renderResult(data);
  refreshRecent();
  return data;
}

function renderResult(data) {
  const box = $("result");
  box.className = "result";
  const avatar = box.querySelector(".avatar");
  const title = box.querySelector("h3");
  const desc = box.querySelector("p");
  const score = data.score || 0;
  $("score-text").textContent = data.score !== undefined ? `${(score * 100).toFixed(1)}%` : "--";
  $("score-fill").style.width = `${Math.min(score * 100, 100)}%`;

  if (data.liveness !== undefined) {
    const lv = data.liveness;
    $("liveness-text").textContent = lv.toFixed(2);
    $("liveness-fill").style.width = `${Math.min(Math.max((lv + 1) * 50, 0), 100)}%`;
    $("liveness-fill").style.background = data.liveness_pass ? "linear-gradient(90deg,#0ca678,#38d9a9)" : "linear-gradient(90deg,#e03131,#ff8787)";
  }

  if (!data.success) {
    box.classList.add("fail");
    avatar.textContent = "!";
    title.textContent = "识别失败";
    desc.textContent = data.message || "请重试";
    return;
  }
  if (data.liveness_pass === false) {
    box.classList.add("fail");
    avatar.textContent = "!";
    title.textContent = "活体检测拦截";
    desc.textContent = data.message || "检测到非真实人脸";
    return;
  }
  if (data.passed) {
    box.classList.add("pass");
    avatar.textContent = data.user.name.slice(0, 1);
    title.textContent = data.user.name;
    desc.textContent = `${data.user.user_id} · ${data.user.category} · 识别通过`;
  } else {
    box.classList.add("fail");
    avatar.textContent = "?";
    title.textContent = "未注册人员";
    desc.textContent = `最高相似度 ${(score * 100).toFixed(1)}%，低于阈值 ${(state.threshold * 100).toFixed(0)}%`;
  }
}

function capture() {
  const video = $("video");
  const canvas = $("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => submitBlob(blob), "image/jpeg", 0.92);
}

$("btn-camera").addEventListener("click", startCamera);
$("btn-stop").addEventListener("click", stopCamera);
$("btn-capture").addEventListener("click", capture);

$("file-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  $("placeholder").classList.add("hidden");
  const url = URL.createObjectURL(file);
  const canvas = $("canvas");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
    canvas.style.display = "block";
    $("video").style.display = "none";
    URL.revokeObjectURL(url);
  };
  img.src = url;
  submitBlob(file);
});

async function refreshRecent() {
  const data = await api("/api/records?limit=5");
  $("recent-list").innerHTML = data.records
    .map(
      (r) =>
        `<li><span>${r.name} · ${r.scene}</span><span class="tag ${r.passed ? "ok" : "no"}">${
          r.passed ? "通过" : "未通过"
        } ${(r.score * 100).toFixed(1)}%</span></li>`
    )
    .join("");
}

/* ---------- 录入 ---------- */
$("enroll-files").addEventListener("change", (event) => {
  state.files = Array.from(event.target.files);
  $("preview").innerHTML = state.files.map((f) => `<img src="${URL.createObjectURL(f)}" alt="" />`).join("");
});

$("enroll-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.files.length) return toast("请先选择人脸照片");
  const form = new FormData(event.target);
  state.files.forEach((file) => form.append("images", file));
  const data = await api("/api/enroll", { method: "POST", body: form });
  toast(data.message);
  if (data.success) {
    event.target.reset();
    state.files = [];
    $("preview").innerHTML = "";
    refreshUsers();
    refreshStats();
  }
});

async function refreshUsers() {
  const data = await api("/api/users");
  $("user-count").textContent = `共 ${data.users.length} 人`;
  $("user-list").innerHTML = data.users
    .map(
      (u) => `<li><div class="who"><b>${u.name}</b><span>${u.user_id} · ${u.category} · ${u.photos} 张特征</span></div>
      <button class="icon-btn" data-id="${u.user_id}">删除</button></li>`
    )
    .join("");
  document.querySelectorAll(".icon-btn").forEach((btn) =>
    btn.addEventListener("click", async () => {
      if (!confirm("确认删除该人员及其人脸特征？")) return;
      await api(`/api/users/${btn.dataset.id}`, { method: "DELETE" });
      refreshUsers();
      refreshStats();
      toast("已删除");
    })
  );
}

/* ---------- 记录 ---------- */
function drawTrend(records) {
  const canvas = $("trend");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 700;
  const height = 110;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const days = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    days.push({ key, count: records.filter((r) => r.rec_time.startsWith(key)).length, label: `${d.getMonth() + 1}/${d.getDate()}` });
  }
  const max = Math.max(1, ...days.map((d) => d.count));
  const bw = width / days.length;
  days.forEach((d, i) => {
    const h = (d.count / max) * (height - 24);
    ctx.fillStyle = d.count ? "#3b5bdb" : "#e5e7eb";
    ctx.fillRect(i * bw + bw * 0.2, height - 18 - h, bw * 0.6, h);
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(d.label, i * bw + bw / 2, height - 4);
  });
}

async function refreshRecords() {
  const [records, stats] = await Promise.all([api("/api/records?limit=100"), api("/api/stats")]);
  $("stat-users").textContent = stats.users;
  $("stat-today").textContent = stats.today;
  $("stat-records").textContent = stats.records;
  $("stat-passed").textContent = stats.passed;
  drawTrend(records.records);
  $("record-body").innerHTML = records.records
    .map(
      (r) => `<tr><td>${r.rec_time.replace("T", " ")}</td><td>${r.user_id || "—"}</td><td>${r.name || "—"}</td>
      <td>${r.scene}</td><td>${(r.score * 100).toFixed(1)}%</td>
      <td>${r.liveness == null ? "—" : `<span class="tag ${r.liveness >= 0 ? "ok" : "no"}">${r.liveness >= 0 ? "真实" : "拦截"}</span>`}</td>
      <td><span class="tag ${r.passed ? "ok" : "no"}">${r.passed ? "通过" : "拒绝"}</span></td></tr>`
    )
    .join("");
}

async function refreshStats() {
  const stats = await api("/api/stats");
  $("stat-users").textContent = stats.users;
  $("stat-today").textContent = stats.today;
  $("stat-records").textContent = stats.records;
  $("stat-passed").textContent = stats.passed;
}

$("btn-refresh").addEventListener("click", refreshRecords);

async function loadSettings() {
  const data = await api("/api/config");
  $("thr-attendance").value = data.thresholds["考勤"];
  $("thr-gate").value = data.thresholds["门禁"];
  $("thr-liveness").value = data.liveness_threshold;
}

$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = await api("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thresholds: {
        考勤: parseFloat($("thr-attendance").value),
        门禁: parseFloat($("thr-gate").value),
      },
      liveness_threshold: parseFloat($("thr-liveness").value),
    }),
  });
  if (data.success) {
    state.threshold = data.thresholds["考勤"];
    $("threshold").textContent = (data.thresholds["考勤"] * 100).toFixed(1) + "%";
    $("liveness-threshold").textContent = data.liveness_threshold.toFixed(1);
    toast("设置已保存");
  }
});

async function loadHealth() {
  const data = await api("/api/health");
  state.threshold = data.threshold;
  $("threshold").textContent = (data.threshold * 100).toFixed(1) + "%";
  $("about-device").textContent = data.device;
  $("about-gallery").textContent = `${data.gallery} 条特征`;
  $("about-threshold").textContent = (data.threshold * 100).toFixed(1) + "%";
  $("liveness-threshold").textContent = data.liveness ? data.liveness_threshold.toFixed(1) : "未启用";
  $("about-liveness").textContent = data.liveness ? `单目 RGB 活体检测（阈值 ${data.liveness_threshold.toFixed(1)}）` : "未启用";
}

loadHealth();
refreshRecent();
refreshStats();
refreshUsers();
