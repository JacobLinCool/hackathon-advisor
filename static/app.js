const form = document.querySelector("#turn-form");
const atlasView = document.querySelector("#atlas-view");
const advisorView = document.querySelector("#advisor-view");
const openAdvisorButton = document.querySelector("#open-advisor");
const openAtlasButton = document.querySelector("#open-atlas");
const refreshDashboardButton = document.querySelector("#refresh-dashboard");
const atlasStatusEl = document.querySelector("#atlas-status");
const atlasStatsEl = document.querySelector("#atlas-stats");
const atlasClustersEl = document.querySelector("#atlas-clusters");
const atlasQuestsEl = document.querySelector("#atlas-quests");
const atlasSvgEl = document.querySelector("#atlas-svg");
const atlasDetailEl = document.querySelector("#atlas-detail");
const atlasReportEl = document.querySelector("#atlas-report");
const atlasRefreshProgressEl = document.querySelector("#atlas-refresh-progress");
const input = document.querySelector("#message");
const submit = document.querySelector("#submit");
const ink = document.querySelector("#ink");
const corrections = document.querySelector("#corrections");
const projectsEl = document.querySelector("#projects");
const whitespaceEl = document.querySelector("#whitespace");
const ideasEl = document.querySelector("#ideas");
const goalsEl = document.querySelector("#goals");
const profileEl = document.querySelector("#profile");
const woodMapEl = document.querySelector("#wood-map");
const scoreEl = document.querySelector("#score");
const planEl = document.querySelector("#plan");
const provenanceEl = document.querySelector("#provenance");
const verdictEl = document.querySelector("#verdict");
const overallEl = document.querySelector("#overall");
const sealEl = document.querySelector("#seal");
const sealVerdictEl = document.querySelector("#seal-verdict");
const sealCopyEl = document.querySelector("#seal-copy");
const verdictStampEl = document.querySelector("#verdict-stamp");
const spreadEl = document.querySelector("#spread");
const ideaCountEl = document.querySelector("#idea-count");
const goalCountEl = document.querySelector("#goal-count");
const demoButton = document.querySelector("#load-demo");
const exportButton = document.querySelector("#export-artifact");
const exportNotesButton = document.querySelector("#export-notes");
const exportChapterButton = document.querySelector("#export-chapter");
const resetButton = document.querySelector("#reset-session");
const recordVoiceButton = document.querySelector("#record-voice");
const uploadVoiceButton = document.querySelector("#upload-voice");
const voiceFileInput = document.querySelector("#voice-file");
const turnProgressEl = document.querySelector("#turn-progress");
const turnStageIconEl = document.querySelector("#turn-stage-icon");
const turnStageTextEl = document.querySelector("#turn-stage-text");
const turnTokensEl = document.querySelector("#turn-tokens");
const turnEtaEl = document.querySelector("#turn-eta");
const turnBarFillEl = document.querySelector("#turn-bar-fill");
const toolChipsEl = document.querySelector("#tool-chips");

const SESSION_STORAGE_KEY = "hackathon-advisor-session-v2";
const STAGE_ICONS = { planning: "🪶", running_tool: "🔧", writing: "✍️" };
const FIELD_NOTES_FILENAME = "hackathon-advisor-field-notes.md";
const CHAPTER_FILENAME = "hackathon-advisor-chapter.md";
const PNG_EXPORT_LABEL = "PNG";

let session = {};
let currentArtifact = null;
let goalOptions = [];
let goalProfiles = [];
let goalProfileById = new Map();
let profileFields = [];
let turnWatchdog = null;
let sawTurnToken = false;
let bootstrapData = null;
let sessionRevision = 0;
let sessionControlsLocked = false;
let voiceBusy = false;
let voiceRecorder = null;
let voiceStream = null;
let voiceChunks = [];
let voiceRecordingState = "idle";
let decodeStartedAt = 0;
let turnProgressTimer = null;
let dashboardData = null;
let selectedClusterId = "";
let selectedQuestId = "";
let selectedProjectId = "";
let dashboardRefreshTimer = null;

setVoiceRecordingState("idle");
setupViewRouting();
loadDashboard().catch(handleDashboardError);
bootstrap().catch(handleBootstrapError);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (sessionControlsLocked || submit.disabled || input.disabled) return;
  const message = input.value.trim();
  if (!message) return;
  await runTurn(message);
});

input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  form.requestSubmit();
});

document.querySelectorAll(".mobile-nav [data-tab]").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab || "page"));
});

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", async () => {
    await runCommand(button.dataset.command || "");
  });
});

setupMenus();

demoButton.addEventListener("click", async () => {
  await loadDemoSession();
});

exportButton.addEventListener("click", () => {
  if (!currentArtifact) return;
  exportArtifact(currentArtifact);
});

exportNotesButton.addEventListener("click", () => exportNotes());

exportChapterButton.addEventListener("click", () => exportChapter());

resetButton.addEventListener("click", () => {
  resetSession();
});

openAdvisorButton?.addEventListener("click", () => {
  window.location.hash = "advisor";
});

openAtlasButton?.addEventListener("click", () => {
  window.location.hash = "atlas";
});

refreshDashboardButton?.addEventListener("click", async () => {
  await startDashboardRefresh();
});

recordVoiceButton.addEventListener("click", async () => {
  await toggleVoiceRecording();
});

uploadVoiceButton.addEventListener("click", () => {
  if (uploadVoiceButton.disabled || voiceBusy || sessionControlsLocked || voiceRecordingState !== "idle") return;
  voiceFileInput.click();
});

voiceFileInput.addEventListener("change", async () => {
  const file = voiceFileInput.files?.[0] || null;
  voiceFileInput.value = "";
  if (!file) return;
  await transcribeVoiceBlob(file, file.name || "voice-note.audio");
});

goalsEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.dataset.goal) return;
  bumpSessionRevision();
  const checked = new Set(
    Array.from(goalsEl.querySelectorAll("input[data-goal]:checked")).map((input) => input.dataset.goal),
  );
  session.goals = goalOptions.filter((option) => checked.has(option));
  syncCurrentIdeaGoals();
  invalidateCurrentSeal("Goals updated. Press Ink or Plan to refresh the score.");
  saveSession();
  renderGoals(session.goals);
  renderIdeas(session.ideas || []);
});

profileEl.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.dataset.profileField) return;
  bumpSessionRevision();
  const profile = { ...(session.profile || {}) };
  const value = target.value.trim();
  if (value) {
    profile[target.dataset.profileField] = value;
  } else {
    delete profile[target.dataset.profileField];
  }
  session.profile = profile;
  invalidateCurrentPlan("Profile updated. Press Plan to refresh the build path.");
  saveSession();
});

ideasEl.addEventListener("click", (event) => {
  const card = event.target.closest("[data-idea-id]");
  if (!(card instanceof HTMLElement) || !ideasEl.contains(card)) return;
  selectIdea(card.dataset.ideaId || "");
});

whitespaceEl.addEventListener("click", async (event) => {
  const card = event.target.closest("[data-gap-prompt]");
  if (!(card instanceof HTMLButtonElement) || !whitespaceEl.contains(card)) return;
  if (card.disabled) return;
  await runTurn(card.dataset.gapPrompt || "");
});

function setupViewRouting() {
  window.addEventListener("hashchange", applyCurrentView);
  applyCurrentView();
}

function applyCurrentView() {
  const view = window.location.hash.replace(/^#/, "") === "advisor" ? "advisor" : "atlas";
  document.body.dataset.view = view;
  if (atlasView) atlasView.hidden = view !== "atlas";
  if (advisorView) advisorView.hidden = view !== "advisor";
  if (view === "advisor" && input && !sessionControlsLocked) {
    window.setTimeout(() => input.focus(), 30);
  }
}

async function loadDashboard() {
  const response = await fetch("/api/dashboard");
  if (!response.ok) throw new Error(`dashboard failed with ${response.status}`);
  const data = await response.json();
  dashboardData = data;
  renderDashboard(data);
  renderRefreshState(data.refresh || {});
  if (data.refresh?.status === "running") scheduleRefreshPoll();
}

function handleDashboardError(error) {
  dashboardData = null;
  if (atlasStatusEl) atlasStatusEl.textContent = `Atlas could not load: ${error.message}`;
  if (atlasSvgEl) atlasSvgEl.innerHTML = "";
  if (atlasStatsEl) atlasStatsEl.innerHTML = "";
  if (atlasDetailEl) atlasDetailEl.innerHTML = `<p>Reload the page to try again.</p>`;
}

async function startDashboardRefresh() {
  if (!refreshDashboardButton || refreshDashboardButton.disabled) return;
  refreshDashboardButton.disabled = true;
  if (atlasStatusEl) atlasStatusEl.textContent = "Starting refresh.";
  try {
    const response = await fetch("/api/dashboard/refresh", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `refresh failed with ${response.status}`);
    renderRefreshState(data);
    scheduleRefreshPoll();
  } catch (error) {
    if (atlasStatusEl) atlasStatusEl.textContent = `Refresh could not start: ${error.message}`;
    if (atlasRefreshProgressEl) {
      atlasRefreshProgressEl.hidden = false;
      atlasRefreshProgressEl.textContent = error.message;
    }
    refreshDashboardButton.disabled = false;
  }
}

function scheduleRefreshPoll() {
  if (dashboardRefreshTimer) window.clearTimeout(dashboardRefreshTimer);
  dashboardRefreshTimer = window.setTimeout(pollDashboardRefresh, 1400);
}

async function pollDashboardRefresh() {
  try {
    const response = await fetch("/api/dashboard/refresh");
    if (!response.ok) throw new Error(`refresh status failed with ${response.status}`);
    const state = await response.json();
    renderRefreshState(state);
    if (state.status === "running") {
      scheduleRefreshPoll();
      return;
    }
    if (state.status === "succeeded") {
      await loadDashboard();
    }
  } catch (error) {
    if (atlasStatusEl) atlasStatusEl.textContent = `Refresh status unavailable: ${error.message}`;
  } finally {
    if (_refreshIsSettled()) refreshDashboardButton.disabled = false;
  }
}

function _refreshIsSettled() {
  const status = String(dashboardData?.refresh?.status || "");
  return status !== "running";
}

function renderRefreshState(state) {
  if (dashboardData) dashboardData.refresh = state || {};
  const status = String(state?.status || "idle");
  const stage = state?.stage_label || state?.stage || "";
  if (atlasStatusEl) {
    if (status === "running") {
      atlasStatusEl.textContent = stage ? `Refresh running: ${stage}.` : "Refresh running.";
    } else if (status === "succeeded") {
      atlasStatusEl.textContent = `Atlas refreshed: ${state.result?.project_count || "current"} projects mapped.`;
    } else if (status === "failed") {
      atlasStatusEl.textContent = `Refresh failed: ${state.error || "unknown error"}`;
    } else if (dashboardData) {
      atlasStatusEl.textContent = atlasProvenanceCopy(dashboardData);
    }
  }
  if (atlasRefreshProgressEl) {
    const show = status === "running" || status === "failed";
    atlasRefreshProgressEl.hidden = !show;
    atlasRefreshProgressEl.textContent =
      status === "running"
        ? `${stage || "Working"} · run ${state.run_id || ""}`
        : state.error || "";
  }
  if (refreshDashboardButton) refreshDashboardButton.disabled = status === "running";
}

function renderDashboard(data) {
  if (!data?.points?.length) {
    handleDashboardError(new Error("empty dashboard payload"));
    return;
  }
  if (!selectedProjectId) selectedProjectId = mostLikedPoint(data.points)?.id || data.points[0].id;
  renderAtlasStats(data);
  renderAtlasClusters(data);
  renderAtlasQuests(data);
  renderAtlasSvg(data);
  renderAtlasDetail(currentAtlasPoint(data));
  renderAtlasReport(data);
  if (atlasStatusEl) atlasStatusEl.textContent = atlasProvenanceCopy(data);
}

function atlasProvenanceCopy(data) {
  const count = Number(data.project_count || data.points?.length || 0);
  const updated = shortDate(data.provenance?.snapshot_generated_at || data.generated_at);
  return `${count} projects mapped · ${data.layout?.algorithm || "layout"} · updated ${updated}`;
}

function renderAtlasStats(data) {
  if (!atlasStatsEl) return;
  const analyzed = data.quest_report?.status === "analyzed";
  const questCount = (data.quest_report?.quests || []).filter((quest) => Number(quest.project_count || 0) > 0).length;
  atlasStatsEl.innerHTML = `
    <div class="atlas-stat"><strong>${Number(data.project_count || 0)}</strong><span>Projects</span></div>
    <div class="atlas-stat"><strong>${Number(data.clusters?.length || 0)}</strong><span>Clusters</span></div>
    <div class="atlas-stat"><strong>${Number(data.links?.length || 0)}</strong><span>Near links</span></div>
    <div class="atlas-stat"><strong>${analyzed ? questCount : "..."}</strong><span>Quest groups</span></div>
  `;
}

function renderAtlasClusters(data) {
  if (!atlasClustersEl) return;
  const allActive = !selectedClusterId;
  atlasClustersEl.innerHTML = "";
  atlasClustersEl.append(
    atlasFilterButton({
      label: "All clusters",
      meta: `${data.project_count || data.points.length} projects`,
      active: allActive,
      onClick: () => {
        selectedClusterId = "";
        renderDashboard(data);
      },
    }),
  );
  for (const cluster of data.clusters || []) {
    atlasClustersEl.append(
      atlasFilterButton({
        label: cluster.label || cluster.id,
        meta: `${cluster.project_count || 0} projects`,
        active: selectedClusterId === cluster.id,
        onClick: () => {
          selectedClusterId = selectedClusterId === cluster.id ? "" : cluster.id;
          renderDashboard(data);
        },
      }),
    );
  }
}

function renderAtlasQuests(data) {
  if (!atlasQuestsEl) return;
  const quests = data.quest_report?.quests || [];
  const analyzed = data.quest_report?.status === "analyzed";
  atlasQuestsEl.innerHTML = "";
  atlasQuestsEl.append(
    atlasFilterButton({
      label: "All quests",
      meta: analyzed ? "No quest filter" : "Refresh to analyze",
      active: !selectedQuestId,
      onClick: () => {
        selectedQuestId = "";
        renderDashboard(data);
      },
    }),
  );
  for (const quest of quests) {
    atlasQuestsEl.append(
      atlasFilterButton({
        label: quest.label || quest.id,
        meta: analyzed ? `${quest.project_count || 0} projects` : "Not analyzed",
        active: selectedQuestId === quest.id,
        onClick: () => {
          selectedQuestId = selectedQuestId === quest.id ? "" : quest.id;
          renderDashboard(data);
        },
      }),
    );
  }
}

function atlasFilterButton({ label, meta, active, onClick }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `atlas-filter ${active ? "active" : ""}`;
  button.innerHTML = `<strong>${escapeHtml(label)}</strong><span>${escapeHtml(meta || "")}</span>`;
  button.addEventListener("click", onClick);
  return button;
}

function renderAtlasSvg(data) {
  if (!atlasSvgEl) return;
  atlasSvgEl.innerHTML = "";
  const pointsById = new Map((data.points || []).map((point) => [point.id, point]));
  const visible = new Set(visibleAtlasPoints(data).map((point) => point.id));
  const clusterIndex = new Map((data.clusters || []).map((cluster, index) => [cluster.id, index]));

  for (const link of data.links || []) {
    const source = pointsById.get(link.source);
    const target = pointsById.get(link.target);
    if (!source || !target) continue;
    const line = svgEl("line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("class", `atlas-link ${visible.has(source.id) && visible.has(target.id) ? "" : "dim"}`);
    atlasSvgEl.append(line);
  }

  for (const point of data.points || []) {
    const circle = svgEl("circle");
    circle.setAttribute("cx", point.x);
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", atlasPointRadius(point));
    circle.setAttribute("fill", atlasColor(clusterIndex.get(point.cluster_id) || 0));
    circle.setAttribute(
      "class",
      `atlas-dot ${visible.has(point.id) ? "" : "dim"} ${point.id === selectedProjectId ? "selected" : ""}`,
    );
    circle.setAttribute("tabindex", "0");
    circle.setAttribute("role", "button");
    circle.setAttribute("aria-label", point.title || point.id);
    circle.addEventListener("mouseenter", () => renderAtlasDetail(point));
    circle.addEventListener("focus", () => renderAtlasDetail(point));
    circle.addEventListener("click", () => {
      selectedProjectId = point.id;
      renderDashboard(data);
    });
    circle.append(svgTitle(point.title || point.id));
    atlasSvgEl.append(circle);
  }

  for (const point of labelAtlasPoints(data)) {
    const text = svgEl("text");
    text.setAttribute("x", boundedPercent(point.x + 1.4));
    text.setAttribute("y", boundedPercent(point.y - 1.1));
    text.setAttribute("class", "atlas-label");
    text.textContent = atlasShortTitle(point.title || point.id);
    atlasSvgEl.append(text);
  }
}

function visibleAtlasPoints(data) {
  return (data.points || []).filter((point) => {
    const clusterMatch = !selectedClusterId || point.cluster_id === selectedClusterId;
    const questMatch = !selectedQuestId || (point.quest_ids || []).includes(selectedQuestId);
    return clusterMatch && questMatch;
  });
}

function labelAtlasPoints(data) {
  const visible = visibleAtlasPoints(data);
  return [...visible].sort((left, right) => Number(right.likes || 0) - Number(left.likes || 0)).slice(0, 12);
}

function currentAtlasPoint(data) {
  return (data.points || []).find((point) => point.id === selectedProjectId) || mostLikedPoint(data.points || []);
}

function mostLikedPoint(points) {
  return [...(points || [])].sort((left, right) => Number(right.likes || 0) - Number(left.likes || 0))[0] || null;
}

function renderAtlasDetail(point) {
  if (!atlasDetailEl) return;
  if (!point) {
    atlasDetailEl.innerHTML = `<p>Select a project dot to inspect its cluster and quest matches.</p>`;
    return;
  }
  const quests = (point.quest_matches || [])
    .map((match) => {
      const confidence = (Number(match.confidence) * 100).toFixed(0);
      return `<span>${escapeHtml(atlasQuestLabel(match.quest))} ${confidence}%</span>`;
    })
    .join("");
  const tags = [...(point.models || []).slice(0, 3), ...visibleProjectTags(point.tags || []).slice(0, 3)]
    .map((tag) => `<span>${escapeHtml(tag)}</span>`)
    .join("");
  atlasDetailEl.innerHTML = `
    <h2>${escapeHtml(point.title || "Untitled project")}</h2>
    ${point.summary ? `<p>${escapeHtml(point.summary)}</p>` : `<p>${escapeHtml(point.id || "")}</p>`}
    <p>${Number(point.likes || 0)} likes · ${escapeHtml(point.sdk || "unknown sdk")}</p>
    <p><a href="${escapeAttribute(point.url || "#")}" target="_blank" rel="noreferrer">Open Space</a></p>
    <div class="atlas-tags">${quests || `<span>Quest analysis pending</span>`}</div>
    <div class="atlas-tags">${tags}</div>
  `;
}

function visibleProjectTags(tags) {
  return (tags || []).filter((tag) => !String(tag || "").toLowerCase().startsWith("region:"));
}

function renderAtlasReport(data) {
  if (!atlasReportEl) return;
  const cluster = selectedClusterId
    ? (data.clusters || []).find((item) => item.id === selectedClusterId)
    : (data.clusters || [])[0];
  if (!cluster) {
    atlasReportEl.innerHTML = `<p>No cluster report is available.</p>`;
    return;
  }
  const projects = (cluster.representative_projects || [])
    .map(
      (project) =>
        `<p><a href="${escapeAttribute(project.url || "#")}" target="_blank" rel="noreferrer">` +
        `${escapeHtml(project.title || project.id)}</a></p>`,
    )
    .join("");
  atlasReportEl.innerHTML = `
    <h2>${escapeHtml(cluster.label || cluster.id)}</h2>
    <p>${Number(cluster.project_count || 0)} projects · ${escapeHtml(
      (cluster.keywords || []).join(", ") || "mixed signals",
    )}</p>
    ${projects}
  `;
}

function svgEl(tagName) {
  return document.createElementNS("http://www.w3.org/2000/svg", tagName);
}

function svgTitle(text) {
  const title = svgEl("title");
  title.textContent = text;
  return title;
}

function atlasColor(index) {
  const palette = [
    "#9a2b22",
    "#b07d12",
    "#2f6b41",
    "#6f4b1d",
    "#3f8453",
    "#74201b",
    "#8a714c",
    "#d8a226",
    "#5d4528",
    "#7c6849",
  ];
  return palette[index % palette.length];
}

function atlasQuestLabel(questId) {
  const quest = (dashboardData?.quest_report?.quests || []).find((item) => item.id === questId);
  return quest?.label || questId;
}

function atlasPointRadius(point) {
  return (0.62 + Math.min(0.72, Math.sqrt(Number(point.likes || 0)) * 0.12)).toFixed(3);
}

function atlasShortTitle(title) {
  const cleaned = String(title || "").trim();
  return cleaned.length > 22 ? `${cleaned.slice(0, 20).trim()}...` : cleaned;
}

async function runTurn(message) {
  if (sessionControlsLocked) return false;
  bumpSessionRevision();
  setActiveTab("page");
  input.value = "";
  submit.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  ink.classList.remove("bleed", "gold");
  corrections.textContent = "";
  planEl.innerHTML = "";
  delete session.ui_status;
  resetTurnProgress();
  startTurnWatchdog();

  let completed = false;
  try {
    const response = await fetch("/api/agent-turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_json: JSON.stringify(session),
      }),
    });
    if (!response.ok) throw new Error(`advisor failed with ${response.status}`);
    if (!response.body) throw new Error("advisor stream was empty");

    for await (const raw of readNdjson(response.body)) {
      handleEvent(JSON.parse(raw));
    }
    completed = true;
  } catch (error) {
    clearTurnWatchdog();
    ink.textContent = `The advisor could not answer: ${error.message}`;
    ink.classList.remove("thinking");
    ink.classList.add("bleed");
  } finally {
    clearTurnWatchdog();
    hideTurnProgress();
    submit.disabled = false;
    setSessionControlsDisabled(false);
    setCommandDisabled(false);
    input.focus();
  }
  return completed;
}

async function* readNdjson(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) yield line;
      newlineIndex = buffer.indexOf("\n");
    }
  }

  buffer += decoder.decode();
  const finalLine = buffer.trim();
  if (finalLine) yield finalLine;
}

async function runCommand(command) {
  if (!command) return;
  const draft = input.value.trim();
  if (draft) {
    const savedDraft = await runTurn(draft);
    if (!savedDraft) return;
  }
  await runTurn(command);
}

async function toggleVoiceRecording() {
  if (voiceRecordingState === "recording" && voiceRecorder?.state === "recording") {
    stopVoiceRecording();
    return;
  }
  if (voiceRecordingState !== "idle") return;
  await startVoiceRecording();
}

async function startVoiceRecording() {
  if (!bootstrapData || sessionControlsLocked || voiceBusy || voiceRecordingState !== "idle") return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setSessionStatus("Voice recording is not available in this browser. Upload a voice note instead.");
    return;
  }
  setVoiceRecordingState("starting");
  submit.disabled = true;
  setCommandDisabled(true);
  try {
    voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    voiceChunks = [];
    const mimeType = recordingMimeType();
    voiceRecorder = new MediaRecorder(voiceStream, mimeType ? { mimeType } : undefined);
    voiceRecorder.addEventListener("dataavailable", (event) => {
      if (event.data?.size) voiceChunks.push(event.data);
    });
    voiceRecorder.addEventListener("stop", () => {
      const recorderMimeType = voiceRecorder?.mimeType || mimeType || "audio/webm";
      const recordedChunks = voiceChunks;
      stopVoiceStream();
      const extension = recorderMimeType.includes("mp4")
        ? "m4a"
        : recorderMimeType.includes("ogg")
          ? "ogg"
          : "webm";
      const blob = new Blob(recordedChunks, { type: recorderMimeType });
      voiceRecorder = null;
      voiceChunks = [];
      if (!blob.size) {
        setVoiceRecordingState("idle");
        submit.disabled = false;
        setCommandDisabled(false);
        setSessionStatus("Voice note is empty.");
        return;
      }
      setVoiceRecordingState("transcribing");
      transcribeVoiceBlob(blob, `recorded-idea.${extension}`);
    });
    voiceRecorder.start();
    setVoiceRecordingState("recording");
    setSessionStatus("Listening. Press Stop when your idea is ready.");
  } catch (error) {
    stopVoiceStream();
    voiceRecorder = null;
    voiceChunks = [];
    setVoiceRecordingState("idle");
    submit.disabled = false;
    setCommandDisabled(false);
    setSessionStatus(`Voice recording could not start: ${error.message}`);
  }
}

function stopVoiceRecording() {
  if (!voiceRecorder || voiceRecorder.state !== "recording") return;
  setVoiceRecordingState("stopping");
  setSessionStatus("Stopping recording.");
  try {
    voiceRecorder.stop();
  } catch (error) {
    stopVoiceStream();
    voiceRecorder = null;
    voiceChunks = [];
    setVoiceRecordingState("idle");
    submit.disabled = false;
    setCommandDisabled(false);
    setSessionStatus(`Voice recording could not stop: ${error.message}`);
  }
}

function recordingMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function stopVoiceStream() {
  if (!voiceStream) return;
  voiceStream.getTracks().forEach((track) => track.stop());
  voiceStream = null;
}

async function transcribeVoiceBlob(blob, filename) {
  if (sessionControlsLocked || voiceBusy) return false;
  if (voiceRecordingState !== "idle" && voiceRecordingState !== "transcribing") return false;
  if (!blob?.size) {
    setSessionStatus("Voice note is empty.");
    return false;
  }
  const revision = bumpSessionRevision();
  voiceBusy = true;
  setVoiceRecordingState("transcribing");
  submit.disabled = true;
  input.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  setSessionStatus("Transcribing voice note.");
  try {
    const formData = new FormData();
    formData.append("audio", blob, filename || "voice-note.audio");
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(`voice note failed with ${response.status}`);
    const data = await response.json();
    const transcript = String(data.transcript || "").trim();
    if (!transcript) throw new Error("empty transcript");
    if (!isCurrentSessionRevision(revision)) return false;
    input.value = mergeDraftWithTranscript(input.value, transcript);
    session.ui_status = "Voice note transcribed. Edit the draft or press Ink.";
    corrections.textContent = session.ui_status;
    saveSession();
    return true;
  } catch (error) {
    if (isCurrentSessionRevision(revision)) setSessionStatus(`Voice note could not be transcribed: ${error.message}`);
    return false;
  } finally {
    voiceBusy = false;
    setVoiceRecordingState("idle");
    if (isCurrentSessionRevision(revision)) {
      submit.disabled = false;
      input.disabled = false;
      setSessionControlsDisabled(false);
      setCommandDisabled(false);
      input.focus();
    }
  }
}

function mergeDraftWithTranscript(draft, transcript) {
  const current = String(draft || "").trim();
  return current ? `${current}\n${transcript}` : transcript;
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  if (!response.ok) throw new Error(`project index failed with ${response.status}`);
  const data = await response.json();
  bootstrapData = data;
  const rawProfiles = Array.isArray(data.goal_profiles) ? data.goal_profiles : [];
  const rawOptions = Array.isArray(data.goal_options) ? data.goal_options : [];
  goalProfiles = normalizeGoalProfiles(rawProfiles, rawOptions);
  goalOptions = goalProfiles.map((goal) => goal.id);
  goalProfileById = new Map(goalProfiles.map((goal) => [goal.id, goal]));
  profileFields = data.profile_fields || [];
  session = normalizeSession(readSavedSession(), defaultSession(data));
  renderProvenance(data);
  renderGoals(session.goals);
  renderProfile(session.profile);
  renderRestoredSession(data);
  renderWhitespace(data.whitespace || []);
  setVoiceRecordingState("idle");
}

function handleBootstrapError(error) {
  bootstrapData = null;
  currentArtifact = null;
  session = {};
  submit.disabled = true;
  input.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  ink.textContent = `The project index could not be opened: ${error.message}`;
  ink.classList.remove("thinking", "gold");
  ink.classList.add("bleed");
  corrections.textContent = "Reload the page to try again.";
  provenanceEl.textContent = "index unavailable";
  renderScore(null);
  setVerdictDisplay("INDEX CLOSED", 0, null);
  renderWoodMap(null);
  renderGoals([]);
  renderProfile({});
  renderIdeas([]);
  renderProjects([]);
  renderWhitespace([]);
  renderPlan([]);
}

function defaultSession(data = bootstrapData) {
  return {
    profile: {},
    goals: data?.default_goals || goalOptions.slice(0, 3),
  };
}

function setActiveTab(tab) {
  if (!spreadEl) return;
  const next = ["page", "proof", "almanac"].includes(tab) ? tab : "page";
  spreadEl.dataset.tab = next;
  document.querySelectorAll(".mobile-nav [data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === next);
  });
}

function setVerdictDisplay(verdict = "READY", overall = 0, score = null) {
  const text = String(verdict || "READY");
  const isEcho = text.startsWith("ECHO");
  const isUnwritten = text.startsWith("UNWRITTEN");
  const numericOverall = Number(overall || score?.overall || 0);

  verdictEl.textContent = text;
  overallEl.textContent = numericOverall.toFixed(1);
  sealEl.classList.toggle("echo", isEcho);
  sealEl.classList.toggle("unwritten", isUnwritten);

  sealVerdictEl.textContent = text;
  sealVerdictEl.classList.toggle("echo", isEcho);
  sealVerdictEl.classList.toggle("unwritten", isUnwritten);
  sealVerdictEl.classList.toggle("ready", !isEcho && !isUnwritten);

  verdictStampEl.classList.toggle("verdict-echo", isEcho);
  verdictStampEl.classList.toggle("verdict-unwritten", isUnwritten);
  verdictStampEl.classList.toggle("verdict-ready", !isEcho && !isUnwritten);

  if (!score) {
    sealCopyEl.textContent = text === "INDEX CLOSED" ? "The project map did not load." : "No idea has been scored yet.";
  } else if (isEcho) {
    sealCopyEl.textContent = "Nearby projects already cover parts of this idea.";
  } else {
    sealCopyEl.textContent = "This idea sits in a quieter part of the current map.";
  }
}

function bumpSessionRevision() {
  sessionRevision += 1;
  return sessionRevision;
}

function isCurrentSessionRevision(revision) {
  return revision === sessionRevision;
}

function restoreExportButtonLabels() {
  setActionButtonLabel(exportNotesButton, "Notes");
  setActionButtonLabel(exportChapterButton, "Chapter");
  setActionButtonLabel(exportButton, PNG_EXPORT_LABEL);
}

function actionButtonLabel(button) {
  return button?.dataset.actionLabel || button?.textContent.trim() || "";
}

function setActionButtonLabel(button, label) {
  if (!button) return;
  button.dataset.actionLabel = label;
  const textNode = Array.from(button.childNodes).find(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
  );
  if (textNode) {
    textNode.textContent = ` ${label}`;
  } else {
    button.append(document.createTextNode(` ${label}`));
  }
}

function setSessionControlsDisabled(disabled) {
  sessionControlsLocked = disabled;
  goalsEl.querySelectorAll("input[data-goal]").forEach((target) => {
    target.disabled = disabled;
  });
  profileEl.querySelectorAll("input[data-profile-field]").forEach((field) => {
    field.disabled = disabled;
  });
  ideasEl.querySelectorAll("button[data-idea-id]").forEach((idea) => {
    idea.disabled = disabled;
  });
  whitespaceEl.querySelectorAll("button[data-gap-prompt]").forEach((gap) => {
    gap.disabled = disabled;
  });
  setVoiceControlsDisabled(disabled);
}

function setVoiceControlsDisabled(disabled) {
  const recording = voiceRecordingState === "recording" && voiceRecorder?.state === "recording";
  const lockedForState = ["starting", "stopping", "transcribing"].includes(voiceRecordingState);
  recordVoiceButton.disabled = !bootstrapData || voiceBusy || lockedForState || (disabled && !recording);
  uploadVoiceButton.disabled = !bootstrapData || voiceBusy || disabled || voiceRecordingState !== "idle";
}

function setVoiceRecordingState(state) {
  voiceRecordingState = state;
  recordVoiceButton.dataset.voiceState = state;
  recordVoiceButton.classList.toggle("recording", state === "recording");
  recordVoiceButton.setAttribute("aria-pressed", state === "recording" ? "true" : "false");
  const labels = {
    idle: "Speak",
    starting: "Starting...",
    recording: "Stop",
    stopping: "Stopping...",
    transcribing: "Hearing...",
  };
  setActionButtonLabel(recordVoiceButton, labels[state] || "Speak");
  setVoiceControlsDisabled(sessionControlsLocked);
}

function resetSession() {
  if (!bootstrapData) return;
  bumpSessionRevision();
  clearTurnWatchdog();
  clearSavedSession();
  session = defaultSession(bootstrapData);
  currentArtifact = null;
  submit.disabled = false;
  input.disabled = false;
  setSessionControlsDisabled(false);
  input.value = "";
  ink.textContent = "The book is open. Describe an idea to start a new page.";
  ink.classList.remove("thinking", "bleed", "gold");
  corrections.textContent = "Session reset.";
  renderGoals(session.goals);
  renderProfile(session.profile);
  renderScore(null);
  setVerdictDisplay("READY", 0, null);
  renderWoodMap(null);
  renderIdeas([]);
  renderPlan([]);
  renderProjects([], "Score an idea to see nearby echoes.");
  renderWhitespace(bootstrapData.whitespace || []);
  restoreExportButtonLabels();
  setCommandDisabled(false);
  saveSession();
  input.focus();
}

async function loadDemoSession() {
  bumpSessionRevision();
  setActiveTab("page");
  submit.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  ink.classList.remove("bleed", "gold");
  ink.classList.add("thinking");
  ink.textContent = "Loading an example idea board.";
  corrections.textContent = "";
  try {
    const response = await fetch("/api/demo-session");
    if (!response.ok) throw new Error(`example session failed with ${response.status}`);
    applyDemoSession(await response.json());
  } catch (error) {
    ink.textContent = `The example session could not be loaded: ${error.message}`;
    ink.classList.remove("thinking");
    ink.classList.add("bleed");
  } finally {
    submit.disabled = false;
    setSessionControlsDisabled(false);
    setCommandDisabled(false);
    input.focus();
  }
}

function applyDemoSession(data) {
  session = data.session || {};
  session.profile = session.profile || {};
  session.goals = Array.isArray(session.goals) ? session.goals : [];
  session.last_response = data.response || session.last_response || "";
  session.ui_status = "Example idea board loaded with a plan and share page.";
  currentArtifact = data.artifact || session.last_artifact || null;
  ink.textContent = data.response || "Example session loaded.";
  ink.classList.remove("thinking");
  if (data.score) {
    setVerdictDisplay(data.score.verdict, data.score.overall, data.score);
    renderScore(data.score);
    ink.classList.toggle("bleed", data.score.verdict.startsWith("ECHO"));
    ink.classList.toggle("gold", data.score.verdict.startsWith("UNWRITTEN"));
  }
  renderGoals(session.goals);
  renderProfile(session.profile);
  renderIdeas(session.ideas || []);
  renderPlan(data.plan || session.last_plan || []);
  renderWhitespace(data.whitespace || []);
  if (currentArtifact?.wood_map) renderWoodMap(currentArtifact.wood_map);
  if (data.score?.echoes?.length) {
    renderCitations(data.score.echoes);
  } else {
    renderProjects(data.projects || []);
  }
  setCommandDisabled(false);
  corrections.textContent = session.ui_status;
  saveSession();
}

function renderProvenance(data) {
  const projectCount = Number(data.project_count || data.top_projects?.length || 0);
  const countLabel = projectCount
    ? `${projectCount} project page${projectCount === 1 ? "" : "s"} mapped`
    : "Current project map loaded";
  const updated = shortDate(data.snapshot_generated_at || data.index_generated_at);
  provenanceEl.textContent = `${countLabel} · updated ${updated}`;
}

function renderRestoredSession(data) {
  const restoredProjectReference = isProjectReferenceTool(session.last_tool_resolution?.call?.name || "");
  if (restoredProjectReference) {
    currentArtifact = null;
    renderProjectReferenceState();
    renderProjects(session.last_projects || [], "No project page matched this request.");
    renderIdeas(session.ideas || []);
    delete session.last_plan;
    renderPlan([]);
    setCommandDisabled(false);
    restoreSessionCopy();
    return;
  }
  const idea = currentIdea();
  const storedArtifact = session.last_artifact || null;
  currentArtifact = !idea || storedArtifact?.title === idea.title ? storedArtifact : null;
  const score = currentArtifact?.seal || idea?.score || null;
  if (score) {
    renderScore(score);
    const verdict = currentArtifact?.verdict || score.verdict || "UNWRITTEN";
    setVerdictDisplay(verdict, currentArtifact?.overall || score.overall || 0, score);
    ink.classList.toggle("bleed", verdict.startsWith("ECHO"));
    ink.classList.toggle("gold", verdict.startsWith("UNWRITTEN"));
    renderWoodMap(currentArtifact?.wood_map || null);
    if (score.echoes?.length) {
      renderCitations(score.echoes);
    } else {
      renderProjects([]);
    }
  } else {
    renderScore(null);
    setVerdictDisplay("READY", 0, null);
    renderWoodMap(null);
    renderProjects([], "Score an idea to see nearby echoes.");
  }
  renderIdeas(session.ideas || []);
  renderPlan(session.last_plan || []);
  setCommandDisabled(false);
  restoreSessionCopy();
}

function readSavedSession() {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeSession(savedSession, defaultSession) {
  const normalized = { ...defaultSession };
  if (!savedSession) return normalized;
  normalized.profile = savedSession.profile && typeof savedSession.profile === "object" ? savedSession.profile : {};
  const savedGoals = Array.isArray(savedSession.goals) ? savedSession.goals : defaultSession.goals;
  normalized.goals = goalOptions.filter((option) => savedGoals.includes(option));
  if (!normalized.goals.length && defaultSession.goals?.length) normalized.goals = [...defaultSession.goals];
  if (Array.isArray(savedSession.ideas)) normalized.ideas = savedSession.ideas;
  if (Array.isArray(savedSession.trace)) normalized.trace = savedSession.trace;
  if (Array.isArray(savedSession.last_plan)) normalized.last_plan = savedSession.last_plan;
  if (savedSession.current_idea_id) normalized.current_idea_id = savedSession.current_idea_id;
  if (savedSession.current_whitespace) normalized.current_whitespace = savedSession.current_whitespace;
  if (savedSession.last_tool_resolution) normalized.last_tool_resolution = savedSession.last_tool_resolution;
  if (savedSession.last_artifact) normalized.last_artifact = savedSession.last_artifact;
  if (Array.isArray(savedSession.last_projects)) normalized.last_projects = savedSession.last_projects;
  if (typeof savedSession.last_response === "string") normalized.last_response = savedSession.last_response;
  if (typeof savedSession.ui_status === "string") normalized.ui_status = savedSession.ui_status;
  return normalized;
}

function restoreSessionCopy() {
  const response = typeof session.last_response === "string" ? session.last_response.trim() : "";
  if (response) ink.textContent = response;
  const status = typeof session.ui_status === "string" ? session.ui_status.trim() : "";
  if (status) corrections.textContent = status;
}

function normalizeGoalProfiles(profiles, options) {
  const byId = new Map(
    profiles
      .filter((profile) => profile && typeof profile.id === "string")
      .map((profile) => [
        profile.id,
        {
          id: profile.id,
          label: String(profile.label || profile.id),
          description: String(profile.description || ""),
        },
      ]),
  );
  return options.map((id) => byId.get(id) || { id, label: id, description: "" });
}

function saveSession() {
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Storage may be disabled in some embeds; the app still works in-memory.
  }
}

function clearSavedSession() {
  try {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Nothing else to clear when storage is unavailable.
  }
}

function renderGoals(selectedGoals) {
  const selected = new Set(selectedGoals || []);
  if (goalCountEl) goalCountEl.textContent = selected.size;
  goalsEl.innerHTML = "";
  if (!goalOptions.length) {
    goalsEl.innerHTML = `<div class="empty">No goals loaded.</div>`;
    return;
  }
  for (const option of goalOptions) {
    const profile = goalProfileById.get(option) || { label: option, description: "" };
    const label = document.createElement("label");
    label.className = `goal-toggle goal ${selected.has(option) ? "on" : ""}`;
    label.innerHTML = `
      <input
        type="checkbox"
        data-goal="${escapeAttribute(option)}"
        aria-label="${escapeAttribute(profile.label)}"
        ${sessionControlsLocked ? "disabled" : ""}
        ${selected.has(option) ? "checked" : ""}
      />
      <span class="check" aria-hidden="true">
        <svg class="icon"><use href="#icon-check"></use></svg>
      </span>
      <span class="goal-copy">
        <strong>${escapeHtml(profile.label)}</strong>
        ${profile.description ? `<small>${escapeHtml(profile.description)}</small>` : ""}
      </span>
    `;
    goalsEl.append(label);
  }
}

function renderProfile(profile) {
  profileEl.innerHTML = "";
  if (!profileFields.length) {
    profileEl.innerHTML = `<div class="empty">No profile fields.</div>`;
    return;
  }
  for (const field of profileFields) {
    const row = document.createElement("label");
    row.className = "profile-field";
    row.innerHTML = `
      <span>${escapeHtml(fieldLabel(field))}</span>
      <input
        data-profile-field="${escapeAttribute(field)}"
        value="${escapeAttribute(profile?.[field] || "")}"
        placeholder="${escapeAttribute(fieldPlaceholder(field))}"
        autocomplete="off"
        ${sessionControlsLocked ? "disabled" : ""}
      />
    `;
    profileEl.append(row);
  }
}

function handleEvent(event) {
  if (event.type === "start") {
    if (event.corrections?.length) {
      corrections.textContent = event.corrections
        .map((item) => `heard: ${item.original} -> ${item.canonical}`)
        .join("   ");
    }
    return;
  }

  if (event.type === "stage") {
    setTurnStage(event.stage, event.label);
    return;
  }

  if (event.type === "model_progress") {
    renderModelProgress(event.tokens, event.max_tokens);
    return;
  }

  if (event.type === "tool_event") {
    addToolChip(event);
    return;
  }

  if (event.type === "fallback") {
    renderComputeFallback(event);
    return;
  }

  if (event.type === "token") {
    markFirstTokenSeen();
    ink.textContent += event.text;
    return;
  }

  if (event.type === "done") {
    setTurnBar(100);
    if (turnEtaEl) turnEtaEl.textContent = "";
    hideTurnProgress();
    if (!sawTurnToken) {
      clearTurnWatchdog();
      ink.textContent = event.response || ink.textContent;
      ink.classList.remove("thinking");
    }
    session = event.state || {};
    session.profile = session.profile || {};
    session.goals = Array.isArray(session.goals) ? session.goals : [];
    session.last_response = event.response || session.last_response || "";
    delete session.ui_status;
    const toolName = session.last_tool_resolution?.call?.name || "";
    const projectReferenceOnly = !event.score && !event.artifact?.title && isProjectReferenceTool(toolName);
    if (projectReferenceOnly) {
      session.last_projects = Array.isArray(event.projects) ? event.projects : [];
    } else {
      delete session.last_projects;
    }
    if (event.score?.echoes?.length) {
      renderCitations(event.score.echoes);
    } else if (event.projects?.length) {
      renderProjects(event.projects);
    } else if (projectReferenceOnly) {
      renderProjects([], "No project page matched this request.");
    }
    if (event.whitespace?.length) renderWhitespace(event.whitespace);
    renderGoals(session.goals);
    renderProfile(session.profile);
    renderIdeas(session.ideas || []);
    renderPlan(event.plan || []);
    if (event.score) {
      setVerdictDisplay(event.score.verdict, event.score.overall, event.score);
      renderScore(event.score);
      ink.classList.toggle("bleed", event.score.verdict.startsWith("ECHO"));
      ink.classList.toggle("gold", event.score.verdict.startsWith("UNWRITTEN"));
    } else if (projectReferenceOnly) {
      renderProjectReferenceState();
    } else if (!event.projects?.length) {
      const idea = currentIdea();
      renderSelectedIdeaSeal(idea);
      renderSelectedIdeaArtifact(idea || {});
    }
    if (event.artifact?.title) {
      currentArtifact = event.artifact;
      renderWoodMap(event.artifact.wood_map || null);
    }
    setCommandDisabled(false);
    saveSession();
  }
}

function renderProjectReferenceState() {
  currentArtifact = null;
  renderScore(null);
  setVerdictDisplay("READY", 0, null);
  sealCopyEl.textContent = "Project pages are shown below. Write or select an idea to score it.";
  ink.classList.remove("bleed", "gold");
  renderWoodMap(null);
}

function isProjectReferenceTool(toolName) {
  return ["list_projects", "get_project"].includes(toolName);
}

function renderIdeas(ideas) {
  if (ideaCountEl) ideaCountEl.textContent = ideas.length;
  ideasEl.innerHTML = "";
  if (!ideas.length) {
    ideasEl.innerHTML = `<div class="empty">Your idea board is empty. Write an idea or choose an under-explored direction.</div>`;
    return;
  }
  for (const idea of visibleIdeas(ideas)) {
    const score = idea.score?.overall ? Number(idea.score.overall).toFixed(1) : "0.0";
    const goals = (idea.goals || []).slice(0, 3).map(goalDisplayName).join(" · ");
    const selected = idea.id === session.current_idea_id;
    const verdict = idea.score?.verdict || "DRAFT";
    const isEcho = String(verdict).startsWith("ECHO");
    const item = document.createElement("button");
    item.type = "button";
    item.className = `idea idea-card ${selected ? "current" : ""} ${isEcho ? "bleed" : ""}`;
    item.disabled = sessionControlsLocked;
    item.dataset.ideaId = idea.id || "";
    item.setAttribute("aria-pressed", selected ? "true" : "false");
    item.setAttribute("aria-label", ideaCardAriaLabel(idea, score, verdict));
    item.innerHTML = `
      <div class="ihead">
        <strong>${escapeHtml(idea.title)}</strong>
        <span class="iscore">${score}</span>
      </div>
      <p>${escapeHtml((idea.pitch || "").slice(0, 120))}</p>
      <span class="iverdict ${isEcho ? "echo" : "unwritten"}">${escapeHtml(verdict)}</span>
      ${goals ? `<small>${escapeHtml(goals)}</small>` : ""}
    `;
    ideasEl.append(item);
  }
}

function visibleIdeas(ideas) {
  const currentId = session.current_idea_id;
  const current = currentId ? ideas.find((idea) => idea.id === currentId) : null;
  const remaining = ideas.filter((idea) => idea.id !== currentId).slice(-3).reverse();
  return current ? [current, ...remaining] : ideas.slice(-4).reverse();
}

function ideaCardAriaLabel(idea, score, verdict) {
  const title = String(idea?.title || "Untitled idea").trim() || "Untitled idea";
  const parts = [`Select idea: ${title}`];
  if (score) parts.push(`score ${score}`);
  if (verdict) parts.push(String(verdict));
  return parts.join(", ");
}

function currentIdea() {
  const ideas = Array.isArray(session.ideas) ? session.ideas : [];
  return ideas.find((idea) => idea.id === session.current_idea_id) || ideas[ideas.length - 1] || null;
}

function selectIdea(ideaId) {
  if (!ideaId || !Array.isArray(session.ideas)) return;
  const idea = session.ideas.find((item) => item.id === ideaId);
  if (!idea) return;
  const changedIdea = idea.id !== session.current_idea_id;
  bumpSessionRevision();
  session.current_idea_id = idea.id;
  if (Array.isArray(idea.goals) && idea.goals.length) {
    session.goals = goalOptions.filter((option) => idea.goals.includes(option));
  }
  renderSelectedIdeaSeal(idea);
  renderSelectedIdeaArtifact(idea);
  renderGoals(session.goals || []);
  renderIdeas(session.ideas);
  if (changedIdea) delete session.last_plan;
  renderPlan(changedIdea ? [] : session.last_plan || []);
  session.ui_status = `selected: ${idea.title}`;
  corrections.textContent = session.ui_status;
  saveSession();
}

function renderSelectedIdeaSeal(idea) {
  const score = idea?.score || null;
  if (!score) {
    renderScore(null);
    setVerdictDisplay("READY", 0, null);
    ink.classList.remove("bleed", "gold");
    renderProjects([], "Score this idea to see nearby echoes.");
    return;
  }
  setVerdictDisplay(score.verdict || "DRAFT", score.overall || 0, score);
  renderScore(score);
  ink.classList.toggle("bleed", String(score.verdict || "").startsWith("ECHO"));
  ink.classList.toggle("gold", String(score.verdict || "").startsWith("UNWRITTEN"));
  if (score.echoes?.length) {
    renderCitations(score.echoes);
  } else {
    renderProjects([]);
  }
}

function renderSelectedIdeaArtifact(idea) {
  const artifact = idea.artifact || (session.last_artifact?.title === idea.title ? session.last_artifact : null);
  if (artifact) {
    currentArtifact = artifact;
    session.last_artifact = artifact;
    renderWoodMap(currentArtifact.wood_map || null);
    setCommandDisabled(false);
    return;
  }
  currentArtifact = null;
  renderWoodMap(null);
  setCommandDisabled(false);
}

function goalDisplayName(goal) {
  return goalProfileById.get(goal)?.label || goal;
}

function renderScore(score) {
  const rows = [
    ["Original", score?.originality || 0],
    ["Delight", score?.delight || 0],
    ["AI Need", score?.ai_necessity || 0],
    ["Feasible", score?.feasibility || 0],
    ["Goal Fit", score?.goal_fit || 0],
  ];
  scoreEl.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="quad">
          <span class="ql">${label}</span>
          <span class="qbar"><span class="qfill" style="width: ${Number(value) * 10}%"></span></span>
          <span class="qv">${value}</span>
        </div>
      `,
    )
    .join("");
}

function renderWoodMap(map) {
  woodMapEl.innerHTML = "";
  if (!map?.dots?.length) {
    woodMapEl.innerHTML = `<div class="wood"><div class="empty wood-empty">Score an idea to plot it on the map.</div></div>`;
    return;
  }
  const field = document.createElement("div");
  field.className = "wood";
  for (const dot of map.dots) {
    const marker = document.createElement(dot.url ? "a" : "span");
    // Namespace the kind class (wood-idea/wood-echo/wood-inked) so it never collides with the
    // global .idea/.echo card styles. The "you" dot stays green regardless of verdict.
    marker.className = `wood-dot wood-${dot.kind || "inked"}`;
    marker.style.left = `${boundedPercent(dot.x)}%`;
    marker.style.top = `${boundedPercent(dot.y)}%`;
    const radius = Math.max(3, Math.min(10, Number(dot.radius || 4)));
    marker.style.width = `${radius * 2}px`;
    marker.style.height = `${radius * 2}px`;
    marker.title = dot.kind === "idea" ? `You: ${dot.title}` : `${dot.title}${dot.score ? ` (${dot.score})` : ""}`;
    if (dot.url) {
      marker.href = dot.url;
      marker.target = "_blank";
      marker.rel = "noreferrer";
    }
    field.append(marker);
  }
  const legend = document.createElement("div");
  legend.className = "wood-legend";
  legend.innerHTML = `
    <span><i style="background: var(--leaf)"></i> You</span>
    <span><i style="background: var(--oxblood)"></i> Echo</span>
    <span><i style="background: rgba(73, 49, 22, 0.34)"></i> Indexed</span>
  `;
  const caption = document.createElement("p");
  caption.className = "wood-cap";
  caption.textContent = map.caption || "Your idea is plotted against the current project map.";
  woodMapEl.append(field, legend, caption);
}

function renderProjects(projects, emptyMessage = "No nearby projects yet.") {
  projectsEl.innerHTML = "";
  if (!projects.length) {
    projectsEl.innerHTML = `<div class="empty">${escapeHtml(emptyMessage)}</div>`;
    return;
  }
  for (const project of projects.slice(0, 5)) {
    const item = document.createElement("a");
    item.className = "project echo-item";
    item.href = project.url;
    item.target = "_blank";
    item.rel = "noreferrer";
    const summary = project.summary ? `<p>${escapeHtml(project.summary)}</p>` : "";
    item.innerHTML = `<strong>${escapeHtml(project.title || "Untitled project")}</strong>${summary}`;
    projectsEl.append(item);
  }
}

function renderCitations(echoes) {
  projectsEl.innerHTML = "";
  if (!echoes.length) {
    projectsEl.innerHTML = `<div class="empty">No nearby project echoes yet.</div>`;
    return;
  }
  for (const echo of echoes.slice(0, 5)) {
    const project = echo.project || {};
    const item = document.createElement("a");
    item.className = "project echo-item citation";
    item.href = project.url || project.host || "#";
    item.target = "_blank";
    item.rel = "noreferrer";
    item.title = project.title || project.id || "Project citation";
    const matched = (echo.matched_terms || []).slice(0, 5).join(", ") || "no shared terms";
    const summary = project.summary ? `<p>${escapeHtml(project.summary)}</p>` : "";
    item.innerHTML = `
      <strong>Page ${escapeHtml(echo.page_number || "?")} · ${escapeHtml(project.title || "Untitled project")}</strong>
      ${summary}
      <span class="matched">${Number(echo.score || 0).toFixed(3)} · ${escapeHtml(matched)}</span>
    `;
    projectsEl.append(item);
  }
}

function renderWhitespace(items) {
  whitespaceEl.innerHTML = "";
  if (!items.length) {
    whitespaceEl.innerHTML = `<div class="empty">No starting directions are loaded yet.</div>`;
    return;
  }
  for (const item of items.slice(0, 4)) {
    const gap = document.createElement("button");
    gap.type = "button";
    gap.className = "gap gap-item";
    gap.disabled = sessionControlsLocked;
    gap.dataset.gapPrompt = `idea: ${item.label} -- ${item.pitch}`;
    gap.innerHTML = `
      <strong>${escapeHtml(item.label)}</strong>
      <p>${escapeHtml(item.pitch)}</p>
      <span class="use">Use this direction</span>
    `;
    whitespaceEl.append(gap);
  }
}

function renderPlan(steps) {
  planEl.innerHTML = "";
  if (!steps.length) {
    planEl.innerHTML = `<li class="empty">Press Plan to draft build steps for the selected idea.</li>`;
    return;
  }
  for (const step of steps) {
    const item = document.createElement("li");
    item.textContent = step;
    planEl.append(item);
  }
}

function setCommandDisabled(disabled) {
  document.querySelectorAll(".command-row button").forEach((button) => {
    const isArtifact = button.id === "export-artifact";
    const isNotes = button.id === "export-notes";
    const isChapter = button.id === "export-chapter";
    const locked = disabled || sessionControlsLocked;
    button.disabled =
      locked ||
      (isArtifact && !currentArtifact) ||
      (isNotes && !session.trace?.length) ||
      (isChapter && !session.ideas?.length);
  });
}

function startTurnWatchdog() {
  clearTurnWatchdog();
  sawTurnToken = false;
  ink.textContent = "Checking the current project map.";
  ink.classList.add("thinking");
  turnWatchdog = window.setTimeout(() => {
    if (sawTurnToken) return;
    ink.textContent = "Still comparing against nearby projects.";
  }, 2200);
}

function markFirstTokenSeen() {
  if (sawTurnToken) return;
  sawTurnToken = true;
  clearTurnWatchdog();
  ink.textContent = "";
  ink.classList.remove("thinking");
}

function clearTurnWatchdog() {
  if (turnWatchdog) {
    window.clearTimeout(turnWatchdog);
    turnWatchdog = null;
  }
}

function setupMenus() {
  document.querySelectorAll(".menu").forEach((menu) => {
    const trigger = menu.querySelector(".menu-trigger");
    const list = menu.querySelector(".menu-list");
    if (!trigger || !list) return;
    trigger.addEventListener("click", (event) => {
      event.stopPropagation();
      const willOpen = list.hidden;
      closeAllMenus();
      if (willOpen) {
        list.hidden = false;
        trigger.setAttribute("aria-expanded", "true");
      }
    });
    list.addEventListener("click", () => closeAllMenus()); // selecting an item closes the menu
  });
  document.addEventListener("click", () => closeAllMenus());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAllMenus();
  });
}

function closeAllMenus() {
  document.querySelectorAll(".menu-list").forEach((list) => {
    list.hidden = true;
  });
  document.querySelectorAll(".menu-trigger").forEach((trigger) => {
    trigger.setAttribute("aria-expanded", "false");
  });
}

// Coarse overall completion per stage, so the bar always advances even when token-level
// progress is unknown (e.g. the rules backend, or the fast tool/writing stages).
const STAGE_PROGRESS = { planning: 8, running_tool: 85, writing: 95 };

function resetTurnProgress() {
  if (!turnProgressEl) return;
  // Stay hidden on submit. Only reveal once the turn is genuinely executing — either real
  // token decoding starts, or it has been running long enough to be worth a progress bar.
  // A fast turn finishes before the timer fires, so the bar never flashes.
  turnProgressEl.hidden = true;
  decodeStartedAt = 0;
  if (toolChipsEl) toolChipsEl.innerHTML = "";
  if (turnTokensEl) turnTokensEl.textContent = "";
  if (turnEtaEl) turnEtaEl.textContent = "";
  setTurnBar(4);
  setTurnStageContent("planning", "Thinking");
  clearTurnProgressTimer();
  turnProgressTimer = window.setTimeout(revealTurnProgress, 450);
}

function revealTurnProgress() {
  if (turnProgressEl) turnProgressEl.hidden = false;
}

function clearTurnProgressTimer() {
  if (turnProgressTimer) {
    window.clearTimeout(turnProgressTimer);
    turnProgressTimer = null;
  }
}

function hideTurnProgress() {
  clearTurnProgressTimer();
  if (turnProgressEl) turnProgressEl.hidden = true;
}

function setTurnBar(percent) {
  if (!turnBarFillEl) return;
  const clamped = Math.max(0, Math.min(100, percent));
  turnBarFillEl.style.width = `${clamped}%`;
}

function setTurnStageContent(stage, label) {
  if (turnStageIconEl) turnStageIconEl.textContent = STAGE_ICONS[stage] || "🪶";
  if (turnStageTextEl) turnStageTextEl.textContent = label || "Thinking";
  if (stage in STAGE_PROGRESS) setTurnBar(STAGE_PROGRESS[stage]);
  if (stage && stage !== "planning" && turnEtaEl) turnEtaEl.textContent = "";
}

function setTurnStage(stage, label) {
  clearTurnWatchdog();
  setTurnStageContent(stage, label);
}

function renderModelProgress(tokens, maxTokens) {
  clearTurnWatchdog();
  revealTurnProgress(); // real token decoding is unambiguous execution — show it now
  const count = Number(tokens) || 0;
  if (turnTokensEl) turnTokensEl.textContent = count ? `· decoded ${count} tokens` : "";
  if (!count) return;
  if (!decodeStartedAt) decodeStartedAt = performance.now();

  const cap = Number(maxTokens) || 0;
  // Map token decode into the 8%–80% band of the overall bar.
  if (cap > 0) setTurnBar(8 + Math.min(1, count / cap) * 72);

  // Estimate remaining time from the live decode rate toward the token cap (an upper bound).
  const elapsed = (performance.now() - decodeStartedAt) / 1000;
  if (turnEtaEl && cap > 0 && elapsed > 0.3) {
    const rate = count / elapsed;
    const remaining = Math.max(0, cap - count) / Math.max(rate, 0.1);
    turnEtaEl.textContent = remaining >= 1 ? `~${Math.ceil(remaining)}s left` : "almost done";
  }
}

function addToolChip(event) {
  if (!toolChipsEl) return;
  const name = event.name || event.tool || "tool";
  const chip = document.createElement("span");
  chip.className = "tool-chip";
  if (event.summary) chip.title = event.summary;
  chip.innerHTML = `<span class="tc-name"></span><span class="tc-check">✓</span>`;
  chip.querySelector(".tc-name").textContent = name;
  toolChipsEl.append(chip);
}

function renderComputeFallback(event) {
  // Acceleration is automatic; a fallback is informational only (no control to flip).
  const reason = event.reason || "Running on CPU (slower).";
  if (turnStageTextEl) turnStageTextEl.textContent = reason;
  if (corrections) corrections.textContent = reason;
}

function syncCurrentIdeaGoals() {
  const currentId = session.current_idea_id;
  if (!currentId || !Array.isArray(session.ideas)) return;
  const idea = session.ideas.find((item) => item.id === currentId);
  if (idea) idea.goals = [...(session.goals || [])];
}

function invalidateCurrentSeal(message) {
  const idea = currentIdea();
  if (idea?.score) idea.score = null;
  clearCurrentArtifactFor(idea);
  invalidateCurrentPlan("");
  renderScore(null);
  setVerdictDisplay("READY", 0, null);
  renderWoodMap(null);
  renderProjects([]);
  setCommandDisabled(false);
  if (message) setSessionStatus(message);
}

function invalidateCurrentPlan(message) {
  if (Array.isArray(session.last_plan)) delete session.last_plan;
  renderPlan([]);
  if (message) setSessionStatus(message);
}

function clearCurrentArtifactFor(idea) {
  if (!idea || currentArtifact?.title === idea.title) currentArtifact = null;
  if (idea?.artifact) delete idea.artifact;
  if (!idea || session.last_artifact?.title === idea.title) delete session.last_artifact;
}

function setSessionStatus(message) {
  session.ui_status = message;
  corrections.textContent = message;
}

async function exportNotes() {
  await exportMarkdown({
    endpoint: "/api/field-notes",
    filename: FIELD_NOTES_FILENAME,
    button: exportNotesButton,
    busyLabel: "Notes...",
    pendingLabel: "Writing notes.",
    successLabel: "Notes saved",
  });
}

async function exportChapter() {
  await exportMarkdown({
    endpoint: "/api/chapter",
    filename: CHAPTER_FILENAME,
    button: exportChapterButton,
    busyLabel: "Chapter...",
    pendingLabel: "Writing chapter.",
    successLabel: "Chapter saved",
  });
}

async function exportMarkdown({ endpoint, filename, button, busyLabel, pendingLabel, successLabel }) {
  if (!button || button.disabled) return;
  const revision = sessionRevision;
  const idleLabel = actionButtonLabel(button);
  button.disabled = true;
  setActionButtonLabel(button, busyLabel);
  session.ui_status = pendingLabel;
  corrections.textContent = session.ui_status;
  saveSession();
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_json: JSON.stringify(session),
      }),
    });
    if (!response.ok) throw new Error(`export failed with ${response.status}`);
    const text = await response.text();
    if (!text.trim()) throw new Error("empty export");
    if (!isCurrentSessionRevision(revision)) return;
    downloadText(filename, text, "text/markdown;charset=utf-8");
    session.ui_status = `${successLabel}: ${filename}`;
    corrections.textContent = session.ui_status;
  } catch (error) {
    if (!isCurrentSessionRevision(revision)) return;
    session.ui_status = `Export failed: ${error.message}`;
    corrections.textContent = session.ui_status;
  } finally {
    setActionButtonLabel(button, idleLabel);
    if (isCurrentSessionRevision(revision)) saveSession();
    setCommandDisabled(false);
  }
}

async function exportArtifact(artifact) {
  const idleLabel = actionButtonLabel(exportButton);
  const revision = sessionRevision;
  exportButton.disabled = true;
  setActionButtonLabel(exportButton, "PNG...");
  session.ui_status = "Drawing PNG.";
  corrections.textContent = session.ui_status;
  saveSession();
  try {
    const response = await fetch("/api/artifact.png", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(artifact),
    });
    if (!response.ok) throw new Error(`PNG rendering failed with ${response.status}`);
    const blob = await response.blob();
    if (!blob.size || !blob.type.includes("png")) throw new Error("PNG rendering failed");
    if (!isCurrentSessionRevision(revision)) return;
    const filename = filenameFromContentDisposition(response.headers.get("content-disposition")) || "unwritten-page.png";
    downloadBlob(filename, blob);
    session.ui_status = `PNG saved: ${filename}`;
    corrections.textContent = session.ui_status;
  } catch (error) {
    if (!isCurrentSessionRevision(revision)) return;
    session.ui_status = `Export failed: ${error.message}`;
    corrections.textContent = session.ui_status;
  } finally {
    if (isCurrentSessionRevision(revision)) saveSession();
    setActionButtonLabel(exportButton, idleLabel || PNG_EXPORT_LABEL);
    setCommandDisabled(false);
  }
}

function downloadBlob(filename, blob) {
  const link = document.createElement("a");
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

function downloadText(filename, text, type = "application/jsonl;charset=utf-8") {
  const blob = new Blob([text], { type });
  const link = document.createElement("a");
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

function filenameFromContentDisposition(value) {
  const match = String(value || "").match(/filename="([^"]+)"/i);
  return match ? match[1] : "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

function fieldLabel(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/^\w/, (char) => char.toUpperCase());
}

function fieldPlaceholder(value) {
  const placeholders = {
    skills: "frontend, notebooks, prompt design",
    time: "one evening, weekend, 3 hours",
    preferences: "visual, playful, practical",
    constraints: "CPU-only, no paid APIs, solo build",
  };
  return placeholders[value] || "";
}

function boundedPercent(value) {
  return Math.max(4, Math.min(96, Number(value || 50)));
}

function shortDate(value) {
  if (!value) return "unknown";
  return String(value).replace("T", " ").replace(/\+00:00$/, "Z").slice(0, 16);
}
