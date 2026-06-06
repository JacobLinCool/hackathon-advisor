import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const form = document.querySelector("#turn-form");
const input = document.querySelector("#message");
const submit = document.querySelector("#submit");
const ink = document.querySelector("#ink");
const corrections = document.querySelector("#corrections");
const projectsEl = document.querySelector("#projects");
const whitespaceEl = document.querySelector("#whitespace");
const ideasEl = document.querySelector("#ideas");
const targetsEl = document.querySelector("#targets");
const profileEl = document.querySelector("#profile");
const woodMapEl = document.querySelector("#wood-map");
const scoreEl = document.querySelector("#score");
const planEl = document.querySelector("#plan");
const provenanceEl = document.querySelector("#provenance");
const verdictEl = document.querySelector("#verdict");
const overallEl = document.querySelector("#overall");
const demoButton = document.querySelector("#load-demo");
const exportButton = document.querySelector("#export-artifact");
const exportNotesButton = document.querySelector("#export-notes");
const exportChapterButton = document.querySelector("#export-chapter");
const resetButton = document.querySelector("#reset-session");

const SESSION_STORAGE_KEY = "hackathon-advisor-session-v1";
const FIELD_NOTES_FILENAME = "hackathon-advisor-field-notes.md";
const CHAPTER_FILENAME = "hackathon-advisor-chapter.md";
const PNG_EXPORT_LABEL = "PNG";

let session = {};
let clientPromise = Client.connect(window.location.origin);
let currentArtifact = null;
let targetOptions = [];
let targetProfiles = [];
let targetProfileById = new Map();
let profileFields = [];
let turnWatchdog = null;
let sawTurnToken = false;
let bootstrapData = null;
let sessionRevision = 0;
let sessionControlsLocked = false;

bootstrap().catch(handleBootstrapError);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  await runTurn(message);
});

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", async () => {
    await runTurn(button.dataset.command);
  });
});

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

targetsEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.dataset.target) return;
  const checked = new Set(
    Array.from(targetsEl.querySelectorAll("input[data-target]:checked")).map((input) => input.dataset.target),
  );
  session.targets = targetOptions.filter((option) => checked.has(option));
  syncCurrentIdeaTargets();
  saveSession();
  renderIdeas(session.ideas || []);
});

profileEl.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.dataset.profileField) return;
  const profile = { ...(session.profile || {}) };
  const value = target.value.trim();
  if (value) {
    profile[target.dataset.profileField] = value;
  } else {
    delete profile[target.dataset.profileField];
  }
  session.profile = profile;
  saveSession();
});

ideasEl.addEventListener("click", (event) => {
  const card = event.target.closest("[data-idea-id]");
  if (!(card instanceof HTMLElement) || !ideasEl.contains(card)) return;
  selectIdea(card.dataset.ideaId || "");
});

async function runTurn(message) {
  bumpSessionRevision();
  input.value = "";
  submit.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  ink.classList.remove("bleed", "gold");
  corrections.textContent = "";
  planEl.innerHTML = "";
  delete session.ui_status;
  startTurnWatchdog();

  try {
    const client = await clientPromise;
    const submission = client.submit("/agent_turn", {
      message,
      session_json: JSON.stringify(session),
    });

    for await (const event of submission) {
      if (event.type !== "data") continue;
      const payloads = Array.isArray(event.data) ? event.data : [event.data];
      for (const raw of payloads) {
        handleEvent(JSON.parse(raw));
      }
    }
  } catch (error) {
    clearTurnWatchdog();
    ink.textContent = `The page tore before it could answer: ${error.message}`;
    ink.classList.remove("thinking");
    ink.classList.add("bleed");
  } finally {
    clearTurnWatchdog();
    submit.disabled = false;
    setCommandDisabled(false);
    setSessionControlsDisabled(false);
    input.focus();
  }
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  if (!response.ok) throw new Error(`project index failed with ${response.status}`);
  const data = await response.json();
  bootstrapData = data;
  const rawProfiles = Array.isArray(data.target_profiles) ? data.target_profiles : [];
  const rawOptions = Array.isArray(data.target_options) ? data.target_options : [];
  targetProfiles = normalizeTargetProfiles(rawProfiles, rawOptions);
  targetOptions = targetProfiles.map((target) => target.id);
  targetProfileById = new Map(targetProfiles.map((target) => [target.id, target]));
  profileFields = data.profile_fields || [];
  session = normalizeSession(readSavedSession(), defaultSession(data));
  renderProvenance(data);
  renderTargets(session.targets);
  renderProfile(session.profile);
  renderRestoredSession(data);
  renderWhitespace(data.whitespace || []);
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
  verdictEl.textContent = "UNWRITTEN";
  overallEl.textContent = "0.0";
  renderWoodMap(null);
  renderTargets([]);
  renderProfile({});
  renderIdeas([]);
  renderProjects([]);
  renderWhitespace([]);
  renderPlan([]);
}

function defaultSession(data = bootstrapData) {
  return {
    profile: {},
    targets: data?.default_targets || targetOptions.slice(0, 3),
  };
}

function bumpSessionRevision() {
  sessionRevision += 1;
  return sessionRevision;
}

function isCurrentSessionRevision(revision) {
  return revision === sessionRevision;
}

function restoreExportButtonLabels() {
  exportNotesButton.textContent = "Notes";
  exportChapterButton.textContent = "Chapter";
  exportButton.textContent = PNG_EXPORT_LABEL;
}

function setSessionControlsDisabled(disabled) {
  sessionControlsLocked = disabled;
  targetsEl.querySelectorAll("input[data-target]").forEach((target) => {
    target.disabled = disabled;
  });
  profileEl.querySelectorAll("input[data-profile-field]").forEach((field) => {
    field.disabled = disabled;
  });
  ideasEl.querySelectorAll("button[data-idea-id]").forEach((idea) => {
    idea.disabled = disabled;
  });
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
  ink.textContent = "The book is open. The next page waits for its first line.";
  ink.classList.remove("thinking", "bleed", "gold");
  corrections.textContent = "Session reset.";
  renderTargets(session.targets);
  renderProfile(session.profile);
  renderScore(null);
  verdictEl.textContent = "UNWRITTEN";
  overallEl.textContent = "0.0";
  renderWoodMap(null);
  renderIdeas([]);
  renderPlan([]);
  renderProjects(bootstrapData.top_projects || []);
  renderWhitespace(bootstrapData.whitespace || []);
  restoreExportButtonLabels();
  exportButton.disabled = true;
  setButtonDisabled(exportNotesButton, true);
  setButtonDisabled(exportChapterButton, true);
  saveSession();
  input.focus();
}

async function loadDemoSession() {
  bumpSessionRevision();
  submit.disabled = true;
  setCommandDisabled(true);
  setSessionControlsDisabled(true);
  ink.classList.remove("bleed", "gold");
  ink.classList.add("thinking");
  ink.textContent = "A sample page is being inked.";
  corrections.textContent = "";
  try {
    const response = await fetch("/api/demo-session");
    if (!response.ok) throw new Error(`demo rehearsal failed with ${response.status}`);
    applyDemoSession(await response.json());
  } catch (error) {
    ink.textContent = `The demo rehearsal could not be loaded: ${error.message}`;
    ink.classList.remove("thinking");
    ink.classList.add("bleed");
  } finally {
    submit.disabled = false;
    setCommandDisabled(false);
    setSessionControlsDisabled(false);
    input.focus();
  }
}

function applyDemoSession(data) {
  session = data.session || {};
  session.profile = session.profile || {};
  session.targets = Array.isArray(session.targets) ? session.targets : [];
  session.last_response = data.response || session.last_response || "";
  session.ui_status = `example loaded: ${data.turn_count || 0} advisor turns`;
  currentArtifact = data.artifact || session.last_artifact || null;
  ink.textContent = data.response || "Demo rehearsal loaded.";
  ink.classList.remove("thinking");
  if (data.score) {
    verdictEl.textContent = data.score.verdict;
    overallEl.textContent = Number(data.score.overall).toFixed(1);
    renderScore(data.score);
    ink.classList.toggle("bleed", data.score.verdict.startsWith("ECHO"));
    ink.classList.toggle("gold", data.score.verdict.startsWith("UNWRITTEN"));
  }
  renderTargets(session.targets);
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
  exportButton.disabled = !currentArtifact;
  setButtonDisabled(exportNotesButton, !(session.trace?.length));
  setButtonDisabled(exportChapterButton, !(session.ideas?.length));
  corrections.textContent = session.ui_status;
  saveSession();
}

function renderProvenance(data) {
  const snapshot = shortDate(data.snapshot_generated_at);
  const index = shortDate(data.index_generated_at);
  const digest = String(data.snapshot_digest || "").slice(0, 10);
  provenanceEl.textContent = `${data.index_algorithm || "index"} · snapshot ${snapshot} · index ${index} · ${digest}`;
}

function renderRestoredSession(data) {
  const idea = currentIdea();
  const storedArtifact = session.last_artifact || null;
  currentArtifact = !idea || storedArtifact?.title === idea.title ? storedArtifact : null;
  const score = currentArtifact?.seal || idea?.score || null;
  if (score) {
    renderScore(score);
    const verdict = currentArtifact?.verdict || score.verdict || "UNWRITTEN";
    verdictEl.textContent = verdict;
    overallEl.textContent = Number(currentArtifact?.overall || score.overall || 0).toFixed(1);
    ink.classList.toggle("bleed", verdict.startsWith("ECHO"));
    ink.classList.toggle("gold", verdict.startsWith("UNWRITTEN"));
    renderWoodMap(currentArtifact?.wood_map || null);
    if (score.echoes?.length) {
      renderCitations(score.echoes);
    } else {
      renderProjects(data.top_projects || []);
    }
    exportButton.disabled = !currentArtifact;
  } else {
    renderScore(null);
    renderWoodMap(null);
    renderProjects(data.top_projects || []);
    exportButton.disabled = true;
  }
  renderIdeas(session.ideas || []);
  renderPlan(session.last_plan || []);
  setButtonDisabled(exportNotesButton, !(session.trace?.length));
  setButtonDisabled(exportChapterButton, !(session.ideas?.length));
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
  const savedTargets = Array.isArray(savedSession.targets) ? savedSession.targets : defaultSession.targets;
  normalized.targets = targetOptions.filter((option) => savedTargets.includes(option));
  if (!normalized.targets.length && defaultSession.targets?.length) normalized.targets = [...defaultSession.targets];
  if (Array.isArray(savedSession.ideas)) normalized.ideas = savedSession.ideas;
  if (Array.isArray(savedSession.trace)) normalized.trace = savedSession.trace;
  if (Array.isArray(savedSession.last_plan)) normalized.last_plan = savedSession.last_plan;
  if (savedSession.current_idea_id) normalized.current_idea_id = savedSession.current_idea_id;
  if (savedSession.current_whitespace) normalized.current_whitespace = savedSession.current_whitespace;
  if (savedSession.last_tool_resolution) normalized.last_tool_resolution = savedSession.last_tool_resolution;
  if (savedSession.last_artifact) normalized.last_artifact = savedSession.last_artifact;
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

function normalizeTargetProfiles(profiles, options) {
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

function renderTargets(selectedTargets) {
  const selected = new Set(selectedTargets || []);
  targetsEl.innerHTML = "";
  if (!targetOptions.length) {
    targetsEl.innerHTML = `<div class="empty">No goals loaded.</div>`;
    return;
  }
  for (const option of targetOptions) {
    const profile = targetProfileById.get(option) || { label: option, description: "" };
    const label = document.createElement("label");
    label.className = "target-toggle";
    label.innerHTML = `
      <input
        type="checkbox"
        data-target="${escapeAttribute(option)}"
        aria-label="${escapeAttribute(profile.label)}"
        ${sessionControlsLocked ? "disabled" : ""}
        ${selected.has(option) ? "checked" : ""}
      />
      <span class="target-copy">
        <strong>${escapeHtml(profile.label)}</strong>
        ${profile.description ? `<small>${escapeHtml(profile.description)}</small>` : ""}
      </span>
    `;
    targetsEl.append(label);
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

  if (event.type === "token") {
    markFirstTokenSeen();
    ink.textContent += event.text;
    return;
  }

  if (event.type === "done") {
    if (!sawTurnToken) {
      clearTurnWatchdog();
      ink.textContent = event.response || ink.textContent;
      ink.classList.remove("thinking");
    }
    session = event.state || {};
    session.profile = session.profile || {};
    session.targets = Array.isArray(session.targets) ? session.targets : [];
    session.last_response = event.response || session.last_response || "";
    delete session.ui_status;
    if (event.score?.echoes?.length) {
      renderCitations(event.score.echoes);
    } else if (event.projects?.length) {
      renderProjects(event.projects);
    }
    if (event.whitespace?.length) renderWhitespace(event.whitespace);
    renderTargets(session.targets);
    renderProfile(session.profile);
    renderIdeas(session.ideas || []);
    renderPlan(event.plan || []);
    if (event.score) {
      verdictEl.textContent = event.score.verdict;
      overallEl.textContent = Number(event.score.overall).toFixed(1);
      renderScore(event.score);
      ink.classList.toggle("bleed", event.score.verdict.startsWith("ECHO"));
      ink.classList.toggle("gold", event.score.verdict.startsWith("UNWRITTEN"));
    }
    if (event.artifact?.title) {
      currentArtifact = event.artifact;
      renderWoodMap(event.artifact.wood_map || null);
      exportButton.disabled = false;
    }
    setButtonDisabled(exportNotesButton, !(session.trace?.length));
    setButtonDisabled(exportChapterButton, !(session.ideas?.length));
    saveSession();
  }
}

function renderIdeas(ideas) {
  ideasEl.innerHTML = "";
  if (!ideas.length) {
    ideasEl.innerHTML = `<div class="empty">No pages written.</div>`;
    return;
  }
  for (const idea of visibleIdeas(ideas)) {
    const score = idea.score?.overall ? Number(idea.score.overall).toFixed(1) : "0.0";
    const targets = (idea.targets || []).slice(0, 3).map(targetDisplayName).join(" · ");
    const selected = idea.id === session.current_idea_id;
    const item = document.createElement("button");
    item.type = "button";
    item.className = `idea ${selected ? "current" : ""}`;
    item.disabled = sessionControlsLocked;
    item.dataset.ideaId = idea.id || "";
    item.setAttribute("aria-pressed", selected ? "true" : "false");
    item.innerHTML = `
      <strong>${escapeHtml(idea.title)}</strong>
      <p>${escapeHtml((idea.pitch || "").slice(0, 120))}</p>
      <span>${escapeHtml(idea.score?.verdict || "DRAFT")} · ${score}</span>
      ${targets ? `<small>${escapeHtml(targets)}</small>` : ""}
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

function currentIdea() {
  const ideas = Array.isArray(session.ideas) ? session.ideas : [];
  return ideas.find((idea) => idea.id === session.current_idea_id) || ideas[ideas.length - 1] || null;
}

function selectIdea(ideaId) {
  if (!ideaId || !Array.isArray(session.ideas)) return;
  const idea = session.ideas.find((item) => item.id === ideaId);
  if (!idea) return;
  bumpSessionRevision();
  session.current_idea_id = idea.id;
  if (Array.isArray(idea.targets) && idea.targets.length) {
    session.targets = targetOptions.filter((option) => idea.targets.includes(option));
  }
  const score = idea.score || null;
  if (score) {
    verdictEl.textContent = score.verdict || "DRAFT";
    overallEl.textContent = Number(score.overall || 0).toFixed(1);
    renderScore(score);
    ink.classList.toggle("bleed", String(score.verdict || "").startsWith("ECHO"));
    ink.classList.toggle("gold", String(score.verdict || "").startsWith("UNWRITTEN"));
  }
  if (session.last_artifact?.title === idea.title) {
    currentArtifact = session.last_artifact;
    renderWoodMap(currentArtifact.wood_map || null);
    exportButton.disabled = false;
  } else {
    currentArtifact = null;
    renderWoodMap(null);
    exportButton.disabled = true;
  }
  renderTargets(session.targets || []);
  renderIdeas(session.ideas);
  renderPlan([]);
  session.ui_status = `selected: ${idea.title}`;
  corrections.textContent = session.ui_status;
  saveSession();
}

function targetDisplayName(target) {
  return targetProfileById.get(target)?.label || target;
}

function renderScore(score) {
  const rows = [
    ["Originality", score?.originality || 0],
    ["Delight", score?.delight || 0],
    ["AI Need", score?.ai_necessity || 0],
    ["Feasible", score?.feasibility || 0],
    ["Prize Fit", score?.prize_fit || 0],
  ];
  scoreEl.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="score-row">
          <span>${label}</span>
          <meter min="0" max="10" value="${value}"></meter>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");
}

function renderWoodMap(map) {
  woodMapEl.innerHTML = "";
  if (!map?.dots?.length) {
    woodMapEl.innerHTML = `<div class="empty">No page has been placed yet.</div>`;
    return;
  }
  const field = document.createElement("div");
  field.className = "wood-map-field";
  for (const dot of map.dots) {
    const marker = document.createElement(dot.url ? "a" : "span");
    const verdictClass = dot.kind === "idea" && String(dot.verdict || "").startsWith("ECHO") ? "echo-idea" : "";
    marker.className = `wood-dot ${dot.kind || "inked"} ${verdictClass}`.trim();
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
  const caption = document.createElement("p");
  caption.className = "wood-map-caption";
  caption.textContent = map.caption || "Your page is plotted against the current Wood.";
  woodMapEl.append(field, caption);
}

function renderProjects(projects) {
  projectsEl.innerHTML = "";
  if (!projects.length) {
    projectsEl.innerHTML = `<div class="empty">No red ink yet.</div>`;
    return;
  }
  for (const project of projects.slice(0, 5)) {
    const item = document.createElement("a");
    item.className = "project";
    item.href = project.url;
    item.target = "_blank";
    item.rel = "noreferrer";
    item.innerHTML = `
      <strong>${escapeHtml(project.title)}</strong>
      <p>${escapeHtml(project.summary || project.id)}</p>
    `;
    projectsEl.append(item);
  }
}

function renderCitations(echoes) {
  projectsEl.innerHTML = "";
  if (!echoes.length) {
    projectsEl.innerHTML = `<div class="empty">No red ink yet.</div>`;
    return;
  }
  for (const echo of echoes.slice(0, 5)) {
    const project = echo.project || {};
    const item = document.createElement("a");
    item.className = "project citation";
    item.href = project.url || project.host || "#";
    item.target = "_blank";
    item.rel = "noreferrer";
    item.title = project.title || project.id || "Project citation";
    const matched = (echo.matched_terms || []).slice(0, 5).join(", ") || "no shared terms";
    item.innerHTML = `
      <strong>Page ${escapeHtml(echo.page_number || "?")} · ${escapeHtml(project.title || project.id || "Untitled")}</strong>
      <p>${escapeHtml(project.summary || project.id || "")}</p>
      <span>${Number(echo.score || 0).toFixed(3)} · ${escapeHtml(matched)}</span>
    `;
    projectsEl.append(item);
  }
}

function renderWhitespace(items) {
  whitespaceEl.innerHTML = "";
  if (!items.length) {
    whitespaceEl.innerHTML = `<div class="empty">Gold has not gathered.</div>`;
    return;
  }
  for (const item of items.slice(0, 4)) {
    const gap = document.createElement("div");
    gap.className = "gap";
    gap.innerHTML = `
      <strong>${escapeHtml(item.label)}</strong>
      <p>${escapeHtml(item.pitch)}</p>
    `;
    whitespaceEl.append(gap);
  }
}

function renderPlan(steps) {
  planEl.innerHTML = "";
  if (!steps.length) {
    planEl.innerHTML = `<li class="empty">No wax path pressed.</li>`;
    return;
  }
  for (const step of steps) {
    const item = document.createElement("li");
    item.textContent = step;
    planEl.append(item);
  }
}

function setButtonDisabled(button, disabled) {
  if (button) button.disabled = disabled;
}

function setCommandDisabled(disabled) {
  document.querySelectorAll(".command-row button").forEach((button) => {
    const isArtifact = button.id === "export-artifact";
    const isNotes = button.id === "export-notes";
    const isChapter = button.id === "export-chapter";
    button.disabled =
      disabled ||
      (isArtifact && !currentArtifact) ||
      (isNotes && !session.trace?.length) ||
      (isChapter && !session.ideas?.length);
  });
}

function startTurnWatchdog() {
  clearTurnWatchdog();
  sawTurnToken = false;
  ink.textContent = "The page is choosing its words.";
  ink.classList.add("thinking");
  turnWatchdog = window.setTimeout(() => {
    if (sawTurnToken) return;
    ink.textContent = "Still riffling the inked pages.";
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

function syncCurrentIdeaTargets() {
  const currentId = session.current_idea_id;
  if (!currentId || !Array.isArray(session.ideas)) return;
  const idea = session.ideas.find((item) => item.id === currentId);
  if (idea) idea.targets = [...(session.targets || [])];
}

async function exportNotes() {
  await exportMarkdown({
    endpoint: "/field_notes",
    filename: FIELD_NOTES_FILENAME,
    button: exportNotesButton,
    busyLabel: "Notes...",
    pendingLabel: "Writing notes.",
    successLabel: "Notes saved",
  });
}

async function exportChapter() {
  await exportMarkdown({
    endpoint: "/chapter",
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
  const idleLabel = button.textContent;
  button.disabled = true;
  button.textContent = busyLabel;
  session.ui_status = pendingLabel;
  corrections.textContent = session.ui_status;
  saveSession();
  try {
    const client = await clientPromise;
    const result = await client.predict(endpoint, {
      session_json: JSON.stringify(session),
    });
    const data = Array.isArray(result.data) ? result.data[0] : result.data;
    const text = String(data || "");
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
    button.textContent = idleLabel;
    if (!isCurrentSessionRevision(revision)) return;
    saveSession();
    setCommandDisabled(false);
  }
}

function exportArtifact(artifact) {
  const idleLabel = exportButton.textContent;
  exportButton.disabled = true;
  exportButton.textContent = "PNG...";
  session.ui_status = "Drawing PNG.";
  corrections.textContent = session.ui_status;
  saveSession();
  try {
    const filename = `${slugify(artifact.title || "unwritten-page")}.png`;
    const canvas = renderArtifactCanvas(artifact);
    const dataUrl = canvas.toDataURL("image/png");
    if (!dataUrl.startsWith("data:image/png")) throw new Error("PNG rendering failed");
    const link = document.createElement("a");
    link.download = filename;
    link.href = dataUrl;
    link.click();
    session.ui_status = `PNG saved: ${filename}`;
    corrections.textContent = session.ui_status;
  } catch (error) {
    session.ui_status = `Export failed: ${error.message}`;
    corrections.textContent = session.ui_status;
  } finally {
    saveSession();
    exportButton.textContent = idleLabel || PNG_EXPORT_LABEL;
    setCommandDisabled(false);
  }
}

function renderArtifactCanvas(artifact) {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 675;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas is unavailable");
  drawParchment(ctx, canvas.width, canvas.height);
  const seal = artifact.seal || {};
  ctx.fillStyle = "#25160e";
  ctx.font = "700 58px Georgia, serif";
  wrapText(ctx, artifact.title, 78, 112, 760, 66);
  ctx.font = "28px Georgia, serif";
  ctx.fillStyle = "#6b4e35";
  wrapText(ctx, artifact.caption || "", 82, 252, 720, 36);
  drawCitationList(ctx, seal.echoes || [], 742, 330, 330);

  ctx.save();
  ctx.translate(930, 226);
  ctx.rotate(-0.08);
  ctx.fillStyle = artifact.verdict?.startsWith("UNWRITTEN") ? "#b68a12" : "#8d2d26";
  ctx.beginPath();
  ctx.arc(0, 0, 120, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff0b5";
  ctx.textAlign = "center";
  ctx.font = "800 27px Inter, sans-serif";
  wrapText(ctx, artifact.verdict || "UNWRITTEN", -92, -28, 184, 32, "center");
  ctx.font = "700 58px Georgia, serif";
  ctx.fillText(Number(artifact.overall || seal.overall || 0).toFixed(1), 0, 48);
  ctx.restore();

  const rows = [
    ["Originality", seal.originality || 0],
    ["Delight", seal.delight || 0],
    ["AI Need", seal.ai_necessity || 0],
    ["Feasible", seal.feasibility || 0],
    ["Prize Fit", seal.prize_fit || 0],
  ];
  rows.forEach(([label, value], index) => {
    const y = 418 + index * 34;
    ctx.fillStyle = "#6b4e35";
    ctx.font = "700 20px Inter, sans-serif";
    ctx.fillText(label, 82, y);
    ctx.fillStyle = "rgba(80, 47, 22, 0.22)";
    ctx.fillRect(240, y - 17, 320, 16);
    ctx.fillStyle = artifact.verdict?.startsWith("UNWRITTEN") ? "#2f7a49" : "#8d2d26";
    ctx.fillRect(240, y - 17, 32 * Number(value), 16);
    ctx.fillStyle = "#25160e";
    ctx.fillText(String(value), 582, y);
  });
  drawWoodMap(ctx, artifact.wood_map, 742, 396, 330, 184, artifact.verdict);
  return canvas;
}

function downloadText(filename, text, type = "application/jsonl;charset=utf-8") {
  const blob = new Blob([text], { type });
  const link = document.createElement("a");
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

function drawParchment(ctx, width, height) {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#ead7a7");
  gradient.addColorStop(0.55, "#d4b476");
  gradient.addColorStop(1, "#b98a4c");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(59, 33, 15, 0.16)";
  for (let i = 0; i < 360; i += 1) {
    const x = (i * 73) % width;
    const y = (i * 37) % height;
    ctx.fillRect(x, y, 2 + (i % 7), 1);
  }
  ctx.strokeStyle = "rgba(72, 39, 18, 0.42)";
  ctx.lineWidth = 16;
  ctx.strokeRect(28, 28, width - 56, height - 56);
}

function drawWoodMap(ctx, map, x, y, width, height, verdict) {
  if (!map?.dots?.length) return;
  ctx.save();
  ctx.fillStyle = "rgba(255, 241, 196, 0.38)";
  ctx.strokeStyle = "rgba(80, 47, 22, 0.34)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#6b4e35";
  ctx.font = "800 18px Inter, sans-serif";
  ctx.fillText("YOU VS THE WOOD", x, y - 14);

  for (const dot of map.dots) {
    const px = x + (width * boundedPercent(dot.x)) / 100;
    const py = y + (height * boundedPercent(dot.y)) / 100;
    const radius = Math.max(3, Math.min(10, Number(dot.radius || 4)));
    if (dot.kind === "idea") {
      ctx.fillStyle = verdict?.startsWith("UNWRITTEN") ? "#2f7a49" : "#8d2d26";
      ctx.strokeStyle = "#fff0b5";
      ctx.lineWidth = 3;
    } else if (dot.kind === "echo") {
      ctx.fillStyle = "#8d2d26";
      ctx.strokeStyle = "rgba(255, 240, 181, 0.72)";
      ctx.lineWidth = 1.5;
    } else {
      ctx.fillStyle = "rgba(80, 47, 22, 0.34)";
      ctx.strokeStyle = "transparent";
      ctx.lineWidth = 0;
    }
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    if (ctx.lineWidth) ctx.stroke();
  }

  ctx.fillStyle = "#6b4e35";
  ctx.font = "700 15px Inter, sans-serif";
  wrapText(ctx, map.caption || "", x, y + height + 24, width, 20);
  ctx.restore();
}

function drawCitationList(ctx, echoes, x, y, maxWidth) {
  if (!echoes.length) return;
  ctx.save();
  ctx.fillStyle = "#6b4e35";
  ctx.font = "800 18px Inter, sans-serif";
  ctx.fillText("CLOSEST PAGES", x, y);
  ctx.font = "700 15px Inter, sans-serif";
  echoes.slice(0, 3).forEach((echo, index) => {
    const project = echo.project || {};
    const label = `Page ${echo.page_number || "?"}: ${project.title || project.id || "Untitled"}`;
    wrapText(ctx, label, x, y + 24 + index * 26, maxWidth, 18);
  });
  ctx.restore();
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight, align = "left") {
  const words = String(text).split(/\s+/);
  let line = "";
  const originalAlign = ctx.textAlign;
  ctx.textAlign = align;
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width > maxWidth && line) {
      ctx.fillText(line, align === "center" ? x + maxWidth / 2 : x, y);
      line = word;
      y += lineHeight;
    } else {
      line = next;
    }
  }
  if (line) ctx.fillText(line, align === "center" ? x + maxWidth / 2 : x, y);
  ctx.textAlign = originalAlign;
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
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

function boundedPercent(value) {
  return Math.max(4, Math.min(96, Number(value || 50)));
}

function shortDate(value) {
  if (!value) return "unknown";
  return String(value).replace("T", " ").replace(/\+00:00$/, "Z").slice(0, 16);
}
