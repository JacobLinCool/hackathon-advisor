import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const form = document.querySelector("#turn-form");
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

const SESSION_STORAGE_KEY = "hackathon-advisor-session-v2";
const FIELD_NOTES_FILENAME = "hackathon-advisor-field-notes.md";
const CHAPTER_FILENAME = "hackathon-advisor-chapter.md";
const PNG_EXPORT_LABEL = "PNG";

let session = {};
let clientPromise = Client.connect(window.location.origin);
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
  startTurnWatchdog();

  let completed = false;
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
    completed = true;
  } catch (error) {
    clearTurnWatchdog();
    ink.textContent = `The advisor could not answer: ${error.message}`;
    ink.classList.remove("thinking");
    ink.classList.add("bleed");
  } finally {
    clearTurnWatchdog();
    submit.disabled = false;
    setSessionControlsDisabled(false);
    setCommandDisabled(false);
    input.focus();
  }
  return completed;
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
  session.ui_status = `Example loaded: ${data.turn_count || 0} advisor turns`;
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
    const verdictClass = dot.kind === "idea" && String(dot.verdict || "").startsWith("ECHO") ? "bleed" : "";
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
    whitespaceEl.innerHTML = `<div class="empty">No under-explored directions are loaded yet.</div>`;
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
  const idleLabel = actionButtonLabel(button);
  button.disabled = true;
  setActionButtonLabel(button, busyLabel);
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
