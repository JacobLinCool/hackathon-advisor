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
const traceEl = document.querySelector("#trace");
const provenanceEl = document.querySelector("#provenance");
const verdictEl = document.querySelector("#verdict");
const overallEl = document.querySelector("#overall");
const exportButton = document.querySelector("#export-artifact");
const exportTraceButton = document.querySelector("#export-trace");
const exportNotesButton = document.querySelector("#export-notes");

let session = {};
let clientPromise = Client.connect(window.location.origin);
let currentArtifact = null;
let targetOptions = [];
let profileFields = [];

bootstrap();

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

exportButton.addEventListener("click", () => {
  if (!currentArtifact) return;
  exportArtifact(currentArtifact);
});

exportTraceButton.addEventListener("click", async () => {
  await exportTrace();
});

exportNotesButton.addEventListener("click", async () => {
  await exportNotes();
});

targetsEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.dataset.target) return;
  const checked = new Set(
    Array.from(targetsEl.querySelectorAll("input[data-target]:checked")).map((input) => input.dataset.target),
  );
  session.targets = targetOptions.filter((option) => checked.has(option));
  syncCurrentIdeaTargets();
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
});

async function runTurn(message) {
  input.value = "";
  submit.disabled = true;
  setCommandDisabled(true);
  ink.textContent = "";
  ink.classList.remove("bleed", "gold");
  corrections.textContent = "";
  planEl.innerHTML = "";

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
    ink.textContent = `The page tore before it could answer: ${error.message}`;
    ink.classList.add("bleed");
  } finally {
    submit.disabled = false;
    setCommandDisabled(false);
    input.focus();
  }
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  const data = await response.json();
  targetOptions = data.target_options || [];
  profileFields = data.profile_fields || [];
  session = {
    profile: {},
    targets: data.default_targets || targetOptions.slice(0, 3),
  };
  renderProvenance(data);
  renderTargets(session.targets);
  renderProfile(session.profile);
  renderProjects(data.top_projects || []);
  renderWhitespace(data.whitespace || []);
  renderIdeas([]);
  renderWoodMap(null);
  renderScore(null);
  renderPlan([]);
  renderTrace([]);
}

function renderProvenance(data) {
  const snapshot = shortDate(data.snapshot_generated_at);
  const index = shortDate(data.index_generated_at);
  const digest = String(data.snapshot_digest || "").slice(0, 10);
  provenanceEl.textContent = `${data.index_algorithm || "index"} · snapshot ${snapshot} · index ${index} · ${digest}`;
}

function renderTargets(selectedTargets) {
  const selected = new Set(selectedTargets || []);
  targetsEl.innerHTML = "";
  if (!targetOptions.length) {
    targetsEl.innerHTML = `<div class="empty">No seals loaded.</div>`;
    return;
  }
  for (const option of targetOptions) {
    const label = document.createElement("label");
    label.className = "target-toggle";
    label.innerHTML = `
      <input type="checkbox" data-target="${escapeAttribute(option)}" ${selected.has(option) ? "checked" : ""} />
      <span>${escapeHtml(option)}</span>
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
    ink.textContent += event.text;
    return;
  }

  if (event.type === "done") {
    session = event.state || {};
    session.profile = session.profile || {};
    session.targets = Array.isArray(session.targets) ? session.targets : [];
    if (event.projects?.length) renderProjects(event.projects);
    if (event.whitespace?.length) renderWhitespace(event.whitespace);
    renderTargets(session.targets);
    renderProfile(session.profile);
    renderIdeas(session.ideas || []);
    renderTrace(session.trace || []);
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
    exportTraceButton.disabled = !(session.trace?.length);
    exportNotesButton.disabled = !(session.trace?.length);
  }
}

function renderIdeas(ideas) {
  ideasEl.innerHTML = "";
  if (!ideas.length) {
    ideasEl.innerHTML = `<div class="empty">No pages written.</div>`;
    return;
  }
  for (const idea of ideas.slice(-4).reverse()) {
    const score = idea.score?.overall ? Number(idea.score.overall).toFixed(1) : "0.0";
    const targets = (idea.targets || []).slice(0, 3).join(" · ");
    const item = document.createElement("div");
    item.className = "idea";
    item.innerHTML = `
      <strong>${escapeHtml(idea.title)}</strong>
      <p>${escapeHtml((idea.pitch || "").slice(0, 120))}</p>
      <span>${escapeHtml(idea.score?.verdict || "DRAFT")} · ${score}</span>
      ${targets ? `<small>${escapeHtml(targets)}</small>` : ""}
    `;
    ideasEl.append(item);
  }
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

function renderTrace(trace) {
  traceEl.innerHTML = "";
  if (!trace.length) {
    traceEl.innerHTML = `<div class="empty">No tool marks yet.</div>`;
    return;
  }
  for (const event of trace.slice(-4).reverse()) {
    const item = document.createElement("div");
    item.className = "trace";
    const tools = (event.tools || []).map((tool) => tool.name).join(" -> ") || "reply";
    item.innerHTML = `
      <strong>${escapeHtml(event.verdict || "TURN")} ${event.overall ? Number(event.overall).toFixed(1) : ""}</strong>
      <p>${escapeHtml(tools)}</p>
    `;
    traceEl.append(item);
  }
}

function setCommandDisabled(disabled) {
  document.querySelectorAll(".command-row button").forEach((button) => {
    const isArtifact = button.id === "export-artifact";
    const isTrace = button.id === "export-trace";
    const isNotes = button.id === "export-notes";
    button.disabled =
      disabled ||
      (isArtifact && !currentArtifact) ||
      (isTrace && !session.trace?.length) ||
      (isNotes && !session.trace?.length);
  });
}

function syncCurrentIdeaTargets() {
  const currentId = session.current_idea_id;
  if (!currentId || !Array.isArray(session.ideas)) return;
  const idea = session.ideas.find((item) => item.id === currentId);
  if (idea) idea.targets = [...(session.targets || [])];
}

async function exportTrace() {
  const client = await clientPromise;
  const result = await client.predict("/trace_artifact", {
    session_json: JSON.stringify(session),
  });
  const data = Array.isArray(result.data) ? result.data[0] : result.data;
  downloadText("hackathon-advisor-trace.jsonl", String(data || ""));
}

async function exportNotes() {
  const client = await clientPromise;
  const result = await client.predict("/field_notes", {
    session_json: JSON.stringify(session),
  });
  const data = Array.isArray(result.data) ? result.data[0] : result.data;
  downloadText("hackathon-advisor-field-notes.md", String(data || ""), "text/markdown;charset=utf-8");
}

function exportArtifact(artifact) {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 675;
  const ctx = canvas.getContext("2d");
  drawParchment(ctx, canvas.width, canvas.height);
  const seal = artifact.seal || {};
  ctx.fillStyle = "#25160e";
  ctx.font = "700 58px Georgia, serif";
  wrapText(ctx, artifact.title, 78, 112, 760, 66);
  ctx.font = "28px Georgia, serif";
  ctx.fillStyle = "#6b4e35";
  wrapText(ctx, artifact.caption || "", 82, 252, 720, 36);

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

  const link = document.createElement("a");
  link.download = `${slugify(artifact.title || "unwritten-page")}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
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
