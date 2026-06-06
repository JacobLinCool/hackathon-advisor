import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const form = document.querySelector("#turn-form");
const input = document.querySelector("#message");
const submit = document.querySelector("#submit");
const ink = document.querySelector("#ink");
const corrections = document.querySelector("#corrections");
const projectsEl = document.querySelector("#projects");
const whitespaceEl = document.querySelector("#whitespace");
const verdictEl = document.querySelector("#verdict");
const overallEl = document.querySelector("#overall");

let session = {};
let clientPromise = Client.connect(window.location.origin);

bootstrap();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  submit.disabled = true;
  ink.textContent = "";
  ink.classList.remove("bleed", "gold");
  corrections.textContent = "";

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
    input.focus();
  }
});

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  const data = await response.json();
  renderProjects(data.top_projects || []);
  renderWhitespace(data.whitespace || []);
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
    if (event.projects?.length) renderProjects(event.projects);
    if (event.whitespace?.length) renderWhitespace(event.whitespace);
    if (event.score) {
      verdictEl.textContent = event.score.verdict;
      overallEl.textContent = Number(event.score.overall).toFixed(1);
      ink.classList.toggle("bleed", event.score.verdict.startsWith("ECHO"));
      ink.classList.toggle("gold", event.score.verdict.startsWith("UNWRITTEN"));
    }
  }
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
