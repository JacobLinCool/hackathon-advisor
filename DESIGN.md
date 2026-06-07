# Build Small Hackathon Advisor — Design & Implementation Notes

> A **small-model agent** with text and voice input that investigates what other people have already built
> for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon) and brainstorms an original new design
> *with you*. Output = streaming text + live visuals (no TTS). All models small, open-weight, run locally.
>
> The literal "advisor" is the **engine**; the user-facing experience is **The Unwritten Almanac** — Mothback, an
> owl-moth archivist, keeps the Wood's book of fates and divines you a still-unwritten project page (ink **bleeds +
> cites real Spaces** if you overlap, **blooms gold** if it's new). This project is itself a Build Small submission
> (hack window 2026-06-05 → 2026-06-15).

---

## 1. Locked decisions & review corrections (2026-06-07)

A multi-agent adversarial review (5 dimensions, web-verified) set the direction below. **This section is the
authoritative decision log; the rest of the doc is written to be consistent with it.**

**Locked decisions (Jacob):**
1. **Concept = The Unwritten Almanac** (chosen 2026-06-07 from a 12-concept brainstorm). Mothback the owl-moth
   archivist divines a fate-page; ink **bleeds and cites the real Spaces** you overlap (page 47, page 112…), or
   **blooms gold + sprouts a leaf** when it's unwritten. Engine unchanged underneath (crawl → whitespace/originality →
   score). The dry "advisor" stays under the hood. Full spec + de-risking grafts in §2.
2. **Text-first with voice input.** The core workflow remains typed/editable text. Voice records or uploads a note,
   transcribes it with batch ASR, and places the transcript in the same idea box. Real-time streaming + in-browser turn
   detection are **deferred**.
3. **Add a 🎯 Well-Tuned fine-tune** — a small LoRA (MiniCPM5 advisor persona / tool-calling), trained on Modal,
   published to the Hub → 6/6 badges → strong shot at 🎖️ Bonus Quest Champion ($2,000).
4. **ASR = Nemotron batch.** `nvidia/nemotron-speech-streaming-en-0.6b` runs through NVIDIA NeMo in a ZeroGPU function.
   Audio is normalized to mono WAV before calling `transcribe([wav])`.

**Verified corrections:**
- **Drop SGLang.** It needs a persistent GPU process → incompatible with ZeroGPU (same root cause as vLLM). Run
  MiniCPM5 via plain `transformers` inside `@spaces.GPU` and parse its XML tool calls in our own code.
- **gr.Server custom UI streaming IS shipped** (the launch blog only deferred the *explanation*). The deployed browser
  UI calls our own same-origin `/api/agent-turn` NDJSON stream with `fetch`; `_engine_turn` itself is wrapped in
  `@spaces.GPU`, so the real MiniCPM5 + LoRA path still runs on ZeroGPU. The `@app.api("/agent_turn")` generator stays
  available for Gradio/Python clients and contract checks, but the visible app no longer depends on the CDN
  `@gradio/client` path after real Space testing showed that browser turn could hang while the backend completed.
- **OpenAI Track has NO model requirement** ("OpenAI's own podium across all submissions") → auto-entered; a free
  lottery ticket. Do NOT add gpt-oss (breaks Tiny Titan, dilutes the small-model thesis). Deliberate non-target.
- **Badges = 6 total** (Tiny Titan is a $1.5k *special award*, not a badge). Decision #3 takes us from 5/6 → 6/6.
- **Tiny Titan** = "best ≤4B model"; our largest single model is MiniCPM5 (1.08B), total stack ~1.9B → eligible.

**New build requirements surfaced by the review (designed into the sections below):**
- **Jargon alias layer (§7):** a 0.6B ASR mistranscribes our own vocab (Nemotron, MiniCPM5, EmbeddingGemma, ZeroGPU…).
  Deterministic code-side fuzzy/alias map over our small CLOSED vocab, applied before any tool call and before display.
  Surface "heard: neutron → Nemotron" as a delightful trust moment. (Active once voice is added.)
- **Tool-call degradation ladder (§8):** the 1B brain WILL emit broken tool calls (MiniCPM5-1B has a documented
  "broken tool calling" report). Wrap parse in try/except, retry once at low temp, validate name+args vs JSON-Schema in
  code (reject-and-repair), canned lines for empty results, a token **watchdog** that shows "trying again" instead of
  dead air (the screen is the only feedback channel — no TTS).
- **Latency / optimistic UI (§9/§11):** ZeroGPU cold start + 1B generation = seconds of potential dead air. Optimistic
  UI on submit, pre-animate the project wall, set a latency budget. (The torch.compile cold-start penalty does NOT
  apply — we don't use it.)

**Day-1 go/no-go spikes (before any feature work):**
- Trivial `@spaces.GPU` hello-cuda build GREEN on torch 2.8+, deps pinned, heavy deps added one at a time.
- `gr.Server` minimal: static `index.html` + one same-origin `/api/agent-turn` NDJSON stream, plus the retained
  `@app.api()` generator for external clients, on the real ZeroGPU Space.
- Nemotron `nemo_toolkit[asr]` install + one batch `transcribe()` inside `@spaces.GPU` (decision #4).

---

## 2. Concept — The Unwritten Almanac (text-first)

The engine, regardless of skin:

1. **Investigate** the `build-small-hackathon` HF org — what Spaces exist, which models, what's saturated, and where
   the **whitespace** is — using a local EmbeddingGemma index.
2. **Brainstorm** with the user: propose ideas, **score** them against a fixed rubric (originality vs. existing
   projects, delight, AI-necessity, feasibility, param budget, prize-fit), and maintain an **idea board**.
3. **Respond** as streaming text + live visuals in a custom `gr.Server` frontend (no TTS — the visual is the "voice").

**The skin (chosen): The Unwritten Almanac.** **Mothback**, a dusty owl-moth archivist, keeps the Wood's *book of
fates*. Every project already built in the org is an inked page; she divines you a destined entry on a still-blank page,
the ink writing itself live.

**The two-beat wow (this IS the engine, rendered):**
- You type one line about yourself / your idea. Inked pages riffle past (each = a real crawled Space).
- **Bleed:** if your idea overlaps existing work, the ink **seeps blood-red** and cites the exact real Spaces — "the
  Wood already wrote this, on page 47 and page 112" (= `get_project` overlap on the top retrieval hits). The burn is
  **factual**, so it can't fall flat the way a 1B's invented joke can.
- **Bloom:** you say "write bolder"; the next entry flows **gold**, a green leaf sprouts — "this page has never been
  inked" (= a `find_whitespace` gold candidate).
- A **wax seal** presses in, lighting five quadrants as the idea qualifies (= `score_idea`: Originality, Delight,
  AI-Necessity, Feasibility, Prize-Fit).

**Engine ↔ skin mapping:** `search_projects`/`get_project` overlap → the bleed + citations; `find_whitespace` → the
blank/gold pages; `score_idea` → the wax-seal quadrants; `save_idea` → the written fate-page; agent persona =
**Mothback** (Layer A system prompt + the 🎯 Well-Tuned LoRA = her voice).

**Shareable artifact (Community Choice):** the page exports as a PNG that looks **torn from an ancient grimoire** —
aged parchment, a coined fate-name as title, the self-written prophecy, the five-quadrant seal, and a verdict stamp
(**"UNWRITTEN · 0 echoes"** vs **"ECHO ×3"**). Built-in caption: "Mothback inked my fate page for #BuildSmall —
UNWRITTEN." People compile draws into a "chapter" and dare friends to get a page that doesn't bleed.

**Grafted de-risking (from runner-up concepts):**
- **Tone = dry-but-benevolent** (Roastleaf's whiplash): the bleed-citation gently stings, the gold-bloom is sincerely
  delighted; the burn is true-by-construction (real cited Spaces).
- **Templated structure (key risk-killer):** bank entry/roast templates (citation + dry verdict + redemptive branch);
  the 1B only fills in real Space titles + the idea — **never improvises whole comedy**.
- **Latin-binomial fate-names** (e.g. "Ludus Vocalis Infantium") via templated scaffolds — built-in wit, backstops a
  1B that might produce corny names.
- **"You vs the Wood" margin glyph:** a tiny cluster-dot thumbnail on the page showing your gold page among the inked
  crowd — cheap SVG, visual PROOF the gap is real.
- **Thin-org mitigation (load-bearing):** precompute whitespace clusters at Modal build-time and pin several DISTINCT
  blank-page candidates so "write bolder" always lands on a real, varied gap (the org may be only ~30–60 Spaces). Tune
  the echo threshold toward *more frequent bleed* so the demo always has its "low" before the "wow".

**Defaults (revisit if time):** single-page artifact first (chapter compiler later); page-numbers visible, real titles
on hover (keep the burn aimed at the idea, not a named builder); seal animation = safe typewriter + static-stamp floor
first, bespoke ink-reveal last. Voice input is batch ASR that fills the same idea box before the user presses Ink.

Input is **text-first**; the experience is fully delightful with typed input alone.

AI is genuinely **load-bearing**: embeddings power the whitespace/originality analysis and the LLM drives the
investigate → ideate → score loop — the experience collapses without the models (supports 🤖 Best Agent + TTW
"AI necessity").

---

## 3. Model stack (confirmed exact repo IDs)

| Role | Model | Params | Runtime | License | Prize hook |
|---|---|---|---|---|---|
| STT (batch voice input) | **`nvidia/nemotron-speech-streaming-en-0.6b`** | 0.6B | NeMo, GPU+CUDA | NVIDIA Open Model (commercial OK) | 🟩 NVIDIA Nemotron Quest |
| LLM brain | **`openbmb/MiniCPM5-1B`** ("OpenCPM5") | 1.08B | **transformers** (self-parse XML) / llama.cpp | **Apache-2.0** | 🏮 OpenBMB |
| Embedder | **`ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`** | ~300M | llama.cpp / llama-cpp-python | Gemma | 🔌 Off the Grid · 🦙 Llama Champion · 🟢 Modal |
| Fine-tune | LoRA on MiniCPM5 → published to Hub | — | PEFT / HF Jobs | — | 🎯 Well-Tuned |

**Total ≈ 1.98B params → ≤4B → 🐜 Tiny Titan eligible.** All open-weight, all runnable locally → 🔌 Off the Grid.

> Naming: "OpenCPM5 1B" = `openbmb/MiniCPM5-1B` (MiniCPM 5.0, ~May 2026). "EmbeddingGemma 270M" =
> `google/embeddinggemma-300m` (308M total; 270M = non-embedding transformer params). **SGLang dropped** (ZeroGPU
> incompatible). STT is used in **batch voice-note** mode, not a persistent stream.

---

## 4. Deployment & architecture (single path)

With **text-first + batch ASR**, the old "streaming ASR vs ZeroGPU" Config A/B tension dissolves — there is one path:

- **ZeroGPU Gradio-SDK Space** (free). GPU is attached only inside `@spaces.GPU` calls (default 60s, max ~120s,
  RTX Pro 6000 Blackwell, `large`=48 GB). Per-turn inference fits this model exactly.
- **Text-first runtime loop:** user types → custom `/api/agent-turn` NDJSON endpoint → one `@spaces.GPU` call runs
  MiniCPM5 (tool loop, in `transformers`) → streamed text tokens + live visual updates. The `@app.api()` endpoint
  remains as the Gradio-client contract for external checks.
- **Voice input:** push-to-talk records an utterance or uploads a voice note → `/api/transcribe` normalizes audio with
  ffmpeg → one `@spaces.GPU` call runs Nemotron ASR through NeMo → transcript fills the idea box. No persistent stream,
  no WebRTC, **no TURN server**.
- **Modal (build-time only):** crawl the org + build the llama.cpp EmbeddingGemma vector index offline; the Space ships
  with checked-in project vectors. Runtime never calls Modal → 🔌 Off the Grid holds (see §10).

> Off the Grid = no proprietary cloud inference APIs. Open weights on an HF GPU Space / local box / Modal all qualify.

**Deferred:** real-time streaming ASR and turn detection are not part of the shipped app.

---

## 5. Per-model implementation notes

### 5.1 ASR — `nvidia/nemotron-speech-streaming-en-0.6b` (batch)

- **Primary, batch usage (simple):**
  ```python
  import nemo.collections.asr as nemo_asr
  asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
  text = asr.transcribe(["utterance.wav"])      # 16 kHz mono WAV in; punctuated EN text out
  ```
  Runtime install: `packages.txt` provides `ffmpeg` and `libsndfile1`; `requirements.txt` pins
  `nemo_toolkit[asr]==2.7.3` plus Cython and packaging. The app records or uploads audio, normalizes it to mono
  16 kHz WAV, runs NeMo in a ZeroGPU function, then returns the transcript to the idea box. Hosted NVIDIA NIM API would
  break Off the Grid, so it is not used.

### 5.2 MiniCPM5-1B brain — `openbmb/MiniCPM5-1B` (transformers, self-parsed XML)

- Context 128K, bilingual (EN/ZH), Apache-2.0. `enable_thinking=False`, `temperature=0.7, top_p=0.95` for fast tool calls.
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
  model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", torch_dtype="auto", device_map="auto")
  inputs = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=False,
                                   tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
  ```
- **Tool calling:** pass JSON-Schema tools via the chat template `tools=` arg; the model emits **XML**
  `<function name="get_weather">{"city":"New York"}</function>`. **Parse this ourselves** (SGLang dropped). Wrap parse
  in try/except and validate against the schema — see the degradation ladder (§8).
- **Local / CPU & llama.cpp (Off the Grid · Llama Champion):** `openbmb/MiniCPM5-1B-GGUF:Q4_K_M` (688 MB) via llama.cpp
  or Ollama (CPU-viable). fp16 ≈ 3–4 GB VRAM. `openbmb/MiniCPM5-1B-MLX` for Apple Silicon. (llama.cpp MiniCPM5
  tool-calling is a pending PR — verify before relying on it for the badge runtime.)
- **1B discipline:** small tool schemas, few params each, clear descriptions, low temp, single-hop tool calls.

### 5.4 EmbeddingGemma GGUF — `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`

- Active retrieval model: `embeddinggemma-300m-qat-Q8_0.gguf`, 768-dimensional normalized embeddings.
- Build-time path: Modal remote function runs `llama-cpp-python` with mean pooling and writes `data/project_index.json`.
- Runtime path: Space embeds each user query through the same GGUF model via llama.cpp, then performs local cosine search
  over checked-in project vectors.
- Evidence is recorded in index metadata: model repo, GGUF filename, runtime, dimensions, build source, builder script,
  llama-cpp-python version, and Modal app name.

### 5.5 llama.cpp support (🦙 Llama Champion)

The active Llama Champion path is the retrieval model: the project index is built with EmbeddingGemma GGUF through
llama.cpp on Modal, and runtime query embeddings use the same llama.cpp path.

| Model | llama.cpp? | Runtime | Notes |
|---|---|---|---|
| `openbmb/MiniCPM5-1B` | ✅ planned only | llama.cpp / Ollama | Not used for deployed tool-calling; Transformers + LoRA is the deployed brain. |
| `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` | ✅ active | llama.cpp / llama-cpp-python | Builds project vectors on Modal and embeds runtime queries in the Space. |
| ASR (Nemotron) | ❌ | NeMo | FastConformer-RNNT |

The checked-in index and runtime query embedder must stay on the same GGUF file.

---

## 6. Agent context design (built for a 1B brain)

Core principle: **the 1B model is a router + arg-filler. All heavy work (crawl, summarize, score, rank, dedup) lives in
code.** Keep live context to ~800–1200 tokens of *curated* view, never raw data.

- **Layer A — System (static, ~250 tok):** identity/character; hackathon hard rules (≤32B, Gradio Space, demo video) so
  it self-filters infeasible ideas; targeted prizes (biases ideation); reply style (short, one question at a time);
  explicit tool-use instructions + the canonical jargon vocabulary (so it can self-correct, §7).
- **Layer B — Session state (re-rendered each turn by code, ~300 tok):** user profile; locked decisions (track, side
  quests, models); **idea board** (2–3 candidates, one line + scores); compact "projects already seen" summary.
- **Layer C — Ephemeral (~300 tok):** last 2–3 turns; the most recent tool result as a **refined card** (not raw JSON).

---

## 7. Agent tool design

Few tools, few params each, short descriptions (1B-friendly). Heavy logic in code.

**Jargon alias layer (input normalization).** Before any tool call and before display, run ASR/user text through a
deterministic fuzzy/alias map over our small CLOSED vocab (model names and goal names) — e.g. RapidFuzz
`token_set_ratio` / double-metaphone — mapping "neutron"/"nemo tron" → Nemotron, "mini cpm" → MiniCPM5, "zero gpu" →
ZeroGPU. Surface the correction ("heard: neutron → Nemotron") as a trust-building, slightly delightful moment.

**Research — investigate existing projects (the core value).** Data = `build-small-hackathon` org Spaces, pre-crawled
into a local snapshot + EmbeddingGemma index (keeps Off the Grid at runtime).

| Tool | Signature | Returns (refined) | Heavy work |
|---|---|---|---|
| `list_projects` | `(track?, sort?)` | top-N project cards | HF Hub API + summarize |
| `search_projects` | `(query)` | top 5 cards | EmbeddingGemma retrieval |
| `get_project` | `(id)` | card + overlap-vs-board verdict | code computes overlap |
| `find_whitespace` | `()` | under-explored niches | cluster the index, find gaps |

`find_whitespace` is the originality engine (TTW judges originality) — it names where nobody has built yet.

**Ideation / state.**

| Tool | Signature | Purpose |
|---|---|---|
| `save_idea` | `(title, pitch)` | add/update a candidate on the idea board |
| `score_idea` | `(id)` | fixed (hardcoded) rubric → scores + gaps; the 1B only triggers + verbalizes |
| `compare_ideas` | `()` | rank the board, articulate tradeoffs |
| `make_plan` | `(id)` | build plan + goals the current direction can support |
| `update_profile` | `(field, value)` | record skills/time/prefs → Layer B |
| `set_goals` | `(goals[])` | change selected goals → updates Layer A bias |

---

## 8. Agent loop (single-hop + degradation ladder)

```
on user input (text; or voice → batch ASR → text):
  normalize via jargon alias layer
  ctx = LayerA + render_state(LayerB) + last_turns + last_tool_card
  out = MiniCPM5(ctx, tools=TOOLS, enable_thinking=False, temp=0.7)   # → tool_call | reply
  try: parse XML tool call
  except / invalid name|args (vs JSON-Schema):                         # degradation ladder
      retry once (temp≈0.3, "emit ONLY one valid tool call")
      still bad → run a safe default tool (find_whitespace) so the screen never freezes
  if tool_call: card = run_tool(out); reply = MiniCPM5(ctx + card)     # single follow-up, no long ReAct
  empty/zero result → canned advisor line (never say nothing)
  stream reply tokens → custom UI   |   token watchdog: no token in N s → "trying again" visual (not dead air)
  update_state(LayerB)
```

**Max one tool-call then reply.** A 1B can't sustain multi-step ReAct; wrap multi-step flows (`search → get_project →
score`) into one *code* "research" action the model calls once. The degradation ladder is a **first-class UX surface**
(§11), not an error branch — the screen is the only feedback channel (no TTS).

---

## 9. ZeroGPU deployment notes

- `import spaces; @spaces.GPU(duration=…)`. GPU only inside decorated fns; **Gradio-SDK Space only** (no Docker ZeroGPU).
- Load models at **module level**, `.to('cuda')` once (emulated until first real GPU call); real compute inside the
  decorator. torch 2.8+; **no `torch.compile`** (use AOT). Quota PRO ~40 min/day → never idle-hold the GPU.
- **Frontend → backend via same-origin `fetch("/api/agent-turn")`** reading NDJSON from our FastAPI route. The GPU
  boundary is `_engine_turn`, decorated with `@spaces.GPU`; `@app.api()` endpoints remain available for Gradio-client
  tests and external callers.
- All four models fit in `large` (48 GB). Keep each `@spaces.GPU` call short for queue priority.

---

## 10. Modal — offline pipeline (build-time only → preserves Off the Grid)

Modal = build-time; runtime never calls it. This is how the app claims **both** 🟢 Modal and 🔌 Off the Grid. The
canonical command is:

```bash
.venv/bin/modal run scripts/modal_build_project_index.py \
  --projects data/projects.json \
  --out data/project_index.json
```

The remote function installs `llama-cpp-python`, downloads
`ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf`, embeds every project card through
llama.cpp, and returns a schema-v2 JSON index. The local entrypoint writes that payload into the repo for Space runtime.

Latest successful run: `hackathon-advisor-llama-index` on Modal, producing a 100-document, 768-dimensional normalized
index at `2026-06-07T08:16:19+00:00`.

---

## 11. Frontend — `gr.Server` custom UI (🎨 Off-Brand)

No TTS → the **visual output is the agent's "voice"**; it must carry the delight (this is what earns Off-Brand, and the
TTW polish + Best Demo score). The visual world is **The Unwritten Almanac** (§2): a candlelit tree-hollow with a heavy
open grimoire as the hero component.

- `gradio.Server` is a FastAPI subclass serving **your own frontend** while still exposing `@app.api(name=...)`
  functions for Gradio/Python clients. The visible app uses first-party `@app.post()` endpoints for deterministic
  browser behavior; the GPU boundary stays in the decorated engine function.
  ```python
  from gradio import Server
  from fastapi.responses import HTMLResponse
  app = Server()

  @app.api(name="agent_turn", concurrency_limit=2)
  async def agent_turn(message: str):
      for token in run_agent_stream(message):   # generator → SSE
          yield token

  @app.get("/", response_class=HTMLResponse)     # custom UI replaces Gradio's default page
  async def home(): return open("index.html").read()
  app.launch()
  ```
- Frontend calls via `fetch("/api/agent-turn")`, parses newline-delimited JSON events, and updates the grimoire as
  `start` / `token` / `done` messages arrive. Notes and chapter exports use `/api/field-notes` and `/api/chapter`.
- **UI surfaces (the grimoire is the canvas):** streaming reply = ink writing itself (typewriter on already-streaming
  tokens); `search_projects`/overlap → **bleed** animation + page-number citations (real titles on hover);
  `find_whitespace` → **gold bloom** + sprouting leaf + a one-shaft light-mask ("the page chooses you");
  `score_idea` → **wax-seal** five-quadrant stamp; the riffling inked pages (fast page-flip of real Spaces) double as
  the project-wall; export = the torn-grimoire PNG artifact (§2). Jargon-correction toasts (§7) read as Mothback's
  margin notes; optimistic-UI loading + watchdog states (§8) are her "the page is choosing its words…". Cheap SFX:
  page-flip, quill scratch, wax-seal thunk.
- **Build the animation floor first:** safe typewriter + static stamp ships first (graceful degradation — the judges
  credited this); upgrade the ink-bleed / gold-bloom / seal-press last.
- **Fallback:** the backend (`tools.py`/`agent.py`) is UI-agnostic — if gr.Server misbehaves, fall back to
  `gr.Blocks` + `gr.HTML`, losing only the $1500 Off-Brand badge, never the submission.

---

## 12. Prize mapping

| Target | How it's earned |
|---|---|
| 🍄 Thousand Token Wood | **The Unwritten Almanac** (§2) — the bleed-citation wow IS the engine rendered; AI load-bearing; original |
| 🐜 Tiny Titan (special, $1.5k) | total ~1.98B, every model ≤4B; largest single = MiniCPM5 1.08B |
| 🔌 Off the Grid (badge) | all open weights run locally; offline index; no cloud inference at runtime |
| 🎯 Well-Tuned (badge) | published LoRA fine-tune of MiniCPM5 on the Hub (§10) → **6/6 badges** |
| 🎨 Off-Brand (badge + $1.5k) | `gr.Server` custom UI is the agent's output surface |
| 🏮 OpenBMB ($10k) | brain = MiniCPM5-1B ("OpenBMB pick") |
| 🟩 NVIDIA Quest (2× RTX 5080) | ASR = Nemotron (§5.1) |
| 🦙 Llama Champion (badge) | EmbeddingGemma GGUF retrieval index and runtime query embeddings run through llama.cpp (§5.5) |
| 📡 Sharing is Caring (badge) | publish the agent's tool-call trace to the Hub |
| 📓 Field Notes (badge) | this DESIGN.md → a build blog post |
| 🎖️ Bonus Quest Champion ($2k) | 6/6 badges (needs the Well-Tuned fine-tune) |
| 🤖 Best Agent ($1k) | real multi-tool loop: investigate → ideate → score → plan |
| 🟢 Modal ($20k credits) | offline crawl+embed + LoRA training on Modal (build-time, separated from runtime) |
| 🎬 Best Demo ($1k) | the mandatory demo video, made to sing (shared artifact + wow beat) |
| 🌀 OpenAI ($10k) | auto-entered ("across all submissions"); free lottery ticket, not a target |
| ❤️ Community Choice ($2k) | shareable tweetable artifact from the experience |

**6 badges** = Off the Grid, Well-Tuned, Off-Brand, Llama Champion, Sharing is Caring, Field Notes. Awards stack across
categories. Single-winner awards (Tiny Titan, Best Agent, Off-Brand, Best Demo) are eligibility ≠ win — the shared
lever is §11 custom-UI polish.

---

## 13. Risks / open items

1. **Deployment smoke tests are mandatory:** ZeroGPU Space build, same-origin NDJSON browser streaming, and Nemotron
   batch ASR in `@spaces.GPU` must be verified after every runtime dependency change.
2. **EmbeddingGemma is gated** — accept Gemma terms + `HF_TOKEN` before any crawl/build.
3. **MiniCPM5 tool-call reliability at 1B** — covered by the degradation ladder (§8); validate name+args in code.
4. **Concept skin** — **chosen: The Unwritten Almanac** (§2). Make-or-break is the bleed/bloom hero animation; build the
   safe typewriter + static-stamp floor first (graceful degradation), upgrade ink last. Watch the thin-org echo
   threshold + the dry-but-benevolent tone (real cited Spaces, never punch at a named builder).
5. **Param-budget claim** — document the 1.98B total in the README/Space card for Tiny Titan judging.

---

## 14. Build order

**Text-first vertical slice first; voice input is now part of the app.** Always keep a demoable artifact.

0. **Day-1 spikes** (§1) — get the three go/no-go builds green.
1. **`crawler.py` + Modal index** — crawl the org, embed with EmbeddingGemma, build the local index. *You immediately
   see what everyone's building and where the whitespace is.*
2. **`tools.py`** — research + ideation tools + the hardcoded `score_idea` rubric + the jargon alias layer, over the index.
3. **`agent.py`** — 3-layer context + single-hop loop + degradation ladder, MiniCPM5 via `transformers` (self-parsed XML).
4. **`app.py`** — `gr.Server` custom frontend (idea board, project/whitespace wall, streaming text), called via
   first-party `/api/...` endpoints; concept skin applied.
5. **Well-Tuned LoRA** — small fine-tune on Modal → publish to Hub (→ 6/6 badges).
6. **Voice input** — push-to-talk record and voice-note upload through Nemotron batch ASR in `/api/transcribe`.
7. **Polish + submission** — demo video + social post (Best Demo / Community Choice), publish agent trace (📡),
   write up Field Notes (📓).

**Deferred:** real-time streaming ASR and turn detection. The shipped path stays batch audio → transcript → editable idea.

---

## 15. Sources

**Models:** [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) ·
[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) · [MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) ·
[embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)

**Platforms:** [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) ·
[Introducing gradio.Server](https://huggingface.co/blog/introducing-gradio-server) · [Gradio Server Mode guide](https://www.gradio.app/guides/server-mode) ·
[Modal GPU](https://modal.com/docs/guide/gpu) · [Modal model weights](https://modal.com/docs/guide/model-weights) · [Modal pricing](https://modal.com/pricing) ·
[Build Small Hackathon](https://huggingface.co/build-small-hackathon)

*Verify-before-ship: Nemotron-in-ZeroGPU after dependency changes; MiniCPM5 license on the live card; llama.cpp MiniCPM5
tool-calling remains planned only and is not used by the deployed brain.*
