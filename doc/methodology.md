# ScreenSeekeR — Technical Methodology & Architecture

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [The ScreenSeekeR Grounding Pipeline](#the-screenseeker-grounding-pipeline)
   - [Stage 1 — Screenshot Capture & DPI Handling](#stage-1--screenshot-capture--dpi-handling)
   - [Stage 2 — Planner (Global Region Proposal)](#stage-2--planner-global-region-proposal)
   - [Stage 3 — Scoring & Candidate Ranking](#stage-3--scoring--candidate-ranking)
   - [Stage 4 — Non-Maximum Suppression (NMS)](#stage-4--non-maximum-suppression-nms)
   - [Stage 5 — Grounder (Precision Localization)](#stage-5--grounder-precision-localization)
   - [Stage 6 — Confirmation / Refinement Step](#stage-6--confirmation--refinement-step)
   - [Stage 7 — Coordinate Mapping & Output](#stage-7--coordinate-mapping--output)
4. [Hybrid Mode: Cloud + Local Model](#hybrid-mode-cloud--local-model)
   - [GUI-Actor Local Model Architecture](#gui-actor-local-model-architecture)
   - [Attention-Based Pointer Network](#attention-based-pointer-network)
   - [Weight Remapping (transformers v4 → v5)](#weight-remapping-transformers-v4--v5)
5. [LLM Client — Provider Abstraction Layer](#llm-client--provider-abstraction-layer)
6. [Automation Layer](#automation-layer)
   - [Orchestrator Pipeline](#orchestrator-pipeline)
   - [Notepad Workflow](#notepad-workflow)
   - [Popup Watchdog](#popup-watchdog)
7. [API Integration](#api-integration)
8. [Configuration System](#configuration-system)
9. [Mathematical Foundations](#mathematical-foundations)
10. [File Reference Map](#file-reference-map)

---

## Overview

ScreenSeekeR is a **vision-based desktop automation system** that locates GUI elements using natural language descriptions — without template matching, accessibility APIs, or hard-coded pixel coordinates. It works by:

1. Taking a screenshot of the screen
2. Using a vision language model (VLM) to **plan** where the target element might be
3. **Scoring** and **filtering** candidate regions using Gaussian centrality and NMS
4. Using a VLM to **ground** (precisely locate) the element within cropped regions
5. **Refining** coordinates via a confirmation crop
6. Converting physical pixel coordinates to logical coordinates (accounting for DPI scaling)
7. Performing the click via PyAutoGUI

The system supports **hybrid mode**: using a cloud API (e.g. Gemini) for planning and a local model (GUI-Actor-3B) for grounding, enabling offline-capable precision localization.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph Entry["Entry Point"]
        MAIN["main.py<br/>Orchestrator"]
    end

    subgraph API["API Layer"]
        POSTS["PostClient<br/>JSONPlaceholder"]
    end

    subgraph Grounding["Grounding Engine"]
        SS["ScreenSeeker<br/>Cascaded Search"]
        PLAN["Planner<br/>Region Proposal"]
        SCORE["Scoring<br/>Rank & NMS"]
        GND["Grounder<br/>Precision Location"]
        SCR["Screenshot<br/>Capture & DPI"]
        LLM["LLMClient<br/>Provider Router"]
    end

    subgraph LocalModel["Local Model (Optional)"]
        LMC["LocalModelClient"]
        GUA["GUIActorAdapter"]
        QWEN["Qwen2.5-VL<br/>+ Pointer Head"]
    end

    subgraph Automation["Automation Layer"]
        NP["NotepadWorkflow"]
        PH["PopupHandler"]
        DT["Desktop Actions<br/>PyAutoGUI"]
    end

    subgraph Providers["Cloud Providers"]
        GEM["Google Gemini"]
        OAI["OpenAI GPT-4o"]
        GRQ["Groq"]
        OLL["Ollama"]
    end

    MAIN --> POSTS
    MAIN --> SS
    MAIN --> NP
    MAIN --> PH

    SS --> PLAN
    SS --> SCORE
    SS --> GND
    SS --> SCR

    PLAN --> LLM
    GND --> LLM
    PH --> LLM

    LLM --> GEM
    LLM --> OAI
    LLM --> GRQ
    LLM --> OLL
    LLM --> LMC

    LMC --> GUA
    GUA --> QWEN

    NP --> SS
    NP --> DT
    PH --> DT

    style SS fill:#1a73e8,color:#fff
    style QWEN fill:#7b1fa2,color:#fff
    style LLM fill:#0d9488,color:#fff
    style MAIN fill:#e65100,color:#fff
```

---

## The ScreenSeekeR Grounding Pipeline

The core grounding algorithm follows a **cascaded search** strategy inspired by the ScreenSeekeR paper (arXiv:2504.07981). Instead of asking a model to pinpoint a tiny element on a full-resolution screen in one shot, it breaks the problem into a coarse-to-fine hierarchy.

### End-to-End Pipeline Flow

```mermaid
flowchart TD
    START(["locate_element(instruction)"])
    CAP["Capture full screenshot<br/>(physical pixels)"]
    PLAN["Planner: propose<br/>candidate regions"]
    PLAN_FAIL{"Planner<br/>succeeded?"}
    FALLBACK["Local model fallback:<br/>ground full screenshot directly"]
    SCORE["Score candidates:<br/>Confidence × Gaussian Centrality"]
    NMS["Apply NMS<br/>(IoU threshold filtering)"]
    LOOP["For each candidate<br/>(ranked by score)"]
    CROP["Crop region from screenshot"]
    GROUND["Grounder: locate element<br/>within crop"]
    CONF_CHECK{"confidence ≥<br/>threshold?"}
    HIGH_CHECK{"confidence ≥<br/>0.85?"}
    NEXT["Next candidate"]
    CONFIRM{"Confirmation<br/>step enabled?"}
    REFINE["Refine: re-ground in<br/>tight 200×200 crop"]
    DPI["Physical → Logical<br/>coordinate conversion"]
    ANNOTATE["Annotate & save<br/>trace screenshot"]
    RESULT(["Return (x, y), confidence"])
    FAIL(["Return None, 0.0"])

    START --> CAP --> PLAN --> PLAN_FAIL
    PLAN_FAIL -->|Yes, has candidates| SCORE
    PLAN_FAIL -->|No candidates + local grounder| FALLBACK
    PLAN_FAIL -->|No candidates, no fallback| FAIL
    FALLBACK --> DPI
    SCORE --> NMS --> LOOP
    LOOP --> CROP --> GROUND --> CONF_CHECK
    CONF_CHECK -->|Below threshold| NEXT
    NEXT --> LOOP
    CONF_CHECK -->|Above threshold| HIGH_CHECK
    HIGH_CHECK -->|"≥ 0.85 — short-circuit"| CONFIRM
    HIGH_CHECK -->|"< 0.85 — keep searching"| NEXT
    CONFIRM -->|Yes| REFINE --> DPI
    CONFIRM -->|No| DPI
    DPI --> ANNOTATE --> RESULT
```

---

### Stage 1 — Screenshot Capture & DPI Handling

**File:** `src/grounding/screenshot.py`

The system captures the primary monitor using `mss`, which returns the screen in **physical pixels**. On a 1920×1080 display at 110% DPI scaling, the physical capture is 2112×1188 pixels.

All internal grounding operations work in physical pixel space. Only at the final output stage are coordinates converted to **logical pixels** for PyAutoGUI:

```
logical_x = physical_x / DPI_SCALING
logical_y = physical_y / DPI_SCALING
```

```mermaid
graph LR
    SCREEN["Physical Screen<br/>2112 × 1188 px"]
    MSS["mss.grab()"]
    PIL["PIL Image<br/>(physical pixels)"]
    GROUND["Grounding Pipeline<br/>(all coords in physical px)"]
    DPI["÷ DPI_SCALING<br/>(e.g. 1.10)"]
    PYAG["PyAutoGUI Click<br/>1920 × 1080 logical px"]

    SCREEN --> MSS --> PIL --> GROUND --> DPI --> PYAG

    style DPI fill:#f57c00,color:#fff
```

---

### Stage 2 — Planner (Global Region Proposal)

**File:** `src/grounding/planner.py`

The Planner receives the full screenshot and a natural language instruction (e.g., *"the Notepad icon shortcut on the desktop"*). It uses a vision LLM to analyze the entire screen and propose **1–4 candidate bounding boxes** where the target element is likely located.

Each candidate is a normalized bounding box `(x_min, y_min, x_max, y_max)` in the range `[0.0, 1.0]` with a confidence score and description.

**Why not just ask the model to click directly?** Small UI elements (icons, buttons) occupy a tiny fraction of a full-resolution screenshot. Vision models perform significantly better when the target element fills a larger portion of the input image. The Planner's job is to narrow the search area so the Grounder gets a close-up view.

```mermaid
sequenceDiagram
    participant SS as ScreenSeeker
    participant P as Planner
    participant LLM as Vision LLM

    SS->>P: propose_candidate_regions(screenshot, instruction)
    P->>LLM: call_vision_api(image, system_prompt, user_prompt)
    LLM-->>P: JSON response with candidates[]
    P->>P: Parse & validate bounding boxes
    P->>P: Clamp coordinates to [0, 1]
    P-->>SS: {candidates: [...], visual_clues: "..."}

    Note over P: If LLM fails → return fallback<br/>quadrants (left column, right column, center)
```

**Fallback Strategy:** If the Planner API call fails entirely (network error, rate limit), it returns three hardcoded search quadrants covering the most common desktop icon locations: left column, right column, and center region.

---

### Stage 3 — Scoring & Candidate Ranking

**File:** `src/grounding/scoring.py`

Each candidate region is scored using a **composite formula** that combines the Planner's confidence with a **Gaussian Centrality** penalty:

$$
\text{Score}(c) = \text{Confidence}(c) \times \exp\!\left(-\frac{d^2}{2\sigma^2}\right)
$$

Where:
- $d$ = Euclidean distance from the candidate's center to the expected reference point (default: screen center `(0.5, 0.5)`)
- $\sigma = 0.3$ (controls how sharply peripheral candidates are penalized)

This biases the search toward screen-center candidates when planner confidence is similar, which is empirically where users most often place targets.

```mermaid
graph TD
    subgraph Input
        C1["Candidate 1<br/>conf=0.8, center=(0.1, 0.5)"]
        C2["Candidate 2<br/>conf=0.7, center=(0.5, 0.5)"]
        C3["Candidate 3<br/>conf=0.6, center=(0.9, 0.1)"]
    end

    subgraph Scoring
        G1["Gaussian(0.1, 0.5) = 0.57"]
        G2["Gaussian(0.5, 0.5) = 1.00"]
        G3["Gaussian(0.9, 0.1) = 0.32"]
    end

    subgraph Results
        S1["Score = 0.8 × 0.57 = 0.456"]
        S2["Score = 0.7 × 1.00 = 0.700 | Top"]
        S3["Score = 0.6 × 0.32 = 0.192"]
    end

    C1 --> G1 --> S1
    C2 --> G2 --> S2
    C3 --> G3 --> S3

    style S2 fill:#2e7d32,color:#fff
```

---

### Stage 4 — Non-Maximum Suppression (NMS)

**File:** `src/grounding/scoring.py`

After scoring, candidates are passed through **Non-Maximum Suppression** to eliminate redundant overlapping regions. The algorithm:

1. Sort candidates by score (descending) — already done in Stage 3
2. Pick the top candidate → keep it
3. Remove all remaining candidates whose **IoU** (Intersection over Union) with the kept candidate exceeds the threshold (default: `0.3`)
4. Repeat until no candidates remain

```mermaid
flowchart LR
    IN["Scored candidates<br/>(sorted by score)"] --> PICK["Pick top<br/>candidate"]
    PICK --> KEEP["Add to<br/>keep list"]
    KEEP --> FILTER["Remove all remaining<br/>with IoU ≥ 0.3"]
    FILTER --> CHECK{"More<br/>candidates?"}
    CHECK -->|Yes| PICK
    CHECK -->|No| OUT["Filtered<br/>candidates"]

    style KEEP fill:#1b5e20,color:#fff
    style FILTER fill:#b71c1c,color:#fff
```

**IoU (Intersection over Union)** measures how much two boxes overlap:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

---

### Stage 5 — Grounder (Precision Localization)

**File:** `src/grounding/grounder.py`

For each surviving candidate region, the Grounder:

1. **Crops** the corresponding area from the full screenshot
2. Sends the crop + instruction to a vision model
3. Receives a **normalized center point** `(x, y)`, bounding box `(width, height)`, confidence, and reasoning
4. If confidence ≥ threshold (`CONFIDENCE_THRESHOLD`, default `0.4`), the result is considered valid

The first candidate to exceed **0.85 confidence** triggers a **short-circuit** — no further candidates are evaluated.

```mermaid
sequenceDiagram
    participant SS as ScreenSeeker
    participant G as Grounder
    participant LLM as Vision LLM
    participant MAP as map_relative_to_absolute()

    loop For each NMS-filtered candidate
        SS->>SS: Crop region from full screenshot
        SS->>G: ground_element(crop, instruction)
        G->>LLM: call_vision_api(crop_image, prompts)
        LLM-->>G: {x, y, width, height, confidence}
        G-->>SS: Grounding result

        alt confidence ≥ 0.85
            SS->>MAP: Convert crop-relative → absolute coords
            Note over SS: Short-circuit! Skip remaining candidates.
        else confidence ≥ threshold
            SS->>MAP: Convert crop-relative → absolute coords
            Note over SS: Track as best-so-far, continue searching.
        else confidence < threshold
            Note over SS: Reject this candidate, try next.
        end
    end
```

**Coordinate Mapping (`map_relative_to_absolute`):**

The Grounder returns coordinates relative to the crop (0–1). These must be mapped back to full-screenshot physical pixels:

```
abs_x = crop_x_min + (relative_x × crop_width)
abs_y = crop_y_min + (relative_y × crop_height)
```

---

### Stage 6 — Confirmation / Refinement Step

**File:** `src/grounding/screenseeker.py` (lines 186–223)

If `CONFIRMATION_STEP` is enabled (default: `true`), the system performs one final refinement:

1. Crop a tight **200×200 pixel** region centered on the best predicted point
2. Re-run the Grounder on this tiny, high-detail crop
3. If the refinement confidence ≥ 0.30, accept the refined coordinates

This narrows accuracy from "roughly correct region" to "exact click pixel."

```mermaid
graph LR
    BEST["Best grounding result<br/>from candidate search"]
    CROP200["Crop 200×200 px<br/>around predicted center"]
    REGROUND["Re-run Grounder<br/>on tight crop"]
    ACCEPT{"Refinement<br/>conf ≥ 0.30?"}
    REFINED["Use refined<br/>coordinates"]
    ORIGINAL["Keep original<br/>coordinates"]

    BEST --> CROP200 --> REGROUND --> ACCEPT
    ACCEPT -->|Yes| REFINED
    ACCEPT -->|No| ORIGINAL

    style REFINED fill:#1b5e20,color:#fff
```

---

### Stage 7 — Coordinate Mapping & Output

The final absolute physical coordinates are converted to logical coordinates for PyAutoGUI, and an annotated screenshot is saved showing:

- 🟡 **Yellow boxes** — all candidate search regions evaluated
- 🟢 **Green box** — the final predicted bounding box
- 🔴 **Red crosshair** — the exact click point
- **Label** — instruction text and confidence percentage

---

## Hybrid Mode: Cloud + Local Model

The system supports running **different providers for Planner and Grounder**. The most powerful configuration is:

| Role | Provider | Model | Purpose |
|------|----------|-------|---------|
| Planner | `gemini` | `gemini-2.0-flash` | Global scene understanding, region proposal |
| Grounder | `local` | `GUI-Actor-3B-Qwen2.5-VL` | Precise element localization |

```mermaid
graph TD
    subgraph Hybrid["Hybrid Mode"]
        direction TB
        SS["ScreenSeeker"]

        subgraph Cloud["Cloud (Planner)"]
            PLAN_C["Planner"]
            LLM_C["LLMClient<br/>provider=gemini"]
            GEMINI["Gemini API"]
        end

        subgraph Local["Local (Grounder)"]
            GND_L["Grounder"]
            LLM_L["LLMClient<br/>provider=local"]
            LMC["LocalModelClient"]
            ADAPTER["GUIActorAdapter"]
            MODEL["Qwen2.5-VL<br/>+ Pointer Head<br/>(GPU)"]
        end

        SS --> PLAN_C --> LLM_C --> GEMINI
        SS --> GND_L --> LLM_L --> LMC --> ADAPTER --> MODEL
    end

    subgraph Fallback["Fallback"]
        FALL["If Planner API fails →<br/>Local model grounds<br/>full screenshot directly"]
    end

    Hybrid -.-> Fallback

    style Cloud fill:#e3f2fd
    style Local fill:#f3e5f5
```

**Fallback behavior:** If the cloud Planner fails (API error, rate limit, network outage) and a local grounder is configured, the system bypasses the cascaded search entirely and uses the local model to ground the element on the full screenshot in a single pass.

---

### GUI-Actor Local Model Architecture

**Files:** `src/grounding/local_model/`

GUI-Actor is a **3B-parameter vision-language model** based on Qwen2.5-VL, fine-tuned by Microsoft for GUI element grounding. Unlike API-based VLMs that return text coordinates, GUI-Actor uses an **attention-based pointer network** that directly attends to visual patches.

```mermaid
graph TD
    subgraph Input
        IMG["Screenshot"]
        INST["Instruction:<br/>'Click on Notepad icon'"]
    end

    subgraph Preprocessing
        RESIZE["Resize to max_pixels<br/>(3200×1800)"]
        CONV["Build conversation:<br/>system + user message"]
        TEMPLATE["Apply chat template<br/>(Qwen2.5-VL format)"]
        PROC["AutoProcessor<br/>tokenize + image encode"]
    end

    subgraph Model["Qwen2.5-VL + Pointer Head"]
        EMBED["Token Embedding<br/>+ Image Patch Embedding"]
        XFORMER["36-layer Transformer<br/>(with RoPE)"]
        DECODE["Token Generation<br/>(with pointer tokens)"]
        POINTER["VisionHead_MultiPatch<br/>(attention pointer network)"]
    end

    subgraph Output
        ATTN["Attention scores<br/>over image patches"]
        BFS["BFS region clustering<br/>+ weighted center"]
        COORD["Normalized (x, y)<br/>coordinates"]
    end

    IMG --> RESIZE --> CONV
    INST --> CONV
    CONV --> TEMPLATE --> PROC --> EMBED
    EMBED --> XFORMER --> DECODE
    DECODE --> POINTER
    POINTER --> ATTN --> BFS --> COORD

    style POINTER fill:#7b1fa2,color:#fff
    style BFS fill:#1565c0,color:#fff
```

---

### Attention-Based Pointer Network

**File:** `src/grounding/local_model/modeling_qwen25vl.py` — `VisionHead_MultiPatch`

Instead of predicting coordinates as text tokens, GUI-Actor uses a **pointer network** that computes attention scores between:

- **Encoder features** — hidden states of image patch tokens (from the vision encoder)
- **Decoder features** — hidden states of special `<|pointer_pad|>` tokens generated during decoding

The attention score for each image patch represents the model's belief that the target element is located at that patch.

```mermaid
flowchart TD
    subgraph Encoder["Image Patch Embeddings"]
        E1["Patch (0,0)"]
        E2["Patch (0,1)"]
        E3["Patch (1,0)"]
        EN["..."]
        E4["Patch (H,W)"]
    end

    subgraph SelfAttn["Self-Attention Layer"]
        SA["Multi-Head Self-Attention<br/>+ LayerNorm + Residual"]
    end

    subgraph Projections["Projection Networks"]
        PROJ_E["projection_enc<br/>Linear → GELU → Linear"]
        PROJ_D["projection_dec<br/>Linear → GELU → Linear"]
    end

    subgraph Decoder["Pointer Token Hidden States"]
        D1["pointer_pad<br/>hidden state"]
    end

    subgraph Score["Scaled Dot-Product"]
        DOT["proj_dec · proj_enc^T<br/>÷ √d_model"]
        SOFT["Softmax"]
        HEAT["Attention heatmap<br/>over patch grid"]
    end

    E1 & E2 & E3 & EN & E4 --> SA --> PROJ_E
    D1 --> PROJ_D
    PROJ_E --> DOT
    PROJ_D --> DOT
    DOT --> SOFT --> HEAT

    style HEAT fill:#d32f2f,color:#fff
    style SA fill:#1565c0,color:#fff
```

**From Heatmap to Coordinates:**

1. **Threshold** — Select patches with attention score > 30% of the maximum
2. **BFS Clustering** — Group connected activated patches into regions
3. **Rank Regions** — Sort by average attention score
4. **Weighted Center** — Compute the attention-weighted center of the top region
5. **Normalize** — Convert grid coordinates to `(0, 1)` range

```mermaid
flowchart LR
    HEAT["Attention<br/>heatmap"]
    THRESH["Threshold<br/>> 0.3 × max"]
    BFS["BFS cluster<br/>connected patches"]
    RANK["Sort regions<br/>by avg score"]
    CENTER["Weighted center<br/>of top region"]
    NORM["Normalize<br/>to (0,1)"]

    HEAT --> THRESH --> BFS --> RANK --> CENTER --> NORM
```

---

### Weight Remapping (transformers v4 → v5)

**File:** `src/grounding/local_model/modeling_qwen25vl.py` — `from_pretrained()`

The GUI-Actor checkpoint was saved with `transformers v4.x`, which used flat key names. When loaded with `transformers v5.x`, the model expects a nested `language_model` prefix:

| Checkpoint Key (v4) | Expected Key (v5) |
|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `model.language_model.layers.0.self_attn.q_proj.weight` |
| `model.embed_tokens.weight` | `model.language_model.embed_tokens.weight` |
| `model.norm.weight` | `model.language_model.norm.weight` |
| `visual.blocks.0.*` | `model.visual.blocks.0.*` |

The custom `from_pretrained()` override:

1. Detects if the checkpoint uses old-format keys (by checking if any key starts with `model.layers.`)
2. Loads all safetensor shards and remaps every key
3. Loads the model architecture with `super().from_pretrained()`
4. Overwrites with the correctly-remapped state dict via `load_state_dict()`
5. Re-ties `lm_head.weight` to `embed_tokens.weight` (since `tie_word_embeddings=true`)

```mermaid
flowchart TD
    START(["from_pretrained(path)"])
    PEEK["Peek at first shard's keys"]
    CHECK{"Keys start with<br/>'model.layers.'?"}

    subgraph Remap["Remap Path"]
        LOAD_SHARDS["Load all .safetensors shards"]
        REMAP["Remap keys:<br/>model.layers.* → model.language_model.layers.*<br/>visual.* → model.visual.*"]
        LOAD_ARCH["Load model architecture<br/>(super().from_pretrained)"]
        APPLY["load_state_dict(remapped)"]
        TIE["tie_weights()<br/>(lm_head ↔ embed_tokens)"]
    end

    subgraph Standard["Standard Path"]
        STD_LOAD["super().from_pretrained()"]
    end

    DONE(["Return model"])

    START --> PEEK --> CHECK
    CHECK -->|Yes — old format| LOAD_SHARDS
    LOAD_SHARDS --> REMAP --> LOAD_ARCH --> APPLY --> TIE --> DONE
    CHECK -->|No — already v5| STD_LOAD --> DONE

    style Remap fill:#fff3e0
```

---

## LLM Client — Provider Abstraction Layer

**File:** `src/grounding/llm_client.py`

The `LLMClient` provides a **unified interface** for calling vision models across five providers. Every provider implements the same `call_vision_api(image, system_prompt, user_prompt)` method.

```mermaid
graph TD
    CALLER["Planner / Grounder / PopupHandler"]
    LLM["LLMClient<br/>call_vision_api()"]

    subgraph Providers
        GEM["Gemini<br/>google.generativeai"]
        OAI["OpenAI<br/>GPT-4o / base64 JPEG"]
        GRQ["Groq<br/>Llama 3.2 Vision"]
        OLL["Ollama<br/>Local REST API"]
        LOC["LocalModelClient<br/>→ GUIActorAdapter"]
    end

    CALLER --> LLM
    LLM -->|provider=gemini| GEM
    LLM -->|provider=openai| OAI
    LLM -->|provider=groq| GRQ
    LLM -->|provider=ollama| OLL
    LLM -->|provider=local| LOC

    style LLM fill:#0d9488,color:#fff
```

| Provider | Image Format | Auth | JSON Mode |
|----------|-------------|------|-----------|
| Gemini | PIL Image (native) | `GEMINI_API_KEY` | `response_mime_type` |
| OpenAI | Base64 JPEG | `OPENAI_API_KEY` | `response_format` |
| Groq | Base64 JPEG | `GROQ_API_KEY` | Via prompt instruction |
| Ollama | Raw JPEG bytes | Local (no key) | `format="json"` |
| Local | PIL Image (native) | None | N/A (structured output) |

---

## Automation Layer

### Orchestrator Pipeline

**File:** `src/main.py`

The main orchestrator coordinates the full end-to-end workflow:

```mermaid
flowchart TD
    START(["Start Pipeline"])
    FETCH["Fetch 10 posts from<br/>JSONPlaceholder API"]
    INIT["Initialize ScreenSeeker,<br/>NotepadWorkflow, PopupHandler"]
    CLEAN["Close any existing<br/>Notepad windows"]

    subgraph Loop["For each of 10 posts"]
        POPUP["Check & dismiss<br/>unexpected popups"]
        LAUNCH["Locate Notepad icon<br/>via ScreenSeeker"]
        DCLICK["Double-click to launch"]
        WAIT["Wait for Notepad<br/>window to appear"]
        TYPE["Type formatted post<br/>content"]
        SAVE["Ctrl+S → type path<br/>→ Enter"]
        VERIFY["Verify file exists<br/>on disk"]
        CLOSE["Close Notepad window"]
    end

    SUMMARY(["Print success/failure summary"])

    START --> FETCH --> INIT --> CLEAN --> Loop
    Loop --> POPUP --> LAUNCH --> DCLICK --> WAIT
    WAIT --> TYPE --> SAVE --> VERIFY --> CLOSE
    CLOSE -->|Next post| POPUP
    CLOSE -->|All done| SUMMARY

    style LAUNCH fill:#1a73e8,color:#fff
    style SAVE fill:#2e7d32,color:#fff
```

---

### Notepad Workflow

**File:** `src/automation/notepad.py`

The `NotepadWorkflow` class handles Windows 11's modern tabbed Notepad:

```mermaid
stateDiagram-v2
    [*] --> ShowDesktop: Win+D
    ShowDesktop --> LocateIcon: ScreenSeeker.locate_element()
    LocateIcon --> DoubleClick: click_at(x, y, double=True)
    DoubleClick --> WaitWindow: Poll for "Notepad" window title
    WaitWindow --> NewTab: Ctrl+N
    NewTab --> TypeContent: pyautogui.write(text)
    TypeContent --> SaveDialog: Ctrl+S
    SaveDialog --> TypePath: pyautogui.write(filepath)
    TypePath --> ConfirmSave: Enter
    ConfirmSave --> VerifyFile: Check file exists on disk
    VerifyFile --> CloseTab: Ctrl+W
    CloseTab --> [*]

    WaitWindow --> SubprocessFallback: Timeout after 5s
    SubprocessFallback --> WaitWindow: notepad.exe via subprocess

    note right of LocateIcon
        Up to 3 attempts before
        falling back to subprocess
    end note
```

---

### Popup Watchdog

**File:** `src/automation/popups.py`

The `PopupHandler` is a **zero-shot dialog detector** that runs before each automation step. It:

1. Takes a screenshot
2. Asks a vision model: *"Are there any blocking popups or dialogs?"*
3. If detected, locates the dismiss button (Close, Cancel, X, etc.)
4. Clicks it to clear the workspace

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant PH as PopupHandler
    participant LLM as Vision LLM
    participant D as Desktop

    O->>PH: check_and_dismiss_popups()
    PH->>PH: capture_screen()
    PH->>LLM: "Any blocking popups?"
    alt Popup detected
        LLM-->>PH: {popup_detected: true, coords: {x, y}}
        PH->>PH: physical_to_logical(x, y)
        PH->>D: click_at(logical_x, logical_y)
        PH-->>O: true (popup dismissed)
    else No popup
        LLM-->>PH: {popup_detected: false}
        PH-->>O: false (workspace clear)
    end
```

---

## API Integration

**File:** `src/api/posts.py`

The `PostClient` fetches blog posts from the JSONPlaceholder REST API with built-in resilience:

```mermaid
flowchart TD
    CALL["fetch_first_10_posts()"]
    RETRY["@robust_retry<br/>(3 attempts, 1s delay)"]
    REQ["GET /posts<br/>timeout=5s"]
    PARSE["Parse first 10 as<br/>Pydantic Post models"]
    SUCCESS(["Return List of Post"])

    FAIL["All 3 attempts failed"]
    MOCK["Generate 10 mock posts<br/>as graceful fallback"]
    FALLBACK(["Return fallback posts"])

    CALL --> RETRY --> REQ --> PARSE --> SUCCESS
    RETRY -->|Exception after retries| FAIL --> MOCK --> FALLBACK

    style MOCK fill:#f57c00,color:#fff
    style SUCCESS fill:#2e7d32,color:#fff
```

Each `Post` formats its content as:
```
Title: {title}

{body}
```

---

## Configuration System

**File:** `src/config.py`

All settings are managed via a Pydantic `Settings` class that loads from `.env`:

```mermaid
graph LR
    ENV[".env file"]
    PYDANTIC["Pydantic BaseSettings"]
    SETTINGS["settings singleton"]

    subgraph Categories
        PROVIDER["Provider Config<br/>LLM_PROVIDER, API keys"]
        MODEL["Model Config<br/>PLANNER_MODEL, GROUNDER_MODEL"]
        LOCAL["Local Model Config<br/>MODEL_PATH, DEVICE, DTYPE"]
        DISPLAY["Display Config<br/>DPI_SCALING"]
        SEARCH["Search Config<br/>CONFIDENCE_THRESHOLD,<br/>IoU_THRESHOLD, etc."]
    end

    ENV --> PYDANTIC --> SETTINGS
    SETTINGS --> PROVIDER
    SETTINGS --> MODEL
    SETTINGS --> LOCAL
    SETTINGS --> DISPLAY
    SETTINGS --> SEARCH
```

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Primary vision LLM provider |
| `PLANNER_PROVIDER` | `None` (= `LLM_PROVIDER`) | Override for planner only |
| `GROUNDER_PROVIDER` | `None` (= `LLM_PROVIDER`) | Override for grounder only |
| `DPI_SCALING` | `1.00` | Windows display scale factor |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum grounding confidence |
| `IoU_THRESHOLD` | `0.3` | NMS overlap threshold |
| `CONFIRMATION_STEP` | `true` | Enable refinement re-grounding |
| `LOCAL_MODEL_PATH` | `None` | Relative path under `models/` |
| `LOCAL_DEVICE` | `cuda:0` | GPU device for local model |
| `LOCAL_TORCH_DTYPE` | `float16` | Model precision |

---

## Mathematical Foundations

### Gaussian Centrality

Scores candidates by their proximity to an expected reference point:

$$
G(c) = \exp\!\left(-\frac{\|c - r\|^2}{2\sigma^2}\right)
$$

- $c$ = candidate center coordinates
- $r$ = reference point (default screen center)
- $\sigma = 0.3$

### Intersection over Union (IoU)

Measures bounding box overlap for NMS:

$$
\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}
$$

### Composite Candidate Score

$$
\text{Score}(c) = \text{Confidence}_{\text{planner}}(c) \times G(c)
$$

### DPI Coordinate Mapping

$$
x_{\text{logical}} = \left\lfloor \frac{x_{\text{physical}}}{\text{DPI\_SCALING}} \right\rceil, \quad
y_{\text{logical}} = \left\lfloor \frac{y_{\text{physical}}}{\text{DPI\_SCALING}} \right\rceil
$$

### Crop-Relative to Absolute Mapping

$$
x_{\text{abs}} = x_{\min}^{\text{crop}} + x_{\text{rel}} \cdot w_{\text{crop}}, \quad
y_{\text{abs}} = y_{\min}^{\text{crop}} + y_{\text{rel}} \cdot h_{\text{crop}}
$$

---

## File Reference Map

| File | Module | Purpose |
|------|--------|---------|
| `src/main.py` | Entry | Orchestrates the full automation pipeline |
| `src/config.py` | Config | Pydantic settings from `.env` |
| `src/api/posts.py` | API | JSONPlaceholder client with retry + fallback |
| `src/grounding/screenseeker.py` | **Core** | Cascaded visual search orchestrator |
| `src/grounding/planner.py` | Grounding | Global region proposal via VLM |
| `src/grounding/grounder.py` | Grounding | Precision element localization in crops |
| `src/grounding/scoring.py` | Grounding | Gaussian centrality scoring + NMS |
| `src/grounding/screenshot.py` | Grounding | Screen capture, DPI mapping, annotations |
| `src/grounding/llm_client.py` | Grounding | Multi-provider VLM abstraction |
| `src/grounding/local_model/client.py` | Local | LLMClient-compatible wrapper |
| `src/grounding/local_model/gui_actor_adapter.py` | Local | GUI-Actor inference + coordinate extraction |
| `src/grounding/local_model/modeling_qwen25vl.py` | Local | Custom Qwen2.5-VL with pointer head |
| `src/grounding/local_model/_inference_utils.py` | Local | Forced token generation for pointer sequence |
| `src/automation/desktop.py` | Automation | PyAutoGUI mouse/keyboard primitives |
| `src/automation/notepad.py` | Automation | Windows 11 Notepad workflow driver |
| `src/automation/popups.py` | Automation | Zero-shot popup detection & dismissal |
| `src/utils/retry.py` | Utils | Tenacity-based retry decorator |
| `src/utils/logging.py` | Utils | Loguru logger configuration |
