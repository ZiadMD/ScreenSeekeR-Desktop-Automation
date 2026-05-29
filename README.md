# ScreenSeekeR-Inspired Vision-Based Windows 11 Desktop Automation

A production-grade Python desktop automation system designed for Windows (1920×1080 at 110% DPI scaling). It uses a vision-based grounding system inspired by the ScreenSeekeR paper (arXiv:2504.07981) to zero-shot locate desktop UI elements using natural language, fetches blog posts from the JSONPlaceholder API, and automates modern tabbed Windows 11 Notepad to format and save them.

## 🚀 Key Features

- **ScreenSeekeR Cascaded Search**: Performs global planning to propose search regions, ranks them using **Gaussian Centrality Scoring ($\sigma=0.3$)** and **Non-Maximum Suppression (NMS)**, and crops search quadrants recursively to achieve high-precision icon grounding.
- **Provider-Agnostic Vision LLM Client**: Defaults to **Google Gemini** (`gemini-2.0-flash`) as the primary engine (the best-performing zero-shot GUI model on the ScreenSpot-Pro Grounding Leaderboard). Provides fully configurable interfaces to switch to **OpenAI** (GPT-4o), **Groq** (Llama 3.2 Vision), or local **Ollama** (llama3.2-vision) instances.
- **110% DPI Scaling Correction**: Compensates for Windows 11 high-DPI scaling. Captures physical screen size (2112×1188 pixels) and mathematically maps predictions back to PyAutoGUI's logical pixel space (1920×1080) for absolute click precision.
- **Modern Windows 11 Notepad Automation**: Specifically designed to handle Windows 11 tabbed Notepad window structures, custom Fluent file dialog navigation, and active tab management.
- **Vision Watchdog Dialog Detector**: Uses zero-shot vision checking to detect unexpected system prompts, notifications, or dialogs blocking the workspace and dismisses them automatically without template matching.
- **Resilient API Integration**: Fetches data from JSONPlaceholder with tenacious request-retry logic and robust offline local mock fallbacks.

---

## 🛠️ Installation & Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package and environment management.

### 1. Install Dependencies
Initialize the virtual environment and sync dependencies:
```bash
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
copy .env.example .env
```

Open `.env` and fill in your primary keys:
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
DPI_SCALING=1.10
```

---

## 🧪 Testing

The codebase includes a comprehensive unit and mathematical test suite checking coordinate mapping translations, NMS mathematical intersections (IoU), centrality ranking, and robust API failure fallbacks:

```bash
uv run pytest tests/ -v
```

---

## 🏃 Running the Application

To execute the end-to-end automation sequence:
```bash
uv run src/main.py
```

*Note: Ensure the Notepad shortcut icon is visible on your desktop and the desktop is not completely covered by other windows before launching.*

---

## 📂 Codebase Architecture

```
z:\Coding\TJM1\
├── pyproject.toml              # uv package configuration
├── .env.example                # Template for environment keys
├── .gitignore                  
├── README.md                   # Setup and usage guide
├── src/
│   ├── main.py                 # Core automation orchestrator
│   ├── config.py               # Pydantic configuration & constants
│   ├── api/
│   │   ├── posts.py            # API client with offline fallback
│   ├── grounding/
│   │   ├── llm_client.py       # Multi-provider client wrapper (Gemini, OpenAI, Groq, Ollama)
│   │   ├── planner.py          # LLM global region proposer
│   │   ├── grounder.py         # Bounded precision target locator
│   │   ├── scoring.py          # Centrality scoring and NMS filtering
│   │   ├── screenshot.py       # Capturing, DPI scaling, and drawing markers
│   │   └── screenseeker.py     # Cascaded visual search loop
│   └── automation/
│       ├── desktop.py          # PyAutoGUI mouse and keyboard adapters
│       ├── notepad.py          # Windows 11 Notepad workspace driver
│       └── popups.py           # Watchdog visual popup dismisser
└── tests/                      # PyTest automated tests
```
