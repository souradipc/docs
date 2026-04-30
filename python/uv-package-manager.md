# ⚡ UV — The Modern Python Package Manager

> UV is a **single tool** that replaces `pip`, `pipx`, and `venv`. It's blazing fast and handles everything from virtual environments to Python version management to publishing packages on PyPI.

---

## 🔁 Daily Workflow

| Command | What it does |
|---|---|
| `uv run main.py` | Runs your script — works even if the virtual env is deleted |
| `uv sync` | Recreates the virtual env from the `.lock` file |
| `uv tree` | Shows the full dependency tree of your project |

**`uv run`** is your go-to for day-to-day execution. UV will auto-resolve dependencies on the fly, so you never have to worry about a broken or missing virtual env.

---

## 🧰 Tool Management (replaces `pipx`)

UV lets you install CLI tools like `ruff` in **isolated environments**, so they're available globally without polluting any project.

| Command | What it does |
|---|---|
| `uv tool install ruff` | Installs `ruff` globally in an isolated environment |
| `uv tool run ruff check` | Runs `ruff` without even installing it first |
| `uv tool upgrade --all` | Upgrades all installed tools at once |
| `uv tool list` | Lists all globally installed tools |

> 💡 `uv tool run` is great for one-off usage — no install needed at all.

---

## 🐍 What Else UV Can Do

UV goes beyond just package management. It's a full Python project toolkit.

---

### 🔢 Multiple Python Versions

No more `pyenv`. UV can install and manage Python versions directly, and pin a specific version per project.

```bash
uv python install 3.12          # Install a specific Python version
uv python list                  # See all installed & available versions
uv python pin 3.11              # Pin this project to Python 3.11 (writes .python-version)
uv venv --python 3.10           # Create a venv with a specific version
```

> UV stores Python versions locally and switches between them automatically based on the `.python-version` file — no shell hooks or manual activation needed.

---

### 📦 Build & Publish Packages to PyPI

UV has first-class support for building and publishing Python packages, replacing tools like `build` and `twine`.

```bash
uv build                        # Build wheel + sdist into /dist
uv publish                      # Upload to PyPI (uses UV_PUBLISH_TOKEN or prompts)
uv publish --index testpypi     # Publish to TestPyPI first (recommended for testing)
```

A typical release flow looks like:

```bash
# 1. Bump your version in pyproject.toml
# 2. Build the distribution
uv build

# 3. Test it on TestPyPI first
uv publish --index testpypi

# 4. Ship it to the real PyPI
uv publish
```

> 💡 UV reads your package metadata from `pyproject.toml` — no separate `setup.py` or `MANIFEST.in` needed.

---

### 🐳 Optimized Docker Containers

UV dramatically speeds up Docker builds by caching dependencies efficiently. The key trick is to **separate dependency installation from copying your app code**, so Docker's layer cache kicks in on rebuilds.

```dockerfile
FROM python:3.12-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency files first — this layer gets cached
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Now copy the rest of your app
COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "main.py"]
```

**Why this is faster:**
- `--frozen` ensures the lock file is used exactly as-is — no re-resolution
- Dependency layer is cached by Docker unless `pyproject.toml` or `uv.lock` changes
- `uv sync` is significantly faster than `pip install` for large dependency trees

> 💡 You can also use `uv run` as the container entrypoint — it handles the environment automatically without needing to activate a venv manually.

---

## 🆚 What UV Replaces

| Old Tool | UV Equivalent |
|---|---|
| `pip` | `uv add` / `uv sync` |
| `venv` | handled automatically |
| `pipx` | `uv tool install` |
| `pyenv` | `uv python install` |

---

> 📦 Install UV: `curl -Lsf https://astral.sh/uv/install.sh | sh`  
> 📖 Docs: [docs.astral.sh/uv](https://docs.astral.sh/uv)
