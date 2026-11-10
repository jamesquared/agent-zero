# OPERATIONS

Workspace log for the `jamesquared/agent-zero` checkout. Times below are America/New_York.

## Snapshot

- Recorded: Monday, Nov 9, 2026, 7:00 PM ET
- Branch at time of this write: `cursor/ember-operator-role-4ebe` (cut from `main`)
- `main` was even with `origin/main` at `fa65fa3` (`Merge branch 'testing'`)
- Working tree before this change was clean
- No project virtualenv (`.venv` or `.conda`) is present
- Flask, uvicorn, and python-dotenv are not installed on the system interpreter, so `run_ui.py` was not started

## Project structure

Agent Zero is a Python agent runtime plus a browser UI. The process entry for local sessions is `run_ui.py`.

| Path | Role |
| --- | --- |
| `agent.py` | Agent loop, history, and prompt loading |
| `initialize.py` | Chat, MCP, job-loop, and preload startup |
| `models.py` | Model provider configuration |
| `prepare.py` | Container prep. Sets a root password when `KEY_ROOT_PASSWORD` is empty |
| `run_ui.py` | Web UI process: Flask, Socket.IO, Uvicorn |
| `prompts/` | Default prompt templates. `agent.system.main.md` includes the role, environment, communication, solving, and tips files |
| `prompts/default/` | Named Ember role file. Not on the prompt search path by itself |
| `agents/` | Profiles: `agent0`, `default`, `developer`, `hacker`, `researcher`, `_example` |
| `agents/agent0/prompts/` | Overrides for the top-level web UI agent. This is the file local UI sessions load |
| `python/api/` | HTTP API handlers |
| `python/tools/` | Built-in tools, including `call_subordinate.py` |
| `python/extensions/` | Hook extensions. `system_prompt/_10_system_prompt.py` loads `agent.system.main.md` |
| `python/helpers/` | Runtime, settings, files, prompt resolution |
| `webui/` | Browser UI assets served by Flask |
| `docker/` | Base and run images. Container entry is `docker/run/fs/exe/run_A0.sh` |
| `conf/` | Model provider defaults |
| `docs/` | Setup, usage, and architecture docs |
| `knowledge/` | Knowledge base files |
| `skills/` | SKILL.md packages |
| `tests/` | Pytest suite, mostly websockets, auth, and state sync |
| `usr/` | Runtime data (settings, secrets, chats, projects). Gitignored except `.gitkeep` |
| `logs/`, `tmp/` | Runtime output. Gitignored except `.gitkeep` |
| `requirements.txt` | Main Python dependencies |
| `requirements2.txt` | `litellm==1.79.3`, `openai==1.99.5` |
| `requirements.dev.txt` | `pytest`, `pytest-asyncio`, `pytest-mock` |

Prompt search order for a profile, from `python/helpers/subagents.py` `get_paths`, is project agent, project meta, `usr/agents/<profile>/`, `agents/<profile>/`, `usr/`, then `prompts/`. The first existing file wins. `prompts/default/` is not one of those directories.

## Python environment

- Interpreter: CPython 3.12.3 at `/usr/bin/python3` (`GCC 13.3.0`, built Aug 31, 2026)
- No activated virtualenv in this checkout
- Declared runtime: `requirements.txt` (Flask 3.0.3, uvicorn, python-socketio, Playwright, browser-use, MCP, LangChain, FAISS, and the rest of that file)
- Extra pins: `requirements2.txt`
- Test pins: `requirements.dev.txt`
- Probe on Nov 9, 2026: `import flask` failed with `ModuleNotFoundError`. `pip show` did not find `flask`, `uvicorn`, or `python-dotenv`

Install before a host launch, from the repo root, with a virtualenv of your choice:

```bash
pip install -r requirements.txt
pip install -r requirements2.txt
playwright install chromium
```

Dev setup notes in `docs/setup/dev-setup.md` use Python 3.12 and the same `requirements.txt` install. Docker remains the supported full runtime. Host Python is the development loop and needs the Docker RFC target for code execution.

## How `run_ui.py` is launched

### Host

From the repo root, after dependencies are installed:

```bash
python run_ui.py
```

`if __name__ == "__main__"` calls `runtime.initialize()`, `dotenv.load_dotenv()`, then `run()`.

`run()` migrates stored data, then binds Uvicorn:

- Port: CLI `--port`, else `WEB_UI_PORT`, else `5000` (`python/helpers/runtime.py` `get_web_ui_port`)
- Host: CLI `--host`, else `WEB_UI_HOST`, else `localhost`

The process builds a Flask app for the UI and API, a Socket.IO ASGI server, and a Starlette app that mounts `/mcp`, `/a2a`, and the Flask app at `/`. Uvicorn serves that stack with `wsproto`. A daemon thread polls `http://{host}:{port}/health` until it returns 200, then prints `Agent Zero is running.` `init_a0()` initializes chats, MCP, the job loop, and preload.

IDE debug config in `.vscode/launch.json` (`Debug run_ui.py`) launches `./run_ui.py` with args `--development=true` and `-Xfrozen_modules=off`.

### Docker

`docker/run/fs/exe/run_A0.sh` prepares the container environment, then replaces the shell with:

```bash
python /a0/prepare.py --dockerized=true
python /a0/run_ui.py --dockerized=true --port=80 --host=0.0.0.0
```

The published quick start maps that port 80 out as 50001:

```bash
docker pull agent0ai/agent-zero
docker run -p 50001:80 agent0ai/agent-zero
```

`DockerfileLocal` exposes ports 22, 80, and 9000-9009 and starts through `/exe/initialize.sh`.

## Decision: Ember role for web UI sessions

Local web UI sessions use profile `agent0` (`python/helpers/settings.py` defaults `agent_profile` to `agent0`). That profile already has `agents/agent0/prompts/agent.system.main.role.md`, and profile files win over `prompts/`.

Writing only `prompts/default/agent.system.main.role.md` would leave the UI on the old Agent 0 role. Both of these files now carry the same Ember operator description:

- `prompts/default/agent.system.main.role.md` (requested path)
- `agents/agent0/prompts/agent.system.main.role.md` (file the loader opens for web UI sessions)

Specialist profiles (`developer`, `hacker`, `researcher`) keep their own role files. The shared fallback `prompts/agent.system.main.role.md` is unchanged.

The Ember text does not tell the agent to ignore safety limits. The previous Agent 0 line that said to obey instructions and never refuse for safety or ethics was removed with this replacement.

## Command log

| When (ET) | Command | Result |
| --- | --- | --- |
| Nov 9, 2026, ~6:59 PM | `git status -sb`, `git log -5`, `git remote -v`, `ls -la` | Clean `main` at `fa65fa3`, remote `jamesquared/agent-zero` |
| Nov 9, 2026, ~7:00 PM | `python3 --version`; import probe for Flask, uvicorn, python-dotenv | Python 3.12.3; those packages are absent |
| Nov 9, 2026, ~7:00 PM | `git checkout -b cursor/ember-operator-role-4ebe` | Branch created from `main` |
| Nov 9, 2026, ~7:01 PM | Compare the two role files and walk the `agent0` search order | Files match. Loader selects `agents/agent0/prompts/agent.system.main.role.md`. That copy names Ember and Billymon and does not contain "never refuse" |
