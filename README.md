# agent-runtime

The HTTP layer shared by my skill-plugin agents. It takes a Claude Agent SDK
configuration and serves it as an API that any UI can call: start a run, stream progress,
read the result.

It exists so the agents themselves stay small. Each one supplies an `AgentSpec` and gets
the whole surface below for free, which also means one client implementation works against
every agent built on it.

Used by:

- [content-skills](https://github.com/pronoy1004/content-skills) `agents/content-producer`
- [codebase-cartography](https://github.com/pronoy1004/codebase-cartography) `agents/codebase-cartographer`

## Install

```bash
pip install git+https://github.com/pronoy1004/agent-runtime
```

The Agent SDK shells out to the Claude Code CLI, so that needs to be on `PATH` and
authenticated (`ANTHROPIC_API_KEY`, or a profile from `claude` itself).

## The API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness. The only route without auth. |
| `GET` | `/agent` | Name, description, and the JSON schema of the request body. A UI can build its own form from this. |
| `POST` | `/runs` | Start a run. Returns `202 {"run_id", "status"}` straight away. |
| `GET` | `/runs/{id}/events` | Server-sent events: progress while the run works, then `done`. |
| `GET` | `/runs/{id}` | Run state. Carries `result` once `status` is `done`. |
| `DELETE` | `/runs/{id}` | Interrupt a running run. |

Runs take minutes, so nothing blocks: you post, then either stream or poll.

### Events

Every agent emits the same four shapes, so progress rendering is written once:

```
event: progress
data: {"type":"init","skills":["content-skills:write-hook", "..."]}

event: progress
data: {"type":"tool_use","name":"Skill","summary":"content-skills:write-hook"}

event: progress
data: {"type":"text","text":"Picked the contrarian angle: it reframes ..."}

event: progress
data: {"type":"result","cost_usd":0.42,"num_turns":18}

event: done
data: {"run_id":"...","status":"done","result":{...}}
```

Events are buffered on the run and replayed from the start on every connection, so a client
that connects late or reconnects after a dropped stream does not lose anything. SSE has no
replay of its own and these runs are long enough that it matters.

### Client sketch

```python
import httpx

h = {"X-API-Key": KEY}
run = httpx.post(f"{BASE}/runs", json=payload, headers=h).json()

with httpx.stream("GET", f"{BASE}/runs/{run['run_id']}/events", headers=h) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:].strip())

result = httpx.get(f"{BASE}/runs/{run['run_id']}", headers=h).json()["result"]
```

## Auth

Set `AGENT_API_KEY` to a secret of at least 16 characters. The app refuses to start without
one, on purpose: these services run a model with tool access on the host, so an open
endpoint is a remote shell.

```bash
export AGENT_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Callers send it as `X-API-Key`. It is compared in constant time.

For browser clients, set `AGENT_CORS_ORIGINS` to a comma-separated origin list. CORS stays
off when it is unset.

## Writing an agent

Supply an `AgentSpec` and call `create_app`:

```python
from agent_runtime import AgentSpec, RunPlan, create_app, plugin_skills
from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel

class Input(BaseModel):
    topic: str

def build(payload: Input) -> RunPlan:
    return RunPlan(
        prompt=f"/my-plugin:do-the-thing {payload.topic}",
        options=ClaudeAgentOptions(
            plugins=[{"type": "local", "path": PLUGIN_ROOT}],
            setting_sources=[],
            skills=plugin_skills(PLUGIN_ROOT),
            allowed_tools=["Skill"],
            permission_mode="dontAsk",
            model="claude-opus-5",
        ),
    )

def collect(outcome) -> dict:
    return outcome.structured_output or {"text": outcome.final_text}

app = create_app(AgentSpec(
    name="my-agent", description="...", input_model=Input,
    build=build, collect=collect,
))
```

Two settings carry most of the weight:

- `plugins=[{"type": "local", "path": ...}]` loads a skill repo exactly as it sits on disk.
  Nothing is copied or restructured, and the skills become `plugin-name:skill-name`.
- `setting_sources=[]` keeps the service out of the operator's personal `~/.claude`. Skills
  still load, because the `plugins` option discovers them on its own. Without this, a run
  would silently inherit whatever settings, skills, and MCP servers happen to be on the host.

`plugin_skills(path)` returns just that plugin's skills, so a run is not carrying Claude
Code's bundled skills in context alongside the pipeline it is meant to drive.

Use `output_format` for a structured result and read it back from
`outcome.structured_output`. It beats parsing prose out of the final message.

## Scope

Runs live in process memory and are evicted by age. A run is bound to a live SDK client in
one process, so it could not be served from another replica anyway. Run one instance per
agent, and put sticky routing in front before scaling out.

## Check

```bash
python3 test_runtime.py
```

Covers event replay and skill namespacing. The rest of the package is wiring that fails
loudly on import.
