"""Turn an AgentSpec into an HTTP service.

Both agents expose the same surface, so one client implementation drives either:

    GET    /agent              descriptor + JSON schema of the request body
    POST   /runs               start a run, returns 202 {run_id, status}
    GET    /runs/{id}/events   SSE progress, replayed from the start
    GET    /runs/{id}          run state, includes result once status is "done"
    DELETE /runs/{id}          cancel a running run
    GET    /healthz            liveness, the only unauthenticated route
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from .auth import load_key, require_key
from .registry import Registry
from .runner import execute
from .spec import AgentSpec

log = logging.getLogger(__name__)


def create_app(spec: AgentSpec) -> FastAPI:
    guard = Depends(require_key(load_key()))
    app = FastAPI(title=spec.name, description=spec.description)
    app.state.registry = Registry()
    # Agent-specific routes must guard themselves with this, not invent their own.
    app.state.guard = guard
    app.state.tasks = set()

    origins = [o for o in os.environ.get("AGENT_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "content-type"],
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "agent": spec.name}

    @app.get("/agent", dependencies=[guard])
    async def descriptor() -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_model.model_json_schema(),
        }

    @app.post("/runs", status_code=202, dependencies=[guard])
    async def start(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "request body is not valid JSON") from exc
        try:
            payload = spec.input_model.model_validate(body)
        except ValidationError as exc:
            # Must be caught before the ValueError below: ValidationError subclasses it,
            # and a schema violation is a 422, not a 400.
            raise HTTPException(422, exc.errors(include_url=False)) from exc
        try:
            # build() does the work that can fail on the caller's input: cloning a repo,
            # resolving a path. That is a bad request, not a server fault.
            plan = spec.build(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        run = app.state.registry.create()
        task = asyncio.create_task(execute(spec, plan, run))
        run._task = task
        # Hold a reference; a bare create_task can be garbage collected mid-run.
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return JSONResponse({"run_id": run.id, "status": run.status}, status_code=202)

    @app.get("/runs/{run_id}", dependencies=[guard])
    async def get_run(run_id: str) -> dict[str, Any]:
        run = app.state.registry.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run.public()

    @app.get("/runs/{run_id}/events", dependencies=[guard])
    async def events(run_id: str, request: Request) -> EventSourceResponse:
        run = app.state.registry.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")

        async def publish():
            async for event in run.stream():
                if await request.is_disconnected():
                    return
                yield {"event": "progress", "data": json.dumps(event)}
            yield {"event": "done", "data": json.dumps(run.public())}

        return EventSourceResponse(publish())

    @app.delete("/runs/{run_id}", dependencies=[guard])
    async def cancel(run_id: str) -> dict[str, Any]:
        run = app.state.registry.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run.status != "running":
            # Already finished. Cancelling now would throw away its result.
            return run.public()
        task = run._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=10)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass  # expected: cancellation raises, or cleanup itself failed
        # A run that finished (or errored) in the race with this cancel keeps its own
        # result; only stamp "cancelled" if it is still sitting in "running".
        if run.status == "running":
            run.finish("cancelled", error="cancelled by client")
        return run.public()

    if spec.extra_routes is not None:
        spec.extra_routes(app)

    return app
