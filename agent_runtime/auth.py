"""Shared-secret auth.

These services run an LLM with tool access on the host. An unauthenticated endpoint
is a remote shell, so the key is required rather than optional: if AGENT_API_KEY is
unset the app refuses to start.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

ENV_VAR = "AGENT_API_KEY"


def load_key() -> str:
    key = os.environ.get(ENV_VAR, "")
    if len(key) < 16:
        raise RuntimeError(
            f"{ENV_VAR} must be set to a secret of at least 16 characters. "
            "Generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    return key


def require_key(expected: str):
    async def dependency(x_api_key: str = Header(default="")) -> None:
        # Compare in constant time so the key cannot be recovered by timing.
        if not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key"
            )

    return dependency
