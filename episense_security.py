"""Shared security configuration and ASGI middleware for Episense.

This module is imported by both the API and the dashboard. It deliberately
depends only on the standard library, python-dotenv and Starlette so it can be
copied into both runtime images without pulling the project's ML stack.

Environment variables
---------------------
ENVIRONMENT / EPISENSE_ENVIRONMENT
    ``production`` (default when neither is set) or ``development``.
    In production a strong API key and dashboard password are mandatory.
EPISENSE_API_KEY
    API token accepted via ``X-API-Key`` or ``Authorization: Bearer ...``.
    The dashboard uses the same token: its UI asks for it and sends it as
    ``X-API-Key`` on ``/api/*`` requests. There is no username/password.
EPISENSE_ENABLE_DOCS
    ``true`` exposes Swagger/OpenAPI in production (disabled by default).
    In development docs are always enabled. The API routes stay token-protected.
EPISENSE_CORS_ORIGINS
    Comma-separated exact origins (e.g. ``https://dash.example.org``).
    Wildcards are rejected. Empty means CORS is disabled.
EPISENSE_RATE_LIMIT_PER_MINUTE
    Per-IP request limit. ``0`` disables the limiter. The limit is enforced
    per process; with multiple uvicorn workers the effective limit scales
    with the number of workers.
TRUSTED_PROXY_CIDR
    Comma-separated proxy networks allowed to supply ``X-Forwarded-For`` /
    ``X-Real-IP``. Untrusted peers cannot spoof their client IP.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import sqlite3
import math
from contextlib import closing
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

try:  # python-dotenv is present in both runtime requirements files.
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)
except Exception:  # pragma: no cover - env vars still work without dotenv
    pass

logger = logging.getLogger(__name__)

DEFAULT_API_KEY = "dev-token-secret-change-in-production"
DEFAULT_RATE_LIMIT_PER_MINUTE = 120
RATE_LIMIT_WINDOW_SECONDS = 60
MIN_API_KEY_LENGTH = 32

_DEVELOPMENT_ENVIRONMENTS = {"development", "dev", "local", "test"}
_API_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
_PLACEHOLDER_MARKERS = (
    "replace-with",
    "change-in-production",
    "change-me",
    "changeme",
    "your-api-key",
    "your-password",
    "example-secret",
)


class SecurityConfigError(RuntimeError):
    """Raised when the security configuration would run insecurely."""


def _is_placeholder_secret(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    if lowered == DEFAULT_API_KEY:
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _compare_secret(provided: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time comparison that tolerates non-ASCII input.

    ``secrets.compare_digest`` raises TypeError for non-ASCII str values, and
    HTTP headers may carry arbitrary latin-1/utf-8 text. Comparing UTF-8 bytes
    keeps the check constant-time and avoids 500s on hostile input.
    """
    if provided is None or expected is None:
        return False
    return secrets.compare_digest(
        provided.encode("utf-8", "surrogatepass"),
        expected.encode("utf-8", "surrogatepass"),
    )


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _read_environment() -> str:
    raw = (
        _env_value("EPISENSE_ENVIRONMENT")
        or _env_value("ENVIRONMENT")
        or "production"
    )
    return raw.strip().lower()


def _parse_int(name: str, default: int, minimum: int = 0) -> int:
    raw = _env_value(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SecurityConfigError(
            f"{name} must be an integer, got {raw!r}"
        ) from exc
    if value < minimum:
        raise SecurityConfigError(
            f"{name} must be >= {minimum}, got {value}"
        )
    return value


def _parse_bool(name: str, default: bool = False) -> bool:
    raw = _env_value(name)
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SecurityConfigError(
        f"{name} must be a boolean (true/false), got {raw!r}"
    )


def _parse_origins(raw: str) -> Tuple[str, ...]:
    if not raw:
        return ()
    origins = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise SecurityConfigError(
                "EPISENSE_CORS_ORIGINS cannot contain '*'; list explicit origins"
            )
        parsed = urlparse(origin)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise SecurityConfigError(
                f"Invalid CORS origin {origin!r}: expected scheme://host[:port]"
            )
        if parsed.username or parsed.password:
            raise SecurityConfigError(
                f"Invalid CORS origin {origin!r}: credentials are not allowed"
            )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise SecurityConfigError(
                f"Invalid CORS origin {origin!r}: paths/query/fragment are not allowed"
            )
        origins.append(f"{parsed.scheme.lower()}://{parsed.netloc}")
    return tuple(dict.fromkeys(origins))


def _parse_trusted_proxies(raw: str) -> Tuple[Any, ...]:
    if not raw:
        return ()
    networks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise SecurityConfigError(
                f"Invalid TRUSTED_PROXY_CIDR entry {item!r}"
            ) from exc
    return tuple(networks)


@dataclass(frozen=True)
class SecuritySettings:
    environment: str
    api_key: Optional[str]
    cors_origins: Tuple[str, ...]
    rate_limit_per_minute: int
    trusted_proxies: Tuple[Any, ...]
    enable_docs: bool = False

    @property
    def is_production(self) -> bool:
        return self.environment not in _DEVELOPMENT_ENVIRONMENTS

    @property
    def docs_enabled(self) -> bool:
        """Docs are always on in development; opt-in in production."""
        return self.enable_docs or not self.is_production

    def validate(self, service: str) -> None:
        """Fail closed in production; warn for insecure development setups.

        Both the API and the dashboard are protected by the same API token, so
        the requirements are identical. ``service`` is kept for clear logs and
        to reject typos.
        """
        if service not in {"api", "dashboard"}:
            raise ValueError(f"Unknown service {service!r}")

        if self.is_production:
            if not self.api_key:
                raise SecurityConfigError(
                    "EPISENSE_API_KEY is required when ENVIRONMENT=production "
                    "(API and dashboard token)"
                )
            if len(self.api_key) < MIN_API_KEY_LENGTH:
                raise SecurityConfigError(
                    "EPISENSE_API_KEY must have at least "
                    f"{MIN_API_KEY_LENGTH} characters in production"
                )
            if _is_placeholder_secret(self.api_key):
                raise SecurityConfigError(
                    "EPISENSE_API_KEY still uses an example value; rotate it"
                )
        elif not self.api_key:
            logger.warning(
                "API token authentication is disabled (development environment)"
            )
        elif _is_placeholder_secret(self.api_key):
            logger.warning(
                "EPISENSE_API_KEY uses an example value; rotate it before deploying"
            )

        if self.cors_origins:
            logger.info(
                "CORS allowed origins: %s", ", ".join(self.cors_origins)
            )
        else:
            logger.info("CORS disabled (no EPISENSE_CORS_ORIGINS configured)")


@lru_cache(maxsize=1)
def get_settings() -> SecuritySettings:
    """Read and cache settings from the environment (and optional .env)."""
    return SecuritySettings(
        environment=_read_environment(),
        api_key=_env_value("EPISENSE_API_KEY") or None,
        cors_origins=_parse_origins(_env_value("EPISENSE_CORS_ORIGINS")),
        rate_limit_per_minute=_parse_int(
            "EPISENSE_RATE_LIMIT_PER_MINUTE",
            DEFAULT_RATE_LIMIT_PER_MINUTE,
            minimum=0,
        ),
        trusted_proxies=_parse_trusted_proxies(
            _env_value("TRUSTED_PROXY_CIDR")
        ),
        enable_docs=_parse_bool("EPISENSE_ENABLE_DOCS", False),
    )


class RateLimiter:
    """Small thread-safe fixed-window limiter.

    It is intentionally in-process: the API currently runs with multiple
    uvicorn workers, so the effective limit is per worker. A shared store
    (Redis/file lock) can replace it later without changing the middleware.
    """

    def __init__(
        self,
        limit_per_minute: int,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self.limit = int(limit_per_minute)
        self.window = max(1, int(window_seconds))
        self._lock = Lock()
        self._buckets: Dict[str, Tuple[float, int]] = {}
        self._last_prune = 0.0

    def check(
        self, key: str, now: Optional[float] = None
    ) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0

        current = time.monotonic() if now is None else float(now)
        with self._lock:
            start, count = self._buckets.get(key, (current, 0))
            if current - start >= self.window:
                start, count = current, 0

            if count >= self.limit:
                retry_after = max(1, int(self.window - (current - start)))
                return False, retry_after

            self._buckets[key] = (start, count + 1)

            if (
                len(self._buckets) > 10_000
                and current - self._last_prune >= self.window
            ):
                self._last_prune = current
                cutoff = current - (2 * self.window)
                self._buckets = {
                    bucket_key: bucket
                    for bucket_key, bucket in self._buckets.items()
                    if bucket[0] >= cutoff
                }

            return True, 0


def extract_api_key(headers: Mapping[str, str]) -> Optional[str]:
    """Extract an API key from X-API-Key or Authorization: Bearer."""
    raw = headers.get("x-api-key")
    if raw:
        raw = raw.strip()
        if raw:
            return raw

    auth = (headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


def client_ip(request: Request, trusted_proxies: Tuple[Any, ...]) -> str:
    """Resolve the client IP without trusting spoofable proxy headers."""
    peer = request.client.host if request.client else "unknown"
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return peer

    if not any(peer_ip in network for network in trusted_proxies):
        return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    for value in reversed(forwarded.split(",")):
        try:
            candidate = ipaddress.ip_address(value.strip())
        except ValueError:
            return peer
        if not any(candidate in network for network in trusted_proxies):
            return str(candidate)
    return peer


def _rate_limit_response(retry_after: int) -> JSONResponse:
    return JSONResponse(
        {"detail": "Rate limit exceeded"},
        status_code=429,
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _record_invalid_token(origin: str) -> int:
    """Atomically apply a failure window with inactivity expiry shared by all workers."""
    database = os.getenv("EPISENSE_AUTH_DB", "/tmp/episense-auth.sqlite3")
    descriptor = os.open(database, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        os.fchmod(descriptor, 0o660)
    except PermissionError:
        # Another service may own the database and grant access through the shared group.
        pass
    finally:
        os.close(descriptor)
    with closing(sqlite3.connect(database, timeout=2, isolation_level=None)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS auth_failures (
            scope TEXT PRIMARY KEY, failures INTEGER, expires REAL, blocked_until REAL
        )""")
        db.execute("BEGIN IMMEDIATE")
        with db:
            now = time.time()
            db.execute("DELETE FROM auth_failures WHERE expires <= ?", (now,))
            buckets = [("global", 120), ("client:" + origin, 6)]
            states = {}
            for scope, _ in buckets:
                states[scope] = db.execute(
                    "SELECT failures, blocked_until FROM auth_failures WHERE scope = ?", (scope,)
                ).fetchone() or (0, 0)
            retry = max(until for _, until in states.values()) - now
            if retry > 0:
                return math.ceil(retry)

            for scope, limit in buckets:
                failures = states[scope][0] + 1
                delay = min(900, 30 * 2 ** min(5, failures - limit)) if failures >= limit else 0
                db.execute(
                    "INSERT OR REPLACE INTO auth_failures VALUES (?, ?, ?, ?)",
                    (scope, failures, now + 900, now + delay),
                )
                retry = max(retry, delay)
                if delay:
                    logger.warning("Authentication throttled (%s)", "global" if scope == "global" else "client")
            return max(0, math.ceil(retry))


async def _invalid_token_response(origin: str, provided: Optional[str]) -> JSONResponse:
    # Browsers may probe for authentication requirements without sending a token.
    if provided:
        try:
            retry_after = await run_in_threadpool(_record_invalid_token, origin)
        except (sqlite3.Error, OSError):
            logger.exception("Authentication limiter unavailable")
            return _rate_limit_response(30)
        if retry_after:
            return _rate_limit_response(retry_after)
    return JSONResponse(
        {"detail": "Invalid or missing token"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
    )


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require the API key (when configured) and apply per-IP rate limiting."""

    def __init__(self, app, settings: SecuritySettings) -> None:
        super().__init__(app)
        self.settings = settings
        self.limiter = RateLimiter(settings.rate_limit_per_minute)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in {"/", "/health"}:
            return await call_next(request)

        # Docs are public when enabled so Swagger UI can load and use its
        # Authorize button; the API routes themselves stay token-protected.
        if self.settings.docs_enabled and path in _API_DOCS_PATHS:
            return await call_next(request)

        origin = client_ip(request, self.settings.trusted_proxies)
        if self.settings.api_key:
            provided = extract_api_key(request.headers)
            if not _compare_secret(provided, self.settings.api_key):
                return await _invalid_token_response(origin, provided)

        allowed, retry_after = self.limiter.check(origin)
        if not allowed:
            return _rate_limit_response(retry_after)

        return await call_next(request)


class DashboardTokenMiddleware(BaseHTTPMiddleware):
    """Protect dashboard routes with the same token used by the API.

    Only the HTML shell (``/``), static assets and ``/health`` are public so
    the token prompt and the container healthcheck can load. Every other route
    requires ``X-API-Key`` (or Bearer) when a token is configured.
    """

    def __init__(self, app, settings: SecuritySettings) -> None:
        super().__init__(app)
        self.settings = settings
        self.limiter = RateLimiter(settings.rate_limit_per_minute)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or path in {"/", "/health"}
            or path.startswith("/static/")
        ):
            return await call_next(request)

        # Dashboard Swagger is optional; let it load so Authorize can be used.
        if self.settings.docs_enabled and path in _API_DOCS_PATHS:
            return await call_next(request)

        origin = client_ip(request, self.settings.trusted_proxies)
        if self.settings.api_key:
            provided = extract_api_key(request.headers)
            if not _compare_secret(provided, self.settings.api_key):
                return await _invalid_token_response(origin, provided)

        allowed, retry_after = self.limiter.check(origin)
        if not allowed:
            return _rate_limit_response(retry_after)

        return await call_next(request)


__all__ = [
    "ApiKeyMiddleware",
    "DashboardTokenMiddleware",
    "RateLimiter",
    "SecurityConfigError",
    "SecuritySettings",
    "client_ip",
    "extract_api_key",
    "get_settings",
]
