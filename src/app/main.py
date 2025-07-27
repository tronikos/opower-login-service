"""Browser automation service via FastAPI for Opower utility websites."""

import asyncio
import base64
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Request as PlaywrightRequest,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from pydantic import BaseModel, Field

# --- Configuration ---
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() != "false"
SESSION_DATA_DIR = Path(os.getenv("SESSION_STORAGE_PATH", "sessions"))
SUCCESS_SESSION_TTL = timedelta(minutes=5)
INACTIVE_SESSION_TTL = timedelta(minutes=15)
HTML_TEMPLATE_PATH = Path(__file__).parent / "index.html"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_LOGGER = logging.getLogger(__name__)

# --- Data Models for Automation Results ---


class LoginSuccess(BaseModel):
    """Response model for a successful login."""

    status: Literal["success"] = "success"
    access_token: str


class MfaRequired(BaseModel):
    """Response model when MFA is required."""

    status: Literal["mfa_required"] = "mfa_required"
    mfa_options: dict[str, str]


class LoginFailed(BaseModel):
    """Response model for a failed login attempt."""

    status: Literal["login_failed"] = "login_failed"
    error: str


LoginResult = LoginSuccess | MfaRequired | LoginFailed


class CodeSent(BaseModel):
    """Response for successful MFA method selection."""

    status: Literal["code_sent"] = "code_sent"


class MfaFailed(BaseModel):
    """Response for failed MFA submission."""

    status: Literal["mfa_failed"] = "mfa_failed"
    error: str


MfaSelectResult = CodeSent | MfaFailed
MfaSubmitResult = LoginSuccess | MfaFailed


# --- Utility Automation Classes ---


class UtilityAutomator(ABC):
    """Abstract base class for utility-specific browser automation logic."""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Return the unique, lowercase identifier for the utility."""

    @staticmethod
    @abstractmethod
    def friendly_name() -> str:
        """Return the user-friendly name for the utility."""

    async def _intercept_and_get_token(self, page: Page, trigger_action: asyncio.Task[Any]) -> str:
        """Set up a request interceptor to capture the Opower bearer token.

        Args:
            page: The Playwright page to monitor.
            trigger_action: An awaitable task that will trigger the API call.

        """
        token_holder: list[str] = []
        interception_event = asyncio.Event()

        async def intercept_request(request: PlaywrightRequest) -> None:
            if "opower.com/ei/edge/apis" in request.url:
                auth_header = request.headers.get("authorization")
                if auth_header and auth_header.lower().startswith("bearer "):
                    token = auth_header.split(" ", 1)[1]
                    if not token_holder:
                        _LOGGER.info("Opower bearer token captured.")
                        token_holder.append(token)
                        interception_event.set()

        page.on("request", intercept_request)
        try:
            # Wait for both the trigger action and the interception event to complete
            await asyncio.gather(trigger_action, asyncio.wait_for(interception_event.wait(), 20))
        except (PlaywrightTimeoutError, TimeoutError) as e:
            raise Exception("Timeout: Did not capture the Opower token within 20 seconds.") from e
        finally:
            page.remove_listener("request", intercept_request)

        if not token_holder:
            raise Exception("Failed to capture token after triggering API call.")
        return token_holder[0]

    @abstractmethod
    async def login(self, page: Page, username: str, password: str) -> LoginResult:
        """Perform login, reusing the session if valid, or logging in fresh."""

    @abstractmethod
    async def select_mfa_method(self, page: Page, method: str) -> MfaSelectResult:
        """Select the MFA method (e.g., clicks the 'Email' or 'SMS' button)."""

    @abstractmethod
    async def submit_mfa_code(self, page: Page, code: str) -> MfaSubmitResult:
        """Submit the MFA code."""


class PgeAutomator(UtilityAutomator):
    """Implement automation for Pacific Gas & Electric (PG&E)."""

    LOGIN_URL: ClassVar[str] = "https://myaccount.pge.com/"

    # Page element selectors
    USERNAME_SELECTOR: ClassVar[str] = 'input[name="username"]'
    MAINTENANCE_SELECTOR: ClassVar[str] = "div.maintenance-description"
    DASHBOARD_SELECTOR: ClassVar[str] = "div.billperiodusageheading"
    MFA_SELECTOR: ClassVar[str] = "fieldset.mfaFieldset"
    ERROR_SELECTOR: ClassVar[str] = "div.errorClass"

    @staticmethod
    def name() -> str:
        """Return the unique, lowercase identifier for the utility."""
        return "pge"

    @staticmethod
    def friendly_name() -> str:
        """Return the user-friendly name for the utility."""
        return "Pacific Gas & Electric (PG&E)"

    async def _get_token_from_dashboard(self, page: Page) -> LoginSuccess:
        """Trigger and intercept token on a page where the user is already logged in."""
        _LOGGER.info("User is already logged in, attempting to capture token.")
        action = asyncio.create_task(page.locator('div.slds-card:has-text("Bill period usage")').click())
        token = await self._intercept_and_get_token(page, action)
        return LoginSuccess(access_token=token)

    async def login(self, page: Page, username: str, password: str) -> LoginResult:
        """Perform login, reusing the session (cookies) if valid, or logging in fresh."""
        _LOGGER.info("Attempting login for utility '%s'", self.name())
        await page.goto(self.LOGIN_URL, wait_until="domcontentloaded")

        initial_selector = f"{self.DASHBOARD_SELECTOR}, {self.USERNAME_SELECTOR}, {self.MAINTENANCE_SELECTOR}"
        await page.wait_for_selector(initial_selector, state="visible", timeout=25000)

        if await page.locator(self.DASHBOARD_SELECTOR).is_visible():
            _LOGGER.info("Session reuse successful. Intercepting token.")
            return await self._get_token_from_dashboard(page)

        if await page.locator(self.MAINTENANCE_SELECTOR).is_visible():
            text = await page.locator(self.MAINTENANCE_SELECTOR).text_content()
            msg = text.strip() if text else "System maintenance"
            _LOGGER.warning("PG&E login failed due to maintenance: %s", msg)
            return LoginFailed(error=msg)

        if await page.locator(self.USERNAME_SELECTOR).is_visible():
            _LOGGER.info("No valid session found, performing manual login.")
            await page.locator(self.USERNAME_SELECTOR).fill(username)
            await page.locator('input[name="password"]').fill(password)
            await page.get_by_role("button", name="Sign In").click()

            outcome_selector = f"{self.DASHBOARD_SELECTOR}, {self.MFA_SELECTOR}, {self.ERROR_SELECTOR}"
            await page.wait_for_selector(outcome_selector, state="visible", timeout=20000)

            if await page.locator(self.DASHBOARD_SELECTOR).is_visible():
                _LOGGER.info("Login successful (no MFA needed).")
                return await self._get_token_from_dashboard(page)

            if await page.locator(self.MFA_SELECTOR).is_visible():
                _LOGGER.info("MFA Required.")
                options: dict[str, str] = {}
                if await page.get_by_role("button", name="Email").is_visible():
                    options["email"] = "Email"
                if await page.get_by_role("button", name="SMS Text").is_visible():
                    options["sms"] = "SMS Text"
                return MfaRequired(mfa_options=options)

            if await page.locator(self.ERROR_SELECTOR).is_visible():
                error = await page.locator(self.ERROR_SELECTOR).evaluate("el => el.firstChild.textContent.trim()")
                _LOGGER.warning("Login Failed: %s", error)
                return LoginFailed(error=error)

        return LoginFailed(error="Login resulted in an unknown page state.")

    async def select_mfa_method(self, page: Page, method: str) -> MfaSelectResult:
        """Select the MFA method (e.g., clicks the 'Email' or 'SMS' button)."""
        button_name = "Email" if method == "email" else "SMS Text"
        await page.get_by_role("button", name=button_name).click()
        await page.get_by_label("Security code").wait_for(state="visible", timeout=15000)
        return CodeSent()

    async def submit_mfa_code(self, page: Page, code: str) -> MfaSubmitResult:
        """Submit the MFA code."""
        try:
            await page.get_by_label("Security code").fill(code)
            await page.get_by_role("button", name="Confirm").click()

            await page.locator(self.DASHBOARD_SELECTOR).wait_for(state="visible", timeout=20000)
            return await self._get_token_from_dashboard(page)
        except Exception:
            if await page.locator(self.ERROR_SELECTOR).is_visible():
                error_text = await page.locator(self.ERROR_SELECTOR).text_content()
                return MfaFailed(error=str(error_text).strip())

            return MfaFailed(error="MFA submission timed out or resulted in an unknown page state.")


# --- Application State and Session Management ---


class ActiveSession(BaseModel):
    """Manage an active Playwright browser session for a user."""

    playwright: Playwright
    browser: Browser
    context: BrowserContext
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["ready", "mfa_required", "success", "error"] = "ready"
    mfa_client_id: str | None = None

    class Config:
        """Configuration for the Pydantic model."""

        arbitrary_types_allowed = True


# Global state
ACTIVE_SESSIONS: dict[str, ActiveSession] = {}  # Key is "utility:username"
MFA_SESSION_MAP: dict[str, str] = {}  # Maps client-facing session_id to internal session key
UTILITY_REGISTRY: dict[str, UtilityAutomator] = {
    impl.name(): impl()  # type: ignore[abstract]
    for impl in UtilityAutomator.__subclasses__()
}

# --- FastAPI Application ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    """Handle application startup and shutdown events."""
    _LOGGER.info(
        "Starting up. Supported utilities: %s. Headless: %s. Session data dir: %s",
        list(UTILITY_REGISTRY.keys()),
        BROWSER_HEADLESS,
        SESSION_DATA_DIR.resolve(),
    )
    SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())
    yield
    _LOGGER.info("Shutting down. Cleaning up all active sessions.")
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    # Final cleanup on shutdown
    await asyncio.gather(*(cleanup_session(key) for key in list(ACTIVE_SESSIONS.keys())))


app = FastAPI(title="Opower Utility Headless Login Service", lifespan=lifespan)

# --- Pydantic Models for API ---


class LoginRequest(BaseModel):
    """Model for the initial login request."""

    utility: str
    username: str
    password: str


class MfaSelectRequest(BaseModel):
    """Model for the MFA method selection request."""

    session_id: str
    method: str


class MfaSubmitRequest(BaseModel):
    """Model for the MFA code submission request."""

    session_id: str
    code: str


class ApiResponse(BaseModel):
    """Generic response model for all API endpoints."""

    status: str
    access_token: str | None = None
    error: str | None = None
    mfa_options: dict[str, str] | None = None
    session_id: str | None = None
    screenshot_uri: str | None = None


# --- Session Management Helpers ---


def get_session_key(utility: str, username: str) -> str:
    """Generate a unique key for the active session dictionary."""
    return f"{utility}:{username}"


def get_session_storage_path(utility: str, username: str) -> Path:
    """Get the file path for a persisted session, ensuring the directory exists."""
    safe_username = base64.urlsafe_b64encode(username.encode()).decode()
    return SESSION_DATA_DIR / f"session-{utility}-{safe_username}.json"


async def save_session_state(context: BrowserContext, utility: str, username: str) -> None:
    """Save the browser context's state to a file."""
    path = get_session_storage_path(utility, username)
    _LOGGER.info("Saving session state for user '%s' to %s", username, path)
    try:
        await context.storage_state(path=path)
    except Exception as e:
        _LOGGER.error("Failed to save session state: %s", e, exc_info=True)


async def get_or_create_session(utility: str, username: str) -> ActiveSession:
    """Retrieve an existing session or create a new one."""
    key = get_session_key(utility, username)
    if key in ACTIVE_SESSIONS:
        _LOGGER.info("Reusing existing session for key: %s", key)
        session = ACTIVE_SESSIONS[key]
        session.last_accessed = datetime.now(UTC)
        return session

    _LOGGER.info("Creating new session for key: %s", key)
    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch(headless=BROWSER_HEADLESS)
    storage_path = get_session_storage_path(utility, username)
    storage_state = storage_path if storage_path.exists() else None
    if storage_state:
        _LOGGER.info("Loading persisted session from %s", storage_path)
    context = await browser.new_context(storage_state=storage_state)
    session = ActiveSession(playwright=playwright, browser=browser, context=context)
    ACTIVE_SESSIONS[key] = session
    return session


async def cleanup_session(key: str) -> None:
    """Safely close all resources for a given session and remove it."""
    session = ACTIVE_SESSIONS.pop(key, None)
    if not session:
        return

    _LOGGER.info("Cleaning up session: %s", key)
    if session.mfa_client_id:
        MFA_SESSION_MAP.pop(session.mfa_client_id, None)

    # Suppress errors on close, as resources might already be disconnected
    with suppress(Exception):
        await session.context.close()
    with suppress(Exception):
        if session.browser.is_connected():
            await session.browser.close()
    with suppress(Exception):
        await session.playwright.stop()


async def capture_screenshot(page: Page) -> str:
    """Capture screenshot as a base64-encoded data URI."""
    try:
        if page.is_closed():
            return ""
        encoded_image = base64.b64encode(await page.screenshot(full_page=True)).decode()
        return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        _LOGGER.error("Failed to capture screenshot: %s", e)
        return ""


# --- API Endpoints ---


@app.get("/api/v1/health", tags=["API"])
async def api_health() -> JSONResponse:
    """Provide a simple health check endpoint."""
    return JSONResponse(content={"status": "ok"})


@app.post("/api/v1/login", response_model=ApiResponse, tags=["API"])
async def api_login(req: LoginRequest, include_screenshot: bool = False) -> JSONResponse:
    """Handle the primary login request, supporting session reuse and MFA."""
    automator = UTILITY_REGISTRY.get(req.utility)
    if not automator:
        raise HTTPException(status_code=400, detail="Unsupported utility")

    key = get_session_key(req.utility, req.username)
    session = await get_or_create_session(req.utility, req.username)
    page = session.context.pages[0] if session.context.pages else await session.context.new_page()
    response_data: dict[str, Any] = {}

    async with session.lock:
        try:
            result = await automator.login(page, req.username, req.password)
            session.last_accessed = datetime.now(UTC)
            response_data = result.model_dump()

            if isinstance(result, LoginSuccess):
                session.status = "success"
                await save_session_state(session.context, req.utility, req.username)

            elif isinstance(result, MfaRequired):
                session.status = "mfa_required"
                session.mfa_client_id = str(uuid.uuid4())
                MFA_SESSION_MAP[session.mfa_client_id] = key
                response_data["session_id"] = session.mfa_client_id

            session.status = "error"

        except Exception as e:
            _LOGGER.error("API Error on login for '%s': %s", key, e, exc_info=True)
            session.status = "error"
            response_data = {"status": "error", "error": str(e)}

        finally:
            if include_screenshot:
                response_data["screenshot_uri"] = await capture_screenshot(page)

    status_code = 200
    if response_data.get("status") in ["login_failed", "error"]:
        status_code = 401 if response_data["status"] == "login_failed" else 500

    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/api/v1/mfa/select", response_model=ApiResponse, tags=["API"])
async def api_mfa_select(req: MfaSelectRequest, include_screenshot: bool = False) -> JSONResponse:
    """Handle the selection of an MFA method."""
    key = MFA_SESSION_MAP.get(req.session_id)
    if not key or not (session := ACTIVE_SESSIONS.get(key)):
        raise HTTPException(status_code=404, detail="Session not found or expired")

    automator = UTILITY_REGISTRY[key.split(":", 1)[0]]
    page = session.context.pages[0]
    response_data = {}
    status_code = 500

    async with session.lock:
        try:
            result = await automator.select_mfa_method(page, req.method)
            session.last_accessed = datetime.now(UTC)
            response_data = result.model_dump()
            status_code = 200 if isinstance(result, CodeSent) else 400
        except Exception as e:
            _LOGGER.error("API Error on MFA select for '%s': %s", key, e, exc_info=True)
            session.status = "error"
            response_data = {"status": "error", "error": str(e)}
        finally:
            if include_screenshot:
                response_data["screenshot_uri"] = await capture_screenshot(page)

    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/api/v1/mfa/submit", response_model=ApiResponse, tags=["API"])
async def api_mfa_submit(req: MfaSubmitRequest, include_screenshot: bool = False) -> JSONResponse:
    """Handle the submission of the MFA code."""
    key = MFA_SESSION_MAP.get(req.session_id)
    if not key or not (session := ACTIVE_SESSIONS.get(key)):
        raise HTTPException(status_code=404, detail="Session not found or expired")

    utility, username = key.split(":", 1)
    automator = UTILITY_REGISTRY[utility]
    page = session.context.pages[0]
    response_data = {}
    status_code = 500

    async with session.lock:
        try:
            result = await automator.submit_mfa_code(page, req.code)
            session.last_accessed = datetime.now(UTC)
            response_data = result.model_dump()

            if isinstance(result, LoginSuccess):
                session.status = "success"
                status_code = 200
                await save_session_state(session.context, utility, username)
                # Clean up MFA mapping on success
                MFA_SESSION_MAP.pop(req.session_id, None)
                session.mfa_client_id = None
            else:
                status_code = 400
                # Don't mark session as "error" on MFA failure, allow retry
        except Exception as e:
            _LOGGER.error("API Error on MFA submit for '%s': %s", key, e, exc_info=True)
            session.status = "error"
            response_data = {"status": "error", "error": str(e)}
        finally:
            if include_screenshot:
                response_data["screenshot_uri"] = await capture_screenshot(page)

    return JSONResponse(content=response_data, status_code=status_code)


# --- Interactive HTML Test Form ---


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_test_form(request: Request) -> HTMLResponse:
    """Serve the main test form which uses JavaScript to call the JSON API."""
    try:
        template_content = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="index.html template not found.") from e

    options_html = "".join(
        f'<option value="{name}">{impl.friendly_name()}</option>' for name, impl in UTILITY_REGISTRY.items()
    )

    final_html = template_content.replace("<!-- UTILITY_OPTIONS_PLACEHOLDER -->", options_html)

    return HTMLResponse(content=final_html)


# --- Background Cleanup Task ---
async def _cleanup_expired_sessions() -> None:
    """Periodically check for and clean up expired sessions."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(UTC)
        expired_keys: list[str] = []

        for key, session in ACTIVE_SESSIONS.items():
            if session.lock.locked():
                _LOGGER.info("Session %s is locked, skipping cleanup check.", key)
                continue

            ttl = SUCCESS_SESSION_TTL if session.status == "success" else INACTIVE_SESSION_TTL
            if now - session.last_accessed > ttl:
                expired_keys.append(key)

        if expired_keys:
            _LOGGER.info("Found %d expired sessions to clean up.", len(expired_keys))
            await asyncio.gather(*(cleanup_session(key) for key in expired_keys))
