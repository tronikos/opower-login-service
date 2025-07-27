# Opower Utility Login Service

This service provides a FastAPI-based web interface to automate logins for Opower-based utility websites using Playwright for headless browser automation. It is designed to handle complex login flows, including Multi-Factor Authentication (MFA), and intelligently manages and persists sessions to minimize re-logins.

This service is intended for use with the [tronikos/opower](https://github.com/tronikos/opower) project.

## Features

- **Headless Browser Automation**: Uses Playwright to reliably simulate a user logging into a website.
- **Multi-Factor Authentication (MFA) Support**: An interactive API flow allows for selecting an MFA method and submitting a code.
- **Intelligent Session Management**: Keeps one active browser session per user in memory to avoid redundant logins and resource creation. Stale sessions are automatically cleaned up.
- **Session Persistence**: Saves successful browser session data (cookies, local storage) to disk, which allows you to skip the MFA verification step on future logins.
- **Modern Tech Stack**: Built with FastAPI, Pydantic, and modern asynchronous Python.
- **Containerized**: Comes with a production-ready, multi-arch (amd64, arm64) Dockerfile and a GitHub Actions workflow for automated builds and releases to GHCR.
- **Extensible**: Easily support new utilities by implementing a simple Python class.

## Supported Utilities

- Pacific Gas & Electric (PG&E)

### Adding a New Utility

In `src/app/main.py`, create a new class that inherits from `UtilityAutomator` similar to the existing `PgeAutomator` class.
The service will automatically discover and register it.

The best way to develop the automation logic is to run the service locally with a visible browser, see below.
Use the web interface at `http://localhost:7937` to test your changes in real-time.
Use your browser's "Inspect Element" tool to find the correct selectors for buttons and input fields.

## Local Development Setup

### 1. Create a Virtual Environment

```sh
python3 -m venv .venv
source .venv/bin/activate
# For Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the Python packages and the required Playwright browser and system dependencies.

```sh
python -m pip install --upgrade pip
pip install -r src/requirements.txt
playwright install --with-deps firefox
```

*Note: The `--with-deps` flag attempts to install necessary system libraries. If you encounter issues, you may need to install them manually.*

### 3. Set Up Pre-commit Hooks

This is recommended for maintaining code quality. The hooks will automatically format and lint your code before each commit.

```sh
pip install pre-commit
pre-commit install

# Optional:
pre-commit autoupdate
pre-commit run --all-files
```

## Running the Service

### Running Locally

For development, you can run the service directly with `uvicorn`. Setting `BROWSER_HEADLESS=false` will open a visible browser window, which is useful for debugging automation scripts.

```sh
# Run in headless mode (default)
uvicorn src.app.main:app --port 7937

# Run with a visible browser for debugging
BROWSER_HEADLESS=false uvicorn src.app.main:app --port 7937
# For Windows PowerShell: $env:BROWSER_HEADLESS="false"; uvicorn src.app.main:app --port 7937
```

Once running, access the interactive test form at `http://localhost:7937`.

### Running with Docker

The provided `Dockerfile` builds a production-ready container.

1. **Build the Docker Image:**

    ```sh
    docker build -t opower-login-service src
    ```

2. **Run the Docker Container:**

    To persist login sessions across container restarts, you must mount a volume to the `/data` directory.

    ```sh
    docker run -d -p 7937:7937 -v "$(pwd)/sessions:/data" --name opower-login-service --rm opower-login-service
    ```

    The service will be available at `http://localhost:7937`.

    To stop the Docker container.

    ```sh
    docker stop opower-login-service
    ```

## API Endpoints

The service provides an interactive HTML form for testing and a JSON API for programmatic use.

- `GET /`: Serves an interactive HTML form for manually testing the login flow.
- `GET /api/v1/health`: Health check which just returns `{"status": "ok"}`.
- `POST /api/v1/login`: Initiates a login attempt.
- `POST /api/v1/mfa/select`: Selects an MFA delivery method (e.g., email, SMS).
- `POST /api/v1/mfa/submit`: Submits an MFA code.

## Configuration

The service can be configured using environment variables:

- `BROWSER_HEADLESS`: Set to `false` to run the browser in non-headless (visible) mode for debugging. Defaults to `true`.
- `SESSION_STORAGE_PATH`: The directory path inside the container where session files are stored. Defaults to `/data` in the Docker environment.
