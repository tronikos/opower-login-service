# Opower Utility Login Service

This service provides a FastAPI-based web interface to automate logins for Opower-based utility websites using Playwright for headless browser automation. It is designed to handle complex login flows, including Multi-Factor Authentication (MFA), and intelligently manages and persists sessions to minimize re-logins.

This service is intended for use with the [tronikos/opower](https://github.com/tronikos/opower) project and the [Opower integration](https://www.home-assistant.io/integrations/opower) in Home Assistant.

## Supported Utilities

- Pacific Gas & Electric (PG&E)

## Installation

You can run this service in two ways: as a Home Assistant Add-on or as a standalone Docker container.

### 1. Home Assistant Add-on (Recommended)

This is the easiest way to get started if you are a Home Assistant user.

1. Click the button below to add the repository to your Home Assistant instance:
    [![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Ftronikos%2Fopower-login-service)
2. In the repository, find the "Opower Login Service" add-on and click **Install**.
3. Start the add-on.

### 2. Docker

The published multi-arch image is available on the GitHub Container Registry.

1. **Pull the latest image:**

    ```sh
    docker pull ghcr.io/tronikos/opower-login-service:latest
    ```

2. **Run the container:**
    To persist login sessions across container restarts, you must mount a volume to the `/data` directory.

    ```sh
    docker run -d -p 7937:7937 -v "$(pwd)/sessions:/data" --name opower-login-service --rm ghcr.io/tronikos/opower-login-service:latest
    ```

    *For Windows PowerShell, use `-v "${pwd}/sessions:/data"`*

The service will be available at `http://localhost:7937`.

## Features

- **Headless Browser Automation**: Uses Playwright to reliably simulate a user logging into a website.
- **Multi-Factor Authentication (MFA) Support**: An interactive API flow allows for selecting an MFA method and submitting a code.
- **Intelligent Session Management**: Keeps one active browser session per user in memory to avoid redundant logins and resource creation. Stale sessions are automatically cleaned up.
- **Session Persistence**: Saves successful browser session data (cookies, local storage) to disk, which allows you to skip the MFA verification step on future logins.
- **Modern Tech Stack**: Built with FastAPI, Pydantic, and modern asynchronous Python.
- **Containerized**: Comes with a production-ready, multi-arch (amd64, arm64/aarch64) Dockerfile and a GitHub Actions workflow for automated builds and releases to GHCR.
- **Extensible**: Easily support new utilities by implementing a simple Python class.

## Local Development Setup

Follow these steps if you want to contribute to the project or run it from source.

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

### 4. Running the Service Locally

For development, you can run the service directly with `uvicorn`.

```sh
# Run in headless mode (default)
uvicorn src.app.main:app --port 7937

# Run with a visible browser for debugging
BROWSER_HEADLESS=false uvicorn src.app.main:app --port 7937
# For Windows PowerShell: $env:BROWSER_HEADLESS="false"; uvicorn src.app.main:app --port 7937
```

Once running, access the interactive test form at `http://localhost:7937`.

Alternatively, you can build and run a Docker container from source code.

```sh
docker build -t opower-login-service src
docker run -d -p 7937:7937 -v "$(pwd)/sessions:/data" --name opower-login-service --rm opower-login-service
docker stop opower-login-service
```

## Adding a New Utility

It's easy to add support for a new Opower-based utility.

1. In `src/app/main.py`, create a new class that inherits from `UtilityAutomator` similar to the existing `PgeAutomator` class. The service will automatically discover and register it.
2. Run the service locally with a visible browser (see above).
3. Use the web interface at `http://localhost:7937` to test your changes in real-time. Use your browser's "Inspect Element" tool to find the correct selectors for buttons and input fields.

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
