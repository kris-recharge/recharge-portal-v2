import os
from zoneinfo import ZoneInfo

# --- Timezones used throughout the app ---
AK_TZ = ZoneInfo("America/Anchorage")
UTC = ZoneInfo("UTC")


# --- Runtime mode detection (local vs. web/Render) ---
def _detect_mode() -> str:
    """
    Decide whether we're running locally or on the web (Render).
    Priority:
      1) APP_MODE env var if provided (explicit override)
      2) Presence of Render-specific env vars
      3) Default to 'local'
    """
    explicit = os.getenv("APP_MODE")
    if explicit:
        return explicit.strip().lower()

    if os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL"):
        return "web"

    return "local"


APP_MODE = _detect_mode()


def is_web() -> bool:
    return APP_MODE == "web"


def is_local() -> bool:
    return not is_web()


# --- Database configuration ---
# Local: prefer LOCAL_SQLITE_PATH, else the standard path we used previously.
SQLITE_PATH = (
    os.getenv("LOCAL_SQLITE_PATH")
    or os.path.expanduser("~/LynkWell DataSync/database/lynkwell_data.db")
)

# Web/Render: use Render-provided URL (or DATABASE_URL fallback)
RENDER_DB_URL = os.getenv("RENDER_DB_URL") or os.getenv("DATABASE_URL")

# Optional Postgres SSL mode (often 'require' on Render)
PGSSLMODE = os.getenv("PGSSLMODE", "")

# Keep DB_BACKEND for compatibility. Auto-pick based on mode unless overridden.
DB_BACKEND = os.getenv(
    "DB_BACKEND",
    "postgres" if is_web() else "sqlite"
).strip().lower()