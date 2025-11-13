import os
from zoneinfo import ZoneInfo

AK_TZ = ZoneInfo("America/Anchorage")
UTC = ZoneInfo("UTC")

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.expanduser("~/LynkWell DataSync/database/lynkwell_data.db"))
RENDER_DB_URL = os.getenv("RENDER_DB_URL")
APP_MODE = os.getenv("APP_MODE", "local").lower()   # "local" | "web"