import os

def using_postgres() -> bool:
    """Return True when we should use Postgres.

    The Streamlit app is deployed against Supabase Postgres. We consider Postgres
    enabled when DATABASE_URL is set. For back-compat, APP_MODE may also force it.
    """
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if db_url:
        return True

    app_mode = (os.getenv("APP_MODE") or "local").strip().lower()
    return app_mode in {"render", "cloud", "prod", "production"}

def some_existing_function():
    pass

# (rest of the file continues unchanged)