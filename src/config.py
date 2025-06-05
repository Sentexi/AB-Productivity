import os


def _read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


NOTION_TOKEN = os.getenv("NOTION_TOKEN") or _read_file("secret.txt")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID") or _read_file("db.txt")

if not NOTION_TOKEN or not NOTION_DATABASE_ID:
    raise RuntimeError("Missing Notion credentials")
