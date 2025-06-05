from typing import Any, Dict, List
import os
import json
from pathlib import Path
import pandas as pd

from .config import NOTION_TOKEN, NOTION_DATABASE_ID
from .client import request

# Cache for resolved page titles to minimise API calls
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "workspace_cache.json"
try:
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        _PAGE_TITLE_CACHE: Dict[str, str | None] = json.load(f)
except Exception:
    _PAGE_TITLE_CACHE: Dict[str, str | None] = {}


def get_page_title(page_id: str, token: str) -> str | None:
    """Return the title of a page, caching results to avoid extra API calls."""
    if page_id in _PAGE_TITLE_CACHE:
        return _PAGE_TITLE_CACHE[page_id]

    data = request("GET", f"/pages/{page_id}", token)
    title: str | None = None
    for prop in data.get("properties", {}).values():
        if prop.get("type") == "title":
            title = "".join(part.get("plain_text", "") for part in prop.get("title", []))
            break

    _PAGE_TITLE_CACHE[page_id] = title
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(_PAGE_TITLE_CACHE, f)
    except Exception:
        pass
    return title


def get_schema(db_id: str | None = None, token: str | None = None) -> Dict[str, Any]:
    db_id = db_id or NOTION_DATABASE_ID
    token = token or NOTION_TOKEN
    return request("GET", f"/databases/{db_id}", token)


def query_database(
    db_id: str | None = None,
    token: str | None = None,
    **payload: Any,
) -> List[Dict[str, Any]]:
    db_id = db_id or NOTION_DATABASE_ID
    token = token or NOTION_TOKEN
    results: List[Dict[str, Any]] = []
    next_cursor: str | None = None
    while True:
        body = dict(payload)
        if next_cursor:
            body["start_cursor"] = next_cursor
        resp = request("POST", f"/databases/{db_id}/query", token, json=body)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        next_cursor = resp.get("next_cursor")
    return results


def _extract_value(prop: Dict[str, Any], token: str | None = None) -> Any:
    """Return a simplified Python value for a Notion property."""
    t = prop.get("type")
    value = prop.get(t)

    if t in ("title", "rich_text"):
        return "".join(part.get("plain_text", "") for part in value)

    if t in ("select", "status"):
        return value.get("name") if value else None

    if t == "multi_select":
        return ";".join(opt.get("name", "") for opt in value)

    if t == "people":
        return ";".join(p.get("name", "") for p in value)

    if t == "date":
        return value.get("start") if value else None

    if t == "relation":
        ids = [r.get("id") for r in value] if isinstance(value, list) else []
        if token:
            names = []
            for page_id in ids:
                title = get_page_title(page_id, token)
                names.append(title or page_id)
            return ";".join(n for n in names if n)
        return ";".join(pid for pid in ids if pid)

    if t == "rollup":
        rtype = value.get("type")
        if rtype == "array":
            results: List[str] = []
            for item in value.get("array", []):
                itype = item.get("type")
                if itype in ("title", "rich_text"):
                    text_parts = item.get(itype, [])
                    results.append("".join(part.get("plain_text", "") for part in text_parts))
                elif itype == "relation":
                    page_id = item.get("relation", {}).get("id")
                    if page_id:
                        if token:
                            title = get_page_title(page_id, token)
                            results.append(title or page_id)
                        else:
                            results.append(page_id)
                else:
                    data = item.get(itype)
                    if isinstance(data, dict):
                        results.append(str(data.get("id") or data.get("name") or data))
                    else:
                        if data is not None:
                            results.append(str(data))
            return ";".join(r for r in results if r)
        if rtype == "number":
            return value.get("number")
        if rtype == "date":
            return value.get("date", {}).get("start")

    return value


def query_database_dataframe(
    db_id: str | None = None,
    token: str | None = None,
    **payload: Any,
) -> pd.DataFrame:
    """Return the entire Notion database as a pandas ``DataFrame``."""
    token = token or NOTION_TOKEN
    pages = query_database(db_id=db_id, token=token, **payload)
    rows: List[Dict[str, Any]] = []
    for page in pages:
        row: Dict[str, Any] = {
            "id": page.get("id"),
            "Created time": page.get("created_time"),
            "Last edited": page.get("last_edited_time"),
        }
        for name, prop in page.get("properties", {}).items():
            row[name] = _extract_value(prop, token)
        rows.append(row)
    return pd.DataFrame(rows)


def export_database_csv(folder_path: str, filename: str = "notion_all.csv") -> str:
    """Fetch the database and save as a CSV in ``folder_path``."""
    os.makedirs(folder_path, exist_ok=True)
    df = query_database_dataframe(token=NOTION_TOKEN)
    csv_path = os.path.join(folder_path, filename)
    df.to_csv(csv_path, index=False)
    return csv_path
