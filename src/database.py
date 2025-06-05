from typing import Any, Dict, List
import os
import pandas as pd

from .config import NOTION_TOKEN, NOTION_DATABASE_ID
from .client import request


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


def _extract_value(prop: Dict[str, Any]) -> Any:
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
    return value


def query_database_dataframe(
    db_id: str | None = None,
    token: str | None = None,
    **payload: Any,
) -> pd.DataFrame:
    """Return the entire Notion database as a pandas ``DataFrame``."""
    pages = query_database(db_id=db_id, token=token, **payload)
    rows: List[Dict[str, Any]] = []
    for page in pages:
        row: Dict[str, Any] = {
            "id": page.get("id"),
            "Created time": page.get("created_time"),
            "Last edited": page.get("last_edited_time"),
        }
        for name, prop in page.get("properties", {}).items():
            row[name] = _extract_value(prop)
        rows.append(row)
    return pd.DataFrame(rows)


def export_database_csv(folder_path: str, filename: str = "notion_all.csv") -> str:
    """Fetch the database and save as a CSV in ``folder_path``."""
    os.makedirs(folder_path, exist_ok=True)
    df = query_database_dataframe()
    csv_path = os.path.join(folder_path, filename)
    df.to_csv(csv_path, index=False)
    return csv_path
