from typing import Any, Dict, List

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
