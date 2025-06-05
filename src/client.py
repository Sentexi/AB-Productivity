import time
import requests
from typing import Any, Dict

BASE_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def request(method: str, path: str, token: str, **kwargs: Any) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    headers = kwargs.pop("headers", {})
    headers.setdefault("Authorization", f"Bearer {token}")
    headers.setdefault("Notion-Version", NOTION_VERSION)
    while True:
        resp = requests.request(method, url, headers=headers, **kwargs)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "1"))
            time.sleep(wait)
            continue
        if not resp.ok:
            raise Exception(f"Notion API error {resp.status_code}: {resp.text}")
        return resp.json()
