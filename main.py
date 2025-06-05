from src.database import get_schema, query_database


def main() -> None:
    schema = get_schema()
    keys = list(schema.get("properties", {}).keys())
    print("Schema keys:", keys)

    rows = query_database()
    print("Rows fetched:", len(rows))


if __name__ == "__main__":
    main()
