"""
Generate TypeScript types from Pydantic schemas.
Uses FastAPI's OpenAPI schema + openapi-typescript.
Writes output to libs/shared-types/src/types/api.ts
"""

import json
import subprocess
import sys
from pathlib import Path


def generate_from_openapi():
    from fastapi.openapi.utils import get_openapi

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.main import app

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    output_dir = (
        Path(__file__).parent.parent.parent.parent
        / "libs"
        / "shared-types"
        / "src"
        / "types"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_dir / "openapi-schema.json"
    with open(schema_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    ts_output = output_dir / "api.ts"
    subprocess.run(
        [
            "npx",
            "openapi-typescript",
            str(schema_path),
            "--output",
            str(ts_output),
        ],
        check=True,
        cwd=str(Path(__file__).parent.parent.parent.parent),
    )

    print(f"Generated TypeScript types at {ts_output}")


if __name__ == "__main__":
    generate_from_openapi()
