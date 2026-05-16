"""
Generate TypeScript types from Pydantic schemas.
Uses FastAPI's OpenAPI schema + openapi-typescript.
Writes output to libs/shared-types/src/types/api.ts

Auto-loads .env from apps/api/ or repo root so no manual sourcing needed.
Gracefully skips if required env vars are missing (safe for CI).
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def _load_env() -> Path | None:
    """Load .env into os.environ before app.main import, without overriding existing vars."""
    from dotenv import load_dotenv

    candidates = [
        Path(__file__).parent.parent / ".env",           # apps/api/.env
        Path(__file__).parent.parent.parent.parent / ".env",  # repo root .env
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            return path
    return None


def _check_required_vars() -> list[str]:
    required = ["GEMINI_API_KEY", "TAVILY_API_KEY", "DATABASE_URL"]
    return [v for v in required if not os.environ.get(v)]


def generate_from_openapi():
    loaded = _load_env()
    missing = _check_required_vars()
    if missing:
        env_info = f" (loaded from {loaded})" if loaded else ""
        print(
            f"Skipping type generation: missing {', '.join(missing)}{env_info}. "
            f"Set them in apps/api/.env or the environment."
        )
        return

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
