from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Epistemic Content Factory API (2026)")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "Factory API Online. Awaiting Sparks."}