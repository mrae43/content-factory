from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.workers.queue_worker import QueueWorker

worker = QueueWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    yield
    await worker.stop()


app = FastAPI(title="Content Factory API (2026)", lifespan=lifespan)

app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "Factory API Online. Awaiting Sparks."}
