from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/content_factory",
)

engine = create_async_engine(
    DATABASE_URL, echo=False, future=True, pool_size=20, max_overflow=10
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    """Dependency for FastAPI endpoints."""
    async with AsyncSessionLocal() as session:
        yield session
