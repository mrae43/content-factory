import uuid

from sqlalchemy import (
    Column,
    Text,
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.orm import relationship

from app.db.models import Base, TrackedJSONB


ScriptJobStatusEnum = ENUM(
    "PENDING",
    "RESEARCHING",
    "RETRIEVAL",
    "SCRIPTING",
    "FACT_CHECKING_SCRIPT",
    "COMPLETED",
    "FAILED",
    "HUMAN_REVIEW_NEEDED",
    name="script_job_status",
    schema="factory",
    create_type=False,
)

FormatJobStatusEnum = ENUM(
    "PENDING",
    "FORMATTING",
    "ASSET_GENERATION",
    "COMPOSITION",
    "COMPLETED",
    "FAILED",
    "HUMAN_REVIEW_NEEDED",
    name="format_job_status",
    schema="factory",
    create_type=False,
)


class ScriptJob(Base):
    """Tracks a script content generation request from a Discord user."""

    __tablename__ = "script_jobs"
    __table_args__ = {"schema": "factory"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    user_reference = Column(Text, nullable=True)
    source_urls = Column(JSONB, nullable=True)
    story_directives = Column(JSONB, nullable=True)

    status = Column(ScriptJobStatusEnum, nullable=False, server_default="PENDING")

    refined_context = Column(Text, nullable=True)
    assembled_context = Column(JSONB, nullable=True)
    script_content = Column(Text, nullable=True)
    claims = Column(TrackedJSONB, nullable=True)
    working_memory = Column(JSONB, nullable=True)
    hedge_index = Column(JSONB, nullable=True)
    error_log = Column(JSONB, nullable=True)

    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(36), nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    format_jobs = relationship(
        "FormatJob", back_populates="script_job", cascade="save-update, merge"
    )


class FormatJob(Base):
    """Tracks a request to render a completed script into a specific format."""

    __tablename__ = "format_jobs"
    __table_args__ = (
        UniqueConstraint(
            "source_job_id",
            "platform",
            "format_type",
            name="uq_format_jobs_source_platform_type",
        ),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("factory.script_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    title = Column(Text, nullable=False)
    platform = Column(Text, nullable=False)
    format_type = Column(Text, nullable=False)

    status = Column(FormatJobStatusEnum, nullable=False, server_default="PENDING")

    # Snapshot columns — copied from script_job at creation time
    script_content = Column(Text, nullable=True)
    claims = Column(JSONB, nullable=True)
    refined_context = Column(Text, nullable=True)
    story_directives = Column(JSONB, nullable=True)
    hedge_index = Column(JSONB, nullable=True)
    epistemic_ledger = Column(JSONB, nullable=True)
    working_memory = Column(JSONB, nullable=True)

    # Output columns
    format_payload = Column(JSONB, nullable=True)
    final_video_url = Column(Text, nullable=True)

    error_log = Column(JSONB, nullable=True)

    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(36), nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    script_job = relationship("ScriptJob", back_populates="format_jobs")
