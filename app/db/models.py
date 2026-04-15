import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Float,
    Integer,
    ForeignKey,
    DateTime,
    text,
    CheckConstraint,
    Index,
    DDL,
    event,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# ==========================================
# 1. ENUMS (Strict State Management)
# ==========================================
JobStatusEnum = ENUM(
    "PENDING",
    "RESEARCHING",
    "FACT_CHECKING_RESEARCH",
    "SCRIPTING",
    "FACT_CHECKING_SCRIPT",
    "ASSET_GENERATION",
    "COMPLETED",
    "FAILED",
    "HUMAN_REVIEW_NEEDED",
    name="job_status",
    schema="factory",
    create_type=True,
)

AssetTypeEnum = ENUM(
    "VISUAL_VEO",
    "AUDIO_LYRIA",
    "VOICEOVER",
    "SUBTITLE_JSON",
    "DATA_CHART",
    name="asset_type",
    schema="factory",
    create_type=True,
)

VerdictEnum = ENUM(
    "SUPPORTED",
    "CONTESTED",
    "UNSUPPORTED",
    "UNCERTAIN",
    name="verdict_status",
    schema="factory",
    create_type=True,
)

# ==========================================
# 2. CORE MODELS
# ==========================================


class RenderJob(Base):
    """
    Step 1 & 8: The master orchestrator for the Content Factory.
    """

    __tablename__ = "render_jobs"
    __table_args__ = (
        Index(
            "ix_render_jobs_active_queue",
            "status",
            postgresql_where=text("status NOT IN ('COMPLETED', 'FAILED')"),
        ),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(String, nullable=False)

    # Step 1: Pre-context given by the user (URLs, base text, constraints)
    pre_context = Column(JSONB, nullable=False, server_default="{}")

    status = Column(JobStatusEnum, nullable=False, server_default="PENDING")
    strict_compliance_mode = Column(Boolean, default=True, nullable=False)

    # Step 8: Final outputs
    final_video_url = Column(String, nullable=True)
    error_log = Column(JSONB, nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(36), nullable=True)

    # Relationships
    research_chunks = relationship(
        "ResearchChunk", back_populates="job", cascade="all, delete-orphan"
    )
    scripts = relationship("Script", back_populates="job", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="job", cascade="all, delete-orphan")


class ResearchChunk(Base):
    """
    Step 2 & 3: Outputs from the Research Agent. RAG-enabled via pgvector.
    """

    __tablename__ = "research_chunks"
    __table_args__ = (
        Index("ix_research_meta_gin", "meta", postgresql_using="gin"),
        Index(
            "ix_research_embedding_hnsw",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("factory.render_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content = Column(String, nullable=False)
    # Gemini 3.1 Embeddings (typically 768 or 1536 dims)
    embedding = Column(Vector(768), nullable=True)

    # Stores sources, URLs, credibility scores
    meta = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    job = relationship("RenderJob", back_populates="research_chunks")


class Script(Base):
    """
    Step 5: Output from the Copywriter Agent.
    """

    __tablename__ = "scripts"
    __table_args__ = (
        UniqueConstraint("job_id", "version", name="uq_script_job_version"),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("factory.render_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version = Column(Integer, nullable=False, default=1)
    content = Column(String, nullable=False)

    # Has this script passed the Red Team Evaluator?
    is_approved = Column(Boolean, default=False, nullable=False)

    # Agentic Reflection Loop history
    feedback_history = Column(JSONB, nullable=False, server_default="[]")

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("now()"))

    job = relationship("RenderJob", back_populates="scripts")
    claims = relationship(
        "FactCheckClaim", back_populates="script", cascade="all, delete-orphan"
    )


class FactCheckClaim(Base):
    """
    Step 4 & 6: The Red Team Evaluator outputs. Strict constraints applied.
    """

    __tablename__ = "fact_check_claims"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="check_confidence_range"
        ),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    script_id = Column(
        UUID(as_uuid=True),
        ForeignKey("factory.scripts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    claim_text = Column(String, nullable=False)
    verdict = Column(VerdictEnum, nullable=False)
    confidence = Column(Float, nullable=False)

    # Link to the ResearchChunk IDs that support/contest this claim
    evidence_references = Column(JSONB, nullable=False, server_default="[]")

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    script = relationship("Script", back_populates="claims")


class Asset(Base):
    """
    Step 7: Multi-modal asset generation (Veo clips, Lyria audio, Charts).
    """

    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_job_type", "job_id", "asset_type"),
        {"schema": "factory"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("factory.render_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    asset_type = Column(AssetTypeEnum, nullable=False)
    url_or_path = Column(String, nullable=False)

    # Metadata for rendering (e.g., start_time, end_time, SynthID watermarks)
    render_meta = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    job = relationship("RenderJob", back_populates="assets")


# ==========================================
# 3. POSTGRESQL TRIGGERS (Automated Maintenance)
# ==========================================
# This ensures updated_at is ALWAYS updated on row modification at the DB level,
# protecting against application-level ORM oversights.

trigger_ddl = DDL("""
CREATE OR REPLACE FUNCTION factory.update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_render_jobs_modtime
    BEFORE UPDATE ON factory.render_jobs
    FOR EACH ROW EXECUTE PROCEDURE factory.update_modified_column();

CREATE TRIGGER update_scripts_modtime
    BEFORE UPDATE ON factory.scripts
    FOR EACH ROW EXECUTE PROCEDURE factory.update_modified_column();
""")

event.listen(
    Base.metadata, "after_create", trigger_ddl.execute_if(dialect="postgresql")
)
