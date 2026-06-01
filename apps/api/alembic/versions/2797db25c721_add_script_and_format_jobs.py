"""add_script_and_format_jobs

Revision ID: 2797db25c721
Revises: 2796db25c720
Create Date: 2026-06-01

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "2797db25c721"
down_revision: Union[str, Sequence[str], None] = "2796db25c720"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "script_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("user_reference", sa.Text(), nullable=True),
        sa.Column(
            "source_urls",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "story_directives",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
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
            ),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column("refined_context", sa.Text(), nullable=True),
        sa.Column(
            "assembled_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("script_content", sa.Text(), nullable=True),
        sa.Column("claims", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "working_memory",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "hedge_index", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("error_log", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="factory",
    )
    op.create_table(
        "format_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_job_id", sa.UUID(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("format_type", sa.Text(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING",
                "FORMATTING",
                "ASSET_GENERATION",
                "COMPLETED",
                "FAILED",
                "HUMAN_REVIEW_NEEDED",
                name="format_job_status",
                schema="factory",
            ),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column("script_content", sa.Text(), nullable=True),
        sa.Column("claims", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("refined_context", sa.Text(), nullable=True),
        sa.Column(
            "story_directives",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "hedge_index", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "epistemic_ledger",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "format_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("final_video_url", sa.Text(), nullable=True),
        sa.Column("error_log", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["source_job_id"],
            ["factory.script_jobs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_job_id",
            "platform",
            "format_type",
            name="uq_format_jobs_source_platform_type",
        ),
        schema="factory",
    )
    op.create_index(
        "ix_format_jobs_active_queue",
        "format_jobs",
        ["status"],
        schema="factory",
        postgresql_where=sa.text("status NOT IN ('COMPLETED', 'FAILED')"),
    )
    op.execute("""
    CREATE TRIGGER update_script_jobs_modtime
        BEFORE UPDATE ON factory.script_jobs
        FOR EACH ROW EXECUTE PROCEDURE factory.update_modified_column();
    """)
    op.execute("""
    CREATE TRIGGER update_format_jobs_modtime
        BEFORE UPDATE ON factory.format_jobs
        FOR EACH ROW EXECUTE PROCEDURE factory.update_modified_column();
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "DROP TRIGGER IF EXISTS update_format_jobs_modtime ON factory.format_jobs;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_script_jobs_modtime ON factory.script_jobs;"
    )
    op.drop_index(
        "ix_format_jobs_active_queue", table_name="format_jobs", schema="factory"
    )
    op.drop_table("format_jobs", schema="factory")
    op.drop_table("script_jobs", schema="factory")
    op.execute("DROP TYPE IF EXISTS factory.format_job_status;")
    op.execute("DROP TYPE IF EXISTS factory.script_job_status;")
