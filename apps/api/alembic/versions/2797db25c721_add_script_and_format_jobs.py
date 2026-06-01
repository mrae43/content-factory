"""add_script_and_format_jobs

Revision ID: 2797db25c721
Revises: 2796db25c720
Create Date: 2026-06-01 12:00:00.000000

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
    # Create enum types
    script_job_status = postgresql.ENUM(
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
        create_type=True,
    )
    script_job_status.create(op.get_bind())

    format_job_status = postgresql.ENUM(
        "PENDING",
        "FORMATTING",
        "ASSET_GENERATION",
        "COMPLETED",
        "FAILED",
        "HUMAN_REVIEW_NEEDED",
        name="format_job_status",
        schema="factory",
        create_type=True,
    )
    format_job_status.create(op.get_bind())

    # Create script_jobs table
    op.create_table(
        "script_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("user_reference", sa.Text(), nullable=True),
        sa.Column(
            "source_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "story_directives", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "status", script_job_status, nullable=False, server_default="PENDING"
        ),
        sa.Column("refined_context", sa.Text(), nullable=True),
        sa.Column(
            "assembled_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("script_content", sa.Text(), nullable=True),
        sa.Column("claims", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "working_memory", postgresql.JSONB(astext_type=sa.Text()), nullable=True
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

    # Create format_jobs table
    op.create_table(
        "format_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_job_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("format_type", sa.Text(), nullable=False),
        sa.Column(
            "status", format_job_status, nullable=False, server_default="PENDING"
        ),
        sa.Column("script_content", sa.Text(), nullable=True),
        sa.Column("claims", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("refined_context", sa.Text(), nullable=True),
        sa.Column(
            "story_directives", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "hedge_index", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "epistemic_ledger", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "format_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True
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
            name="fk_format_jobs_source_job_id",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "source_job_id",
            "platform",
            "format_type",
            name="uq_format_jobs_source_platform_type",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="factory",
    )

    # Partial index for active queue polling
    op.create_index(
        "ix_format_jobs_active_queue",
        "format_jobs",
        ["status"],
        schema="factory",
        postgresql_where=sa.text("status NOT IN ('COMPLETED', 'FAILED')"),
    )

    # Triggers for updated_at
    op.execute("""
        CREATE TRIGGER update_script_jobs_modtime
            BEFORE UPDATE ON factory.script_jobs
            FOR EACH ROW EXECUTE FUNCTION factory.update_modified_column();
    """)
    op.execute("""
        CREATE TRIGGER update_format_jobs_modtime
            BEFORE UPDATE ON factory.format_jobs
            FOR EACH ROW EXECUTE FUNCTION factory.update_modified_column();
    """)


def downgrade() -> None:
    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS update_format_jobs_modtime ON factory.format_jobs;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_script_jobs_modtime ON factory.script_jobs;"
    )

    # Drop index
    op.drop_index(
        "ix_format_jobs_active_queue",
        table_name="format_jobs",
        schema="factory",
        postgresql_where=sa.text("status NOT IN ('COMPLETED', 'FAILED')"),
    )

    # Drop tables (must drop before enum types)
    op.drop_table("format_jobs", schema="factory")
    op.drop_table("script_jobs", schema="factory")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS factory.format_job_status;")
    op.execute("DROP TYPE IF EXISTS factory.script_job_status;")
