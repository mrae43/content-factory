"""Drop research_chunks.job_id FK to render_jobs

The script pipeline (ScriptJob) also writes to research_chunks with its own
job_id, but the FK only references render_jobs. The column is already nullable
(migration 2796db25c720) and used as a loose scoping identifier for semantic_search.
Drop the FK to allow both pipelines (RenderJob + ScriptJob) to share the table.

Revision ID: 20260604110300
Revises: 20260603215257
Create Date: 2026-06-04 11:03:00

"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260604110300"
down_revision: Union[str, Sequence[str], None] = "20260603215257"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("research_chunks_job_id_fkey"),
        "research_chunks",
        schema="factory",
        type_="foreignkey",
    )


def downgrade() -> None:
    op.create_foreign_key(
        op.f("research_chunks_job_id_fkey"),
        "research_chunks",
        "render_jobs",
        ["job_id"],
        ["id"],
        source_schema="factory",
        referent_schema="factory",
        ondelete="SET NULL",
    )
