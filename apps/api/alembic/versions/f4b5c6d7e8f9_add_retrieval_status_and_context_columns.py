"""Add RETRIEVAL status, assembled_context and retrieval_retry_count columns

Revision ID: f4b5c6d7e8f9
Revises: f2b3c4d5e6f7
Create Date: 2026-05-19 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "f4b5c6d7e8f9"
down_revision: Union[str, Sequence[str], None] = "f2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE factory.job_status ADD VALUE 'RETRIEVAL'")

    op.add_column(
        "render_jobs",
        sa.Column("assembled_context", JSONB(), nullable=True),
        schema="factory",
    )
    op.add_column(
        "render_jobs",
        sa.Column(
            "retrieval_retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        schema="factory",
    )


def downgrade() -> None:
    op.drop_column("render_jobs", "retrieval_retry_count", schema="factory")
    op.drop_column("render_jobs", "assembled_context", schema="factory")
    # NOTE: ALTER TYPE ... REMOVE VALUE is not supported by PostgreSQL,
    # so RETRIEVAL remains in the enum. This is safe as it's never referenced.
