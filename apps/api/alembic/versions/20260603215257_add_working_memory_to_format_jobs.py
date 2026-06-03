"""Add working_memory column to format_jobs

Revision ID: 20260603215257
Revises: b2c3d4e5f6a7
Create Date: 2026-06-03 21:52:57

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "20260603215257"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add working_memory column to format_jobs."""
    op.add_column(
        "format_jobs",
        sa.Column("working_memory", JSONB(), nullable=True),
        schema="factory",
    )


def downgrade() -> None:
    """Remove working_memory column from format_jobs."""
    op.drop_column("format_jobs", "working_memory", schema="factory")
