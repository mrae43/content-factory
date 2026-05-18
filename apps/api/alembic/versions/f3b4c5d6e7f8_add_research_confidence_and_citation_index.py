"""add research_confidence and citation_index to render_jobs

Revision ID: f3b4c5d6e7f8
Revises: f2b3c4d5e6f7
Create Date: 2026-05-18 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "f3b4c5d6e7f8"
down_revision: Union[str, Sequence[str], None] = "f2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "render_jobs",
        sa.Column("research_confidence", sa.Float(), nullable=True),
        schema="factory",
    )
    op.add_column(
        "render_jobs",
        sa.Column("citation_index", JSONB(), nullable=True),
        schema="factory",
    )


def downgrade() -> None:
    op.drop_column("render_jobs", "research_confidence", schema="factory")
    op.drop_column("render_jobs", "citation_index", schema="factory")
