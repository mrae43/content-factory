"""add evidence_text column to fact_check_claims

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-05-09 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "fact_check_claims",
        sa.Column("evidence_text", sa.Text(), nullable=True),
        schema="factory",
    )


def downgrade() -> None:
    op.drop_column("fact_check_claims", "evidence_text", schema="factory")
