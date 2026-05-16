"""add evidence_text_inline, hedge_required, hedge_index columns

Revision ID: a3b4c5d6e7f8
Revises: 446e7dda836c
Create Date: 2026-05-16 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "a3b4c5d6e7f8"
down_revision: Union[str, Sequence[str], None] = "446e7dda836c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Fix 3 — Inline evidence text
    op.add_column(
        "fact_check_claims",
        sa.Column("evidence_text_inline", JSONB(), nullable=False, server_default="[]"),
        schema="factory",
    )
    # Fix 1 — Hedge mechanism
    op.add_column(
        "fact_check_claims",
        sa.Column("hedge_required", sa.Boolean(), nullable=False, server_default="false"),
        schema="factory",
    )
    op.add_column(
        "render_jobs",
        sa.Column("hedge_index", JSONB(), nullable=True),
        schema="factory",
    )


def downgrade() -> None:
    op.drop_column("render_jobs", "hedge_index", schema="factory")
    op.drop_column("fact_check_claims", "hedge_required", schema="factory")
    op.drop_column("fact_check_claims", "evidence_text_inline", schema="factory")
