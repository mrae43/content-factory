"""add FORMATTING status to job_status enum

Revision ID: f1a2b3c4d5e6
Revises: e8f2a1b3c4d5
Create Date: 2026-05-09 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e8f2a1b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE factory.job_status ADD VALUE IF NOT EXISTS 'FORMATTING'")


def downgrade() -> None:
    pass
