"""add CAROUSEL_SLIDE to asset_type enum

Revision ID: 25b1c3d4e5f6
Revises: 8b89dd041b5f
Create Date: 2026-05-20 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "25b1c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "8b89dd041b5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE factory.asset_type ADD VALUE IF NOT EXISTS 'CAROUSEL_SLIDE'"
    )


def downgrade() -> None:
    pass
