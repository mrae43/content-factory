"""add optimization_history to scripts
Revision ID: b1c2d3e4f5a6
Revises: 22520681da2b
Create Date: 2026-05-28 12:00:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "2695ce2f5280"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "scripts",
        sa.Column(
            "optimization_history",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema="factory",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("scripts", "optimization_history", schema="factory")
