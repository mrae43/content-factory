"""drop_strict_compliance_mode

Revision ID: 8b89dd041b5f
Revises: 0a560a3395a6
Create Date: 2026-05-19 23:04:17.746942

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b89dd041b5f'
down_revision: Union[str, Sequence[str], None] = '0a560a3395a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('render_jobs', 'strict_compliance_mode', schema='factory')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        'render_jobs',
        sa.Column('strict_compliance_mode', sa.BOOLEAN(), autoincrement=False, nullable=False, server_default=sa.text('true')),
        schema='factory',
    )
