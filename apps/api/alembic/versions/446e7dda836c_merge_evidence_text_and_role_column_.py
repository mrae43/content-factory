"""merge evidence_text and role column heads

Revision ID: 446e7dda836c
Revises: a1b2c3d4e5f6, a2b1c3d4e5f6
Create Date: 2026-05-15 15:14:42.893198

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "446e7dda836c"
down_revision: Union[str, Sequence[str], None] = ("a1b2c3d4e5f6", "a2b1c3d4e5f6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
