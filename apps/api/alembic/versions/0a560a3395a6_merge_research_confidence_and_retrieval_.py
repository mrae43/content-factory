"""merge research_confidence and retrieval_status heads

Revision ID: 0a560a3395a6
Revises: f3b4c5d6e7f8, f4b5c6d7e8f9
Create Date: 2026-05-19 15:49:01.250855

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0a560a3395a6"
down_revision: Union[str, Sequence[str], None] = ("f3b4c5d6e7f8", "f4b5c6d7e8f9")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
