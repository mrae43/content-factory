"""add_short_format_enums

Revision ID: b2c3d4e5f6a7
Revises: a4388489c303
Create Date: 2026-06-02 14:56:05.899021

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a4388489c303"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add SHORT format enum values.

    ALTER TYPE ... ADD VALUE cannot run inside a transaction block.
    We use autocommit_block to execute these DDL statements outside
    the Alembic transaction.
    """
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE factory.job_status ADD VALUE 'COMPOSITION'")
        op.execute("ALTER TYPE factory.format_job_status ADD VALUE 'COMPOSITION'")
        op.execute("ALTER TYPE factory.asset_type ADD VALUE 'SHORT_VIDEO_CLIP'")
        op.execute("ALTER TYPE factory.asset_type ADD VALUE 'SHORT_STILL_IMAGE'")
        op.execute("ALTER TYPE factory.asset_type ADD VALUE 'VOCAL_ALIGNMENT'")
        op.execute("ALTER TYPE factory.asset_type ADD VALUE 'SHORT_COMPOSED_VIDEO'")


def downgrade() -> None:
    """Downgrade schema.

    PostgreSQL does not support removing values from an ENUM via ALTER TYPE.
    Removing enum values is complex (requires type recreation + column
    migration) and is deferred. Added values are harmless for backward
    compatibility.
    """
    pass
