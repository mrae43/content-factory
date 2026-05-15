"""add role column to scripts, make format_type nullable

Revision ID: a2b1c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-05-15 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a2b1c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add role column
    op.add_column(
        "scripts",
        sa.Column("role", sa.String(), nullable=False, server_default="master"),
        schema="factory",
    )
    # 2. Create check constraint
    op.create_check_constraint(
        "ck_scripts_role",
        "scripts",
        "role IN ('master', 'format')",
        schema="factory",
    )
    # 3. Backfill existing master scripts: format_type='VIDEO', format_payload=NULL
    op.execute(
        "UPDATE factory.scripts SET role = 'master' "
        "WHERE format_type = 'VIDEO' AND format_payload IS NULL"
    )
    # 4. Backfill existing format scripts: format_payload IS NOT NULL
    op.execute(
        "UPDATE factory.scripts SET role = 'format' "
        "WHERE format_payload IS NOT NULL"
    )
    # 5. Null out format_type for existing master scripts (Decision 2: master has no format)
    op.execute(
        "UPDATE factory.scripts SET format_type = NULL "
        "WHERE role = 'master' AND format_type = 'VIDEO'"
    )
    # 6. Drop old check constraint on format_type
    op.drop_constraint("ck_scripts_format_type", "scripts", schema="factory", type_="check")
    # 7. Alter format_type to nullable
    op.alter_column("scripts", "format_type", nullable=True, schema="factory")
    # 8. Create new check constraint allowing NULL for master scripts
    op.create_check_constraint(
        "ck_scripts_format_type",
        "scripts",
        "format_type IS NULL OR format_type IN ('VIDEO', 'BLOG', 'CAROUSEL')",
        schema="factory",
    )


def downgrade() -> None:
    op.drop_constraint("ck_scripts_format_type", "scripts", schema="factory", type_="check")
    op.alter_column("scripts", "format_type", nullable=False, server_default="VIDEO", schema="factory")
    op.create_check_constraint(
        "ck_scripts_format_type",
        "scripts",
        "format_type IN ('VIDEO', 'BLOG', 'CAROUSEL')",
        schema="factory",
    )
    op.drop_constraint("ck_scripts_role", "scripts", schema="factory", type_="check")
    op.drop_column("scripts", "role", schema="factory")
