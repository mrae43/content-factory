"""add format columns

Revision ID: e8f2a1b3c4d5
Revises: 7c3881a3c229
Create Date: 2026-05-09 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "e8f2a1b3c4d5"
down_revision: Union[str, Sequence[str], None] = "7c3881a3c229"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "render_jobs",
        sa.Column("format_type", sa.String(), nullable=False, server_default="all"),
        schema="factory",
    )
    op.add_column(
        "render_jobs",
        sa.Column("platform", sa.String(), nullable=True),
        schema="factory",
    )
    op.add_column(
        "scripts",
        sa.Column("format_type", sa.String(), nullable=False, server_default="VIDEO"),
        schema="factory",
    )
    op.add_column(
        "scripts",
        sa.Column("format_payload", JSONB(), nullable=True),
        schema="factory",
    )
    op.create_check_constraint(
        "ck_render_jobs_format_type",
        "render_jobs",
        "format_type IN ('all', 'video', 'blog', 'carousel')",
        schema="factory",
    )
    op.create_check_constraint(
        "ck_render_jobs_platform",
        "render_jobs",
        "platform IS NULL OR platform IN ('twitter', 'linkedin', 'instagram', 'youtube')",
        schema="factory",
    )
    op.create_check_constraint(
        "ck_scripts_format_type",
        "scripts",
        "format_type IN ('VIDEO', 'BLOG', 'CAROUSEL')",
        schema="factory",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_scripts_format_type", "scripts", schema="factory", type_="check"
    )
    op.drop_constraint(
        "ck_render_jobs_platform", "render_jobs", schema="factory", type_="check"
    )
    op.drop_constraint(
        "ck_render_jobs_format_type", "render_jobs", schema="factory", type_="check"
    )
    op.drop_column("scripts", "format_payload", schema="factory")
    op.drop_column("scripts", "format_type", schema="factory")
    op.drop_column("render_jobs", "platform", schema="factory")
    op.drop_column("render_jobs", "format_type", schema="factory")
