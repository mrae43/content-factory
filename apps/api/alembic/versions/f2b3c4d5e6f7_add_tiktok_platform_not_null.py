"""Add tiktok to platform, make platform NOT NULL

Revision ID: f2b3c4d5e6f7
Revises: a3b4c5d6e7f8
Create Date: 2026-05-18 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2b3c4d5e6f7"
down_revision: Union[str, Sequence[str], None] = "a3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_render_jobs_format_type", "render_jobs", schema="factory", type_="check"
    )
    op.drop_constraint(
        "ck_render_jobs_platform", "render_jobs", schema="factory", type_="check"
    )

    op.alter_column("render_jobs", "platform", nullable=False, schema="factory")

    op.create_check_constraint(
        "ck_render_jobs_format_type",
        "render_jobs",
        "format_type IN ('all', 'video', 'blog', 'carousel')",
        schema="factory",
    )
    op.create_check_constraint(
        "ck_render_jobs_platform",
        "render_jobs",
        "platform IN ('twitter', 'linkedin', 'instagram', 'youtube', 'tiktok')",
        schema="factory",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_render_jobs_platform", "render_jobs", schema="factory", type_="check"
    )

    op.alter_column("render_jobs", "platform", nullable=True, schema="factory")

    op.create_check_constraint(
        "ck_render_jobs_platform",
        "render_jobs",
        "platform IS NULL OR platform IN ('twitter', 'linkedin', 'instagram', 'youtube')",
        schema="factory",
    )
    op.create_check_constraint(
        "ck_render_jobs_format_type",
        "render_jobs",
        "format_type IN ('all', 'video', 'blog', 'carousel')",
        schema="factory",
    )
