"""Fix subscription tier limits.

Revision ID: 004
Revises: 003
Create Date: 2024-01-04 00:00:00.000000

Corrects subscription tier limits to official values:
| Tier       | device_limit | memory_limit |
|------------|--------------|--------------|
| free       | 2            | 5,000        |
| pro        | 5            | 50,000       |
| team       | 10           | unlimited(-1)|
| enterprise | unlimited(-1)| unlimited(-1)|
| admin      | unlimited(-1)| unlimited(-1)|
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fix column defaults for NEW subscriptions
    op.execute("ALTER TABLE subscriptions ALTER COLUMN device_limit SET DEFAULT 2;")
    op.execute("ALTER TABLE subscriptions ALTER COLUMN memory_limit SET DEFAULT 5000;")

    # Fix existing subscription data
    op.execute(
        "UPDATE subscriptions SET device_limit = 2, memory_limit = 5000 " "WHERE tier = 'free';"
    )
    op.execute(
        "UPDATE subscriptions SET device_limit = 5, memory_limit = 50000 " "WHERE tier = 'pro';"
    )
    op.execute(
        "UPDATE subscriptions SET device_limit = 10, memory_limit = -1 " "WHERE tier = 'team';"
    )
    op.execute(
        "UPDATE subscriptions SET device_limit = -1, memory_limit = -1 "
        "WHERE tier = 'enterprise';"
    )
    op.execute(
        "UPDATE subscriptions SET device_limit = -1, memory_limit = -1 " "WHERE tier = 'admin';"
    )

    # Set default seats for team tier subscriptions
    op.execute("UPDATE subscriptions SET seats_included = 5 WHERE tier = 'team';")


def downgrade() -> None:
    # Revert to old defaults
    op.execute("ALTER TABLE subscriptions ALTER COLUMN device_limit SET DEFAULT 3;")
    op.execute("ALTER TABLE subscriptions ALTER COLUMN memory_limit SET DEFAULT 10000;")

    # Note: We can't fully revert the data changes as we don't know original values
