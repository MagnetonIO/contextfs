#!/usr/bin/env python3
"""Seed admin user for ContextFS Sync Service.

Usage:
    python -m service.scripts.seed_admin --email admin@example.com --password <password>

Or with env vars:
    ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD=secret python -m service.scripts.seed_admin
"""

import argparse
import asyncio
import hashlib
import os
import sys
from uuid import uuid4

import aiosqlite

ADMIN_USER_ID = "admin-user-00000000-0000-0000-0000-000000000001"


def hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate API key. Returns (full_key, prefix)."""
    import base64
    import secrets

    random_bytes = secrets.token_bytes(32)
    key_part = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
    full_key = f"ctxfs_{key_part}"
    return full_key, key_part[:8]


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def seed_admin(db_path: str, email: str, password: str, api_key: str | None = None):
    """Create admin user with API key."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Check if admin exists
        cursor = await db.execute("SELECT id FROM users WHERE id = ?", (ADMIN_USER_ID,))
        if await cursor.fetchone():
            print(f"Admin user already exists (id: {ADMIN_USER_ID})")

            # Check for existing API key
            cursor = await db.execute(
                "SELECT key_prefix FROM api_keys WHERE user_id = ? AND is_active = 1",
                (ADMIN_USER_ID,),
            )
            key = await cursor.fetchone()
            if key:
                print(f"Admin has active API key with prefix: {key['key_prefix']}...")
                return
        else:
            # Create admin user
            password_hash = hash_password(password)
            await db.execute(
                """
                INSERT INTO users (id, email, name, provider, password_hash, email_verified, created_at)
                VALUES (?, ?, ?, ?, ?, 1, datetime('now'))
                """,
                (ADMIN_USER_ID, email, "Admin", "system", password_hash),
            )
            print(f"Created admin user: {email}")

        # Generate or use provided API key
        if api_key and api_key.startswith("ctxfs_"):
            full_key = api_key
            key_prefix = api_key[6:14]
        else:
            full_key, key_prefix = generate_api_key()

        key_hash = hash_api_key(full_key)
        key_id = str(uuid4())

        await db.execute(
            """
            INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, datetime('now'))
            """,
            (key_id, ADMIN_USER_ID, "Admin API Key", key_hash, key_prefix),
        )
        await db.commit()

        print("=" * 60)
        print("ADMIN CREDENTIALS (save these!):")
        print(f"  Email:    {email}")
        print(f"  Password: {password}")
        print(f"  API Key:  {full_key}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Seed admin user for ContextFS")
    parser.add_argument("--email", default=os.environ.get("ADMIN_EMAIL"), help="Admin email")
    parser.add_argument(
        "--password", default=os.environ.get("ADMIN_PASSWORD"), help="Admin password"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ADMIN_API_KEY"), help="Optional: specific API key"
    )
    parser.add_argument(
        "--db", default=os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db"), help="Database path"
    )

    args = parser.parse_args()

    if not args.email or not args.password:
        print(
            "Error: --email and --password required (or set ADMIN_EMAIL, ADMIN_PASSWORD env vars)"
        )
        sys.exit(1)

    asyncio.run(seed_admin(args.db, args.email, args.password, args.api_key))


if __name__ == "__main__":
    main()
