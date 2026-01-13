"""Email service for ContextFS using Mailgun.

Sends welcome emails and password reset links to users.
"""

import hashlib

# Mailgun configuration (from environment)
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "appmail.magnetonlabs.com")
MAILGUN_BASE_URL = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}"

# App URLs - use environment variable or default to localhost for dev
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:3000")


async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str | None = None,
) -> bool:
    """Send an email via Mailgun.

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML body content
        text_content: Plain text body (optional, will strip HTML if not provided)

    Returns:
        True if email sent successfully, False otherwise
    """
    if not MAILGUN_API_KEY:
        print(f"MAILGUN_API_KEY not configured. Email to {to_email} not sent.")
        return False

    if not text_content:
        # Strip HTML tags for plain text version
        import re

        text_content = re.sub(r"<[^>]+>", "", html_content)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{MAILGUN_BASE_URL}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={
                    "from": "ContextFS <noreply@appmail.magnetonlabs.com>",
                    "to": to_email,
                    "subject": subject,
                    "text": text_content,
                    "html": html_content,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
            return False


def generate_reset_token() -> tuple[str, str]:
    """Generate a password reset token.

    Returns:
        Tuple of (raw_token, token_hash)
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


async def create_password_reset_token(
    session,
    user_id: str,
    expires_hours: int = 24,
) -> str:
    """Create a password reset token in the database.

    Args:
        session: Database session
        user_id: User ID to create token for
        expires_hours: Hours until token expires (default 24)

    Returns:
        Raw token string (to send in email)
    """
    from service.db.models import PasswordResetToken

    raw_token, token_hash = generate_reset_token()

    # Delete any existing tokens for this user
    from sqlalchemy import delete

    await session.execute(delete(PasswordResetToken).where(PasswordResetToken.user_id == user_id))

    # Create new token
    reset_token = PasswordResetToken(
        id=str(uuid4()),
        user_id=user_id,
        token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours),
    )
    session.add(reset_token)
    await session.flush()

    return raw_token


async def send_welcome_email(
    to_email: str,
    user_name: str | None,
    reset_token: str,
) -> bool:
    """Send welcome email to new user with password setup link.

    Args:
        to_email: User's email address
        user_name: User's name (or None)
        reset_token: Password reset token

    Returns:
        True if sent successfully
    """
    name = user_name or to_email.split("@")[0]
    reset_url = f"{APP_BASE_URL}/reset-password?token={reset_token}"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to ContextFS</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #09090b; color: #fafafa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Logo/Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #8b5cf6; font-size: 32px; margin: 0;">ContextFS</h1>
            <p style="color: #a1a1aa; margin-top: 8px;">AI Memory That Follows You</p>
        </div>

        <!-- Main Content -->
        <div style="background-color: #18181b; border-radius: 12px; padding: 32px; border: 1px solid #27272a;">
            <h2 style="color: #fafafa; font-size: 24px; margin: 0 0 16px 0;">Welcome, {name}!</h2>

            <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 24px 0;">
                Your ContextFS account has been created. To get started, please set up your password by clicking the button below.
            </p>

            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{reset_url}"
                   style="display: inline-block; padding: 14px 32px; background: linear-gradient(to right, #8b5cf6, #7c3aed); color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                    Set Up Your Password
                </a>
            </div>

            <p style="color: #71717a; font-size: 14px; line-height: 1.6; margin: 24px 0 0 0;">
                This link will expire in 24 hours. If you didn't request this account, you can safely ignore this email.
            </p>
        </div>

        <!-- What's Next -->
        <div style="margin-top: 32px; padding: 24px; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
            <h3 style="color: #fafafa; font-size: 18px; margin: 0 0 16px 0;">What's Next?</h3>
            <ul style="color: #a1a1aa; font-size: 14px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Set up your password to secure your account</li>
                <li>Install the ContextFS CLI: <code style="background: #27272a; padding: 2px 6px; border-radius: 4px;">pip install contextfs</code></li>
                <li>Login to the cloud: <code style="background: #27272a; padding: 2px 6px; border-radius: 4px;">contextfs cloud login</code></li>
                <li>Start syncing your AI memories across devices!</li>
            </ul>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding-top: 24px; border-top: 1px solid #27272a;">
            <p style="color: #71717a; font-size: 12px; margin: 0;">
                &copy; 2024 ContextFS. All rights reserved.<br>
                <a href="{APP_BASE_URL}" style="color: #8b5cf6; text-decoration: none;">contextfs.ai</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

    text_content = f"""
Welcome to ContextFS, {name}!

Your account has been created. To get started, please set up your password by visiting:

{reset_url}

This link will expire in 24 hours.

What's Next?
- Set up your password to secure your account
- Install the ContextFS CLI: pip install contextfs
- Login to the cloud: contextfs cloud login
- Start syncing your AI memories across devices!

If you didn't request this account, you can safely ignore this email.

---
ContextFS - AI Memory That Follows You
https://contextfs.ai
"""

    return await send_email(
        to_email=to_email,
        subject="Welcome to ContextFS - Set Up Your Password",
        html_content=html_content,
        text_content=text_content,
    )


async def send_password_reset_email(
    to_email: str,
    user_name: str | None,
    reset_token: str,
) -> bool:
    """Send password reset email to existing user.

    Args:
        to_email: User's email address
        user_name: User's name (or None)
        reset_token: Password reset token

    Returns:
        True if sent successfully
    """
    name = user_name or to_email.split("@")[0]
    reset_url = f"{APP_BASE_URL}/reset-password?token={reset_token}"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Your Password - ContextFS</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #09090b; color: #fafafa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Logo/Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #8b5cf6; font-size: 32px; margin: 0;">ContextFS</h1>
        </div>

        <!-- Main Content -->
        <div style="background-color: #18181b; border-radius: 12px; padding: 32px; border: 1px solid #27272a;">
            <h2 style="color: #fafafa; font-size: 24px; margin: 0 0 16px 0;">Reset Your Password</h2>

            <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 24px 0;">
                Hi {name}, we received a request to reset your password. Click the button below to choose a new password.
            </p>

            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{reset_url}"
                   style="display: inline-block; padding: 14px 32px; background: linear-gradient(to right, #8b5cf6, #7c3aed); color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                    Reset Password
                </a>
            </div>

            <p style="color: #71717a; font-size: 14px; line-height: 1.6; margin: 24px 0 0 0;">
                This link will expire in 24 hours. If you didn't request a password reset, you can safely ignore this email - your password will remain unchanged.
            </p>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding-top: 24px; border-top: 1px solid #27272a;">
            <p style="color: #71717a; font-size: 12px; margin: 0;">
                &copy; 2024 ContextFS. All rights reserved.<br>
                <a href="{APP_BASE_URL}" style="color: #8b5cf6; text-decoration: none;">contextfs.ai</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

    text_content = f"""
Reset Your Password

Hi {name},

We received a request to reset your password. Visit the link below to choose a new password:

{reset_url}

This link will expire in 24 hours. If you didn't request a password reset, you can safely ignore this email.

---
ContextFS
https://contextfs.ai
"""

    return await send_email(
        to_email=to_email,
        subject="Reset Your Password - ContextFS",
        html_content=html_content,
        text_content=text_content,
    )
