-- ContextFS Sync Service - Multi-tenant Migration
-- This migration adds user_id columns for multi-tenant isolation
-- and creates missing auth/billing tables

-- =============================================================================
-- Add user_id columns for multi-tenant isolation
-- =============================================================================

-- Add user_id to devices table
ALTER TABLE devices ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_id);

-- Add user_id to memories table
ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);

-- Add user_id to sessions table
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- =============================================================================
-- Subscriptions Table
-- User subscription management
-- =============================================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tier TEXT DEFAULT 'free',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    device_limit INTEGER DEFAULT 3,
    memory_limit INTEGER DEFAULT 10000,
    status TEXT DEFAULT 'active',
    current_period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe_customer ON subscriptions(stripe_customer_id);

-- =============================================================================
-- Usage Table
-- User usage tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS usage (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    device_count INTEGER DEFAULT 0,
    memory_count INTEGER DEFAULT 0,
    last_sync_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Password Reset Tokens Table
-- For email-based password reset
-- =============================================================================
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_password_reset_user ON password_reset_tokens(user_id);

-- =============================================================================
-- Update triggers for new tables (use DROP IF EXISTS + CREATE for idempotency)
-- =============================================================================
DROP TRIGGER IF EXISTS subscriptions_updated_at ON subscriptions;
CREATE TRIGGER subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS usage_updated_at ON usage;
CREATE TRIGGER usage_updated_at
    BEFORE UPDATE ON usage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Grants for new tables
-- =============================================================================
GRANT ALL PRIVILEGES ON subscriptions TO contextfs;
GRANT ALL PRIVILEGES ON usage TO contextfs;
GRANT ALL PRIVILEGES ON password_reset_tokens TO contextfs;
