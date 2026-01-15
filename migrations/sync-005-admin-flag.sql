-- Add is_admin flag to users table
-- This replaces hardcoded email domain checks for admin access

ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;

-- Set existing admin user (from ADMIN_EMAIL env var) as admin
-- This will be handled by the application on startup
