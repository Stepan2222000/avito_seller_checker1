"""
Database schema definitions for avito_seller_checker.
Contains DDL queries for creating tables and indexes.
"""

# Tasks table - stores URLs to be processed
CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    payload JSONB DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'in_progress', 'completed', 'failed', 'abandoned')),
    attempt INTEGER NOT NULL DEFAULT 1,
    max_attempts INTEGER NOT NULL DEFAULT 4,
    last_proxy TEXT,
    enqueued_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
"""

# Results table - stores validation results
CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    seller_url TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('passed', 'failed', 'error')),
    reason TEXT,
    confidence REAL,
    flags TEXT[] DEFAULT '{}',
    items JSONB DEFAULT '[]',
    llm_attempts INTEGER DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

# Proxies table - stores proxy addresses and their status
CREATE_PROXIES_TABLE = """
CREATE TABLE IF NOT EXISTS proxies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    address TEXT NOT NULL UNIQUE,
    is_blocked BOOLEAN NOT NULL DEFAULT FALSE,
    blocked_reason TEXT,
    blocked_at TIMESTAMP,
    in_use BOOLEAN NOT NULL DEFAULT FALSE,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

# Indexes for performance optimization
# Note: UNIQUE constraints automatically create indexes, so we don't need separate indexes for:
# - tasks.url (has UNIQUE constraint)
# - proxies.address (has UNIQUE constraint)
CREATE_TASKS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_enqueued_at ON tasks(enqueued_at);
"""

CREATE_RESULTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_results_task_id ON results(task_id);
CREATE INDEX IF NOT EXISTS idx_results_verdict ON results(verdict);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at);
"""

CREATE_PROXIES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_proxies_is_blocked ON proxies(is_blocked);
CREATE INDEX IF NOT EXISTS idx_proxies_in_use ON proxies(in_use);
"""

# Trigger to update updated_at timestamp
CREATE_UPDATED_AT_TRIGGER = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_proxies_updated_at ON proxies;
CREATE TRIGGER update_proxies_updated_at
    BEFORE UPDATE ON proxies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""

# Combined initialization query
INIT_DATABASE = f"""
{CREATE_TASKS_TABLE}
{CREATE_RESULTS_TABLE}
{CREATE_PROXIES_TABLE}
{CREATE_TASKS_INDEXES}
{CREATE_RESULTS_INDEXES}
{CREATE_PROXIES_INDEXES}
{CREATE_UPDATED_AT_TRIGGER}
"""
