-- ============================================================================
-- Database Schema for OVOS Plugin Arena
-- Corrected and production-safe version
-- PostgreSQL compatible
-- ============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUM DEFINITIONS
-- ============================================================================

-- Authorization roles for authenticated users
CREATE TYPE user_role_enum AS ENUM (
    'admin',  -- Can register plugins and manage system
    'voter'   -- Can participate in battles and vote
);

-- Supported plugin modalities
CREATE TYPE modality_enum AS ENUM (
    'tts',        -- Text-to-Speech
    'stt',        -- Speech-to-Text
    'wake_word',  -- Wake Word Detection
    'intent'      -- Intent Classification
);

-- Battle execution lifecycle
CREATE TYPE battle_status_enum AS ENUM (
    'PENDING',  -- Created, awaiting worker
    'RUNNING',  -- Worker executing plugins
    'READY',    -- Outputs ready for voting
    'FAILED'    -- Failed after retries or fatal error
);

-- Vote outcomes
CREATE TYPE vote_result_enum AS ENUM (
    'candidate_1',
    'candidate_2',
    'tie',
    'both_wrong'
);

-- ============================================================================
-- USERS TABLE
-- ============================================================================

-- Stores authenticated users only
CREATE TABLE users (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Login credentials
    email VARCHAR(255) NOT NULL,
    password_hash TEXT NOT NULL,

    -- Authorization role
    role user_role_enum NOT NULL DEFAULT 'voter',

    -- Account state
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Optional profile data
    full_name VARCHAR(255),

    -- Audit timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Enforce unique emails
CREATE UNIQUE INDEX ix_users_email ON users(email);

-- Speed up role checks
CREATE INDEX ix_users_role ON users(role);

-- ============================================================================
-- PLUGINS TABLE (IDENTITY + METADATA)
-- ============================================================================

-- Represents a plugin as a product/identity
CREATE TABLE plugins (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Unique module or package name
    plugin_name TEXT NOT NULL UNIQUE,

    -- Human-readable name
    display_name TEXT NOT NULL,

    -- Author or organization
    author TEXT,

    -- Description shown on profile page
    description TEXT,

    -- Informational list of supported modalities
    supported_modalities modality_enum[] NOT NULL,

    -- External references
    homepage_url TEXT,
    license TEXT,
    tags TEXT[],

    -- Free-form metadata (docs links, notes, etc.)
    metadata JSONB,

    -- Audit timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- COMPETITORS TABLE (PLUGIN + CONFIG + MODALITY)
-- ============================================================================

-- Represents a concrete evaluation unit
CREATE TABLE competitors (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent plugin
    plugin_id UUID NOT NULL REFERENCES plugins(id) ON DELETE CASCADE,

    -- Modality for this competitor
    modality modality_enum NOT NULL,

    -- Hash of configuration JSON
    config_hash TEXT NOT NULL,

    -- Full configuration used for execution
    config_json JSONB NOT NULL,

    -- Current ELO rating
    elo INTEGER NOT NULL DEFAULT 1200,

    -- Aggregate stats
    battles_fought INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    ties INTEGER NOT NULL DEFAULT 0,

    -- Registration timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicate configs per plugin
    UNIQUE (plugin_id, config_hash)
);

-- Performance-critical indexes
CREATE INDEX idx_competitors_plugin ON competitors(plugin_id);
CREATE INDEX idx_competitors_modality ON competitors(modality);

-- ============================================================================
-- BATTLES TABLE
-- ============================================================================

-- Represents one A/B comparison
CREATE TABLE battles (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Modality of the battle
    modality modality_enum NOT NULL,

    -- Competing configurations
    competitor_a_id UUID NOT NULL REFERENCES competitors(id),
    competitor_b_id UUID NOT NULL REFERENCES competitors(id),

    -- Reference to input (text or audio identifier)
    input_ref TEXT NOT NULL,

    -- References to generated outputs (e.g., MinIO paths)
    output_a_ref TEXT,
    output_b_ref TEXT,

    -- Structured results for disagreement validation
    result_a_data JSONB,
    result_b_data JSONB,

    -- Execution state
    status battle_status_enum NOT NULL DEFAULT 'PENDING',

    -- Failure reason if execution fails
    failure_reason TEXT,

    -- Retry counter
    attempt INTEGER NOT NULL DEFAULT 0,

    -- Optional metadata (dataset, prompt, worker version)
    metadata JSONB,

    -- Audit timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for worker and API queries
CREATE INDEX idx_battles_status ON battles(status);
CREATE INDEX idx_battles_modality ON battles(modality);

-- ============================================================================
-- VOTES TABLE
-- ============================================================================

-- Stores the final judgment for a battle
CREATE TABLE votes (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Battle being voted on
    battle_id UUID NOT NULL REFERENCES battles(id) ON DELETE CASCADE,

    -- User who submitted the vote
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Vote result
    result vote_result_enum NOT NULL,

    -- Timestamp of submission
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Enforce exactly one vote per battle per user
    UNIQUE (battle_id, user_id)
);

CREATE INDEX idx_votes_created_at ON votes(created_at);
CREATE INDEX idx_votes_user ON votes(user_id);
CREATE INDEX idx_votes_battle ON votes(battle_id);

-- ============================================================================
-- ELO HISTORY TABLE
-- ============================================================================

-- Tracks ELO changes for audit and recomputation
CREATE TABLE elo_history (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Competitor whose ELO changed
    competitor_id UUID NOT NULL REFERENCES competitors(id) ON DELETE CASCADE,

    -- Battle that caused the change
    battle_id UUID NOT NULL REFERENCES battles(id) ON DELETE CASCADE,

    -- ELO before the battle
    old_elo INTEGER NOT NULL,

    -- ELO after the battle
    new_elo INTEGER NOT NULL,

    -- Change amount (can be derived but stored for convenience)
    elo_change INTEGER NOT NULL,

    -- Timestamp of change
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_elo_history_competitor ON elo_history(competitor_id);
CREATE INDEX idx_elo_history_battle ON elo_history(battle_id);
CREATE INDEX idx_elo_history_created_at ON elo_history(created_at);

-- ============================================================================
-- PLUGIN DAILY METRICS (TIME-SERIES ANALYTICS)
-- ============================================================================

-- Aggregated metrics for graphs and trends
CREATE TABLE plugin_metrics_daily (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Plugin this metric belongs to
    plugin_id UUID NOT NULL REFERENCES plugins(id) ON DELETE CASCADE,

    -- Modality of the metrics
    modality modality_enum NOT NULL,

    -- Date of aggregation (UTC)
    metric_date DATE NOT NULL,

    -- Aggregated ELO statistics
    avg_elo INTEGER NOT NULL,
    max_elo INTEGER NOT NULL,
    min_elo INTEGER NOT NULL,

    -- Aggregated battle stats
    battles_fought INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    ties INTEGER NOT NULL,

    -- Ensure one row per plugin/modality/day
    UNIQUE (plugin_id, modality, metric_date)
);

CREATE INDEX idx_plugin_metrics_date ON plugin_metrics_daily(metric_date);

-- ============================================================================
-- UPDATED_AT TRIGGERS
-- ============================================================================

-- Automatically updates updated_at on row modification
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to plugins table
DROP TRIGGER IF EXISTS update_plugins_updated_at ON plugins;
CREATE TRIGGER update_plugins_updated_at
    BEFORE UPDATE ON plugins
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to battles table
DROP TRIGGER IF EXISTS update_battles_updated_at ON battles;
CREATE TRIGGER update_battles_updated_at
    BEFORE UPDATE ON battles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();