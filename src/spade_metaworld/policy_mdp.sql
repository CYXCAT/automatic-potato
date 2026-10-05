PRAGMA foreign_keys = ON;

CREATE TABLE world (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    discount REAL NOT NULL CHECK (discount >= 0.0 AND discount < 1.0)
);

CREATE TABLE design_state (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    complexity_min REAL,
    complexity_max REAL,
    environment_spec_json TEXT,
    is_terminal INTEGER NOT NULL CHECK (is_terminal IN (0, 1)),
    terminal_kind TEXT CHECK (
        terminal_kind IS NULL OR terminal_kind IN ('accepted', 'rejected')
    ),
    UNIQUE (world_id, name),
    CHECK (
        (complexity_min IS NULL AND complexity_max IS NULL)
        OR
        (complexity_min >= 0.0 AND complexity_max > complexity_min)
    )
);

CREATE TABLE design_action (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE transition (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    action_id INTEGER NOT NULL REFERENCES design_action(id),
    next_state_id INTEGER NOT NULL REFERENCES design_state(id),
    probability REAL NOT NULL CHECK (probability > 0.0 AND probability <= 1.0),
    reward REAL NOT NULL,
    UNIQUE (world_id, state_id, action_id, next_state_id)
);

-- Raw evidence used to estimate P(s'|s,a). Rows are append-only observations from
-- prior Week 1 history or newly executed designer/validator trials.
CREATE TABLE transition_observation (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    action_id INTEGER NOT NULL REFERENCES design_action(id),
    next_state_id INTEGER NOT NULL REFERENCES design_state(id),
    sample_seed INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('week1_history', 'active_sampling')),
    passed_validation INTEGER NOT NULL CHECK (passed_validation IN (0, 1)),
    reward REAL NOT NULL,
    environment_spec_json TEXT,
    validation_report_json TEXT
);

CREATE TABLE transition_model_fit (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    name TEXT NOT NULL,
    prior_strength REAL NOT NULL CHECK (prior_strength >= 0.0),
    samples_per_action INTEGER NOT NULL CHECK (samples_per_action >= 0),
    random_seed INTEGER NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, name)
);

CREATE TABLE fitted_transition (
    fit_id INTEGER NOT NULL REFERENCES transition_model_fit(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    action_id INTEGER NOT NULL REFERENCES design_action(id),
    next_state_id INTEGER NOT NULL REFERENCES design_state(id),
    observed_count INTEGER NOT NULL CHECK (observed_count >= 0),
    probability REAL NOT NULL CHECK (probability > 0.0 AND probability <= 1.0),
    reward_mean REAL NOT NULL,
    PRIMARY KEY (fit_id, state_id, action_id, next_state_id)
);

CREATE TABLE policy (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    name TEXT NOT NULL,
    iteration INTEGER NOT NULL CHECK (iteration >= 0),
    UNIQUE (world_id, name)
);

CREATE TABLE policy_probability (
    policy_id INTEGER NOT NULL REFERENCES policy(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    action_id INTEGER NOT NULL REFERENCES design_action(id),
    probability REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    PRIMARY KEY (policy_id, state_id, action_id)
);

CREATE TABLE algorithm_run (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES world(id),
    name TEXT NOT NULL,
    algorithm TEXT NOT NULL CHECK (
        algorithm IN ('iterative_evaluation', 'exact_evaluation', 'policy_iteration')
    ),
    tolerance REAL NOT NULL CHECK (tolerance > 0.0),
    max_iterations INTEGER NOT NULL CHECK (max_iterations > 0),
    status TEXT NOT NULL CHECK (status IN ('running', 'converged', 'failed')),
    iterations INTEGER NOT NULL DEFAULT 0 CHECK (iterations >= 0),
    UNIQUE (world_id, name)
);

CREATE TABLE value_snapshot (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES algorithm_run(id),
    policy_id INTEGER NOT NULL REFERENCES policy(id),
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    bellman_residual REAL NOT NULL CHECK (bellman_residual >= 0.0),
    changed_policy_states INTEGER,
    UNIQUE (run_id, sequence)
);

CREATE TABLE state_value (
    snapshot_id INTEGER NOT NULL REFERENCES value_snapshot(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    value REAL NOT NULL,
    PRIMARY KEY (snapshot_id, state_id)
);

CREATE TABLE action_value (
    snapshot_id INTEGER NOT NULL REFERENCES value_snapshot(id),
    state_id INTEGER NOT NULL REFERENCES design_state(id),
    action_id INTEGER NOT NULL REFERENCES design_action(id),
    value REAL NOT NULL,
    PRIMARY KEY (snapshot_id, state_id, action_id)
);

CREATE INDEX transition_lookup
ON transition(world_id, state_id, action_id);

CREATE INDEX transition_observation_lookup
ON transition_observation(world_id, state_id, action_id, next_state_id);

CREATE INDEX snapshot_history
ON value_snapshot(run_id, sequence);
