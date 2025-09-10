-- Story Engine centralized schema (Oracle 19c+ compatible)
-- Run as a privileged user or a user with CREATE TABLE/INDEX privileges

-- Providers
CREATE TABLE providers (
  provider_id  NUMBER PRIMARY KEY,
  name         VARCHAR2(64) NOT NULL,
  type         VARCHAR2(32) NOT NULL, -- lmstudio, koboldcpp, openai, etc.
  endpoint     VARCHAR2(512) NOT NULL,
  status       VARCHAR2(16) DEFAULT 'active',
  meta         CLOB CHECK (meta IS JSON),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX ux_providers_name ON providers (name);

-- Providers ID sequence + trigger
CREATE SEQUENCE providers_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER providers_bi
BEFORE INSERT ON providers FOR EACH ROW
WHEN (NEW.provider_id IS NULL)
BEGIN
  :NEW.provider_id := providers_seq.NEXTVAL;
END;
/

-- Models
CREATE TABLE models (
  model_id     NUMBER PRIMARY KEY,
  provider_id  NUMBER NOT NULL REFERENCES providers(provider_id) ON DELETE CASCADE,
  model_key    VARCHAR2(256) NOT NULL,  -- id/slug as reported by provider
  display_name VARCHAR2(256),
  family       VARCHAR2(128),           -- llama, qwen, phi, etc.
  size_b       NUMBER(10,2),            -- billions params (nullable)
  capabilities CLOB CHECK (capabilities IS JSON),
  modalities   CLOB CHECK (modalities IS JSON),
  defaults     CLOB CHECK (defaults IS JSON), -- default gen params
  active       CHAR(1) DEFAULT 'Y' CHECK (active IN ('Y','N')),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX ux_models_provider_key ON models(provider_id, model_key);
CREATE INDEX ix_models_provider ON models(provider_id);

-- Models ID sequence + trigger
CREATE SEQUENCE models_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER models_bi
BEFORE INSERT ON models FOR EACH ROW
WHEN (NEW.model_id IS NULL)
BEGIN
  :NEW.model_id := models_seq.NEXTVAL;
END;
/

-- Personas
CREATE TABLE personas (
  persona_id   NUMBER PRIMARY KEY,
  code         VARCHAR2(64) NOT NULL, -- e.g., 'story_designer'
  name         VARCHAR2(128) NOT NULL,
  description  VARCHAR2(512),
  defaults     CLOB CHECK (defaults IS JSON), -- temp, max_tokens, etc.
  config       CLOB CHECK (config IS JSON),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX ux_personas_code ON personas(code);

-- Personas ID sequence + trigger
CREATE SEQUENCE personas_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER personas_bi
BEFORE INSERT ON personas FOR EACH ROW
WHEN (NEW.persona_id IS NULL)
BEGIN
  :NEW.persona_id := personas_seq.NEXTVAL;
END;
/

-- Templates
CREATE TABLE templates (
  template_id  NUMBER PRIMARY KEY,
  persona_id   NUMBER REFERENCES personas(persona_id) ON DELETE SET NULL,
  path         VARCHAR2(512) NOT NULL,
  version      VARCHAR2(64) NOT NULL,
  checksum     VARCHAR2(64),
  content      CLOB,
  active       CHAR(1) DEFAULT 'Y' CHECK (active IN ('Y','N')),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX ux_templates_path_vers ON templates(path, version);

-- Templates ID sequence + trigger
CREATE SEQUENCE templates_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER templates_bi
BEFORE INSERT ON templates FOR EACH ROW
WHEN (NEW.template_id IS NULL)
BEGIN
  :NEW.template_id := templates_seq.NEXTVAL;
END;
/

-- Prompt sets
CREATE TABLE prompt_sets (
  set_id      NUMBER PRIMARY KEY,
  name        VARCHAR2(128) NOT NULL,
  tier        VARCHAR2(16)  NOT NULL CHECK (tier IN ('golden','silver','bronze')),
  description VARCHAR2(512),
  created_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX ux_prompt_sets_name ON prompt_sets(name);

-- Prompt sets ID sequence + trigger
CREATE SEQUENCE prompt_sets_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER prompt_sets_bi
BEFORE INSERT ON prompt_sets FOR EACH ROW
WHEN (NEW.set_id IS NULL)
BEGIN
  :NEW.set_id := prompt_sets_seq.NEXTVAL;
END;
/

-- Prompts
CREATE TABLE prompts (
  prompt_id   NUMBER PRIMARY KEY,
  set_id      NUMBER REFERENCES prompt_sets(set_id) ON DELETE CASCADE,
  external_id VARCHAR2(128), -- stable id in repo, if any
  payload     CLOB CHECK (payload IS JSON) NOT NULL, -- prompt spec
  tags        CLOB CHECK (tags IS JSON),
  created_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_prompts_set ON prompts(set_id);

-- Prompts ID sequence + trigger
CREATE SEQUENCE prompts_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER prompts_bi
BEFORE INSERT ON prompts FOR EACH ROW
WHEN (NEW.prompt_id IS NULL)
BEGIN
  :NEW.prompt_id := prompts_seq.NEXTVAL;
END;
/

-- Generations (runs)
CREATE TABLE generations (
  run_id          NUMBER PRIMARY KEY,
  prompt_id       NUMBER REFERENCES prompts(prompt_id) ON DELETE SET NULL,
  persona_id      NUMBER REFERENCES personas(persona_id) ON DELETE SET NULL,
  provider_id     NUMBER NOT NULL REFERENCES providers(provider_id),
  model_id        NUMBER REFERENCES models(model_id),
  model_key       VARCHAR2(256),
  request_json    CLOB CHECK (request_json IS JSON),
  response_text   CLOB,
  response_json   CLOB CHECK (response_json IS JSON),
  usage_prompt    NUMBER,
  usage_completion NUMBER,
  cost_usd        NUMBER(12,6),
  latency_ms      NUMBER,
  status          VARCHAR2(16) NOT NULL CHECK (status IN ('ok','error','timeout','rejected')),
  error_json      CLOB CHECK (error_json IS JSON),
  started_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  finished_at     TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_gen_prompt   ON generations(prompt_id);
CREATE INDEX ix_gen_persona  ON generations(persona_id);
CREATE INDEX ix_gen_provider ON generations(provider_id);
CREATE INDEX ix_gen_model    ON generations(model_id);
CREATE INDEX ix_gen_started  ON generations(started_at);

-- Generations ID sequence + trigger
CREATE SEQUENCE generations_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER generations_bi
BEFORE INSERT ON generations FOR EACH ROW
WHEN (NEW.run_id IS NULL)
BEGIN
  :NEW.run_id := generations_seq.NEXTVAL;
END;
/

-- Evaluations
CREATE TABLE evaluations (
  eval_id      NUMBER PRIMARY KEY,
  run_id       NUMBER NOT NULL REFERENCES generations(run_id) ON DELETE CASCADE,
  criteria     CLOB CHECK (criteria IS JSON),
  score_overall NUMBER(5,2),
  breakdown    CLOB CHECK (breakdown IS JSON),
  notes        CLOB,
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_eval_run ON evaluations(run_id);

-- Evaluations ID sequence + trigger
CREATE SEQUENCE evaluations_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER evaluations_bi
BEFORE INSERT ON evaluations FOR EACH ROW
WHEN (NEW.eval_id IS NULL)
BEGIN
  :NEW.eval_id := evaluations_seq.NEXTVAL;
END;
/

-- Scenes
CREATE TABLE scenes (
  scene_id     NUMBER PRIMARY KEY,
  run_id       NUMBER REFERENCES generations(run_id) ON DELETE SET NULL,
  title        VARCHAR2(256),
  beat_name    VARCHAR2(128),
  content      CLOB,
  meta         CLOB CHECK (meta IS JSON),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_scenes_run ON scenes(run_id);

-- Scenes ID sequence + trigger
CREATE SEQUENCE scenes_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER scenes_bi
BEFORE INSERT ON scenes FOR EACH ROW
WHEN (NEW.scene_id IS NULL)
BEGIN
  :NEW.scene_id := scenes_seq.NEXTVAL;
END;
/

-- Dialogue
CREATE TABLE dialogue (
  dialogue_id  NUMBER PRIMARY KEY,
  scene_id     NUMBER NOT NULL REFERENCES scenes(scene_id) ON DELETE CASCADE,
  character_id VARCHAR2(128),
  line_no      NUMBER,
  text         CLOB,
  meta         CLOB CHECK (meta IS JSON),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_dialogue_scene ON dialogue(scene_id);
CREATE INDEX ix_dialogue_char  ON dialogue(character_id);

-- Dialogue ID sequence + trigger
CREATE SEQUENCE dialogue_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER dialogue_bi
BEFORE INSERT ON dialogue FOR EACH ROW
WHEN (NEW.dialogue_id IS NULL)
BEGIN
  :NEW.dialogue_id := dialogue_seq.NEXTVAL;
END;
/

-- World entities
CREATE TABLE world_entities (
  entity_id    NUMBER PRIMARY KEY,
  type         VARCHAR2(64) NOT NULL,
  name         VARCHAR2(256) NOT NULL,
  attributes   CLOB CHECK (attributes IS JSON),
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  updated_at   TIMESTAMP WITH TIME ZONE
);

CREATE UNIQUE INDEX ux_entities_name_type ON world_entities(type, name);

-- World entities ID sequence + trigger
CREATE SEQUENCE world_entities_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER world_entities_bi
BEFORE INSERT ON world_entities FOR EACH ROW
WHEN (NEW.entity_id IS NULL)
BEGIN
  :NEW.entity_id := world_entities_seq.NEXTVAL;
END;
/

-- World facts
CREATE TABLE world_facts (
  fact_id       NUMBER PRIMARY KEY,
  subject_id    NUMBER NOT NULL REFERENCES world_entities(entity_id) ON DELETE CASCADE,
  predicate     VARCHAR2(128) NOT NULL,
  object_id     NUMBER REFERENCES world_entities(entity_id),
  value_text    CLOB,
  confidence    NUMBER(3,2),
  source_run_id NUMBER REFERENCES generations(run_id) ON DELETE SET NULL,
  effective_from TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  effective_to   TIMESTAMP WITH TIME ZONE,
  created_at     TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_facts_subject   ON world_facts(subject_id);
CREATE INDEX ix_facts_predicate ON world_facts(predicate);

-- World facts ID sequence + trigger
CREATE SEQUENCE world_facts_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER world_facts_bi
BEFORE INSERT ON world_facts FOR EACH ROW
WHEN (NEW.fact_id IS NULL)
BEGIN
  :NEW.fact_id := world_facts_seq.NEXTVAL;
END;
/

-- World relations
CREATE TABLE world_relations (
  relation_id    NUMBER PRIMARY KEY,
  from_entity_id NUMBER NOT NULL REFERENCES world_entities(entity_id) ON DELETE CASCADE,
  to_entity_id   NUMBER NOT NULL REFERENCES world_entities(entity_id) ON DELETE CASCADE,
  type           VARCHAR2(128) NOT NULL,
  attributes     CLOB CHECK (attributes IS JSON),
  created_at     TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  updated_at     TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_rel_from ON world_relations(from_entity_id);
CREATE INDEX ix_rel_to   ON world_relations(to_entity_id);
CREATE INDEX ix_rel_type ON world_relations(type);

-- World relations ID sequence + trigger
CREATE SEQUENCE world_relations_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER world_relations_bi
BEFORE INSERT ON world_relations FOR EACH ROW
WHEN (NEW.relation_id IS NULL)
BEGIN
  :NEW.relation_id := world_relations_seq.NEXTVAL;
END;
/

-- Cache entries (optional backing for ResponseCache)
CREATE TABLE cache_entries (
  cache_key   VARCHAR2(256) PRIMARY KEY,
  provider    VARCHAR2(64),
  model_key   VARCHAR2(256),
  params_hash VARCHAR2(64),
  value       CLOB,
  created_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  expires_at  TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_cache_expires ON cache_entries(expires_at);

-- Metrics events
CREATE TABLE metrics_events (
  event_id    NUMBER PRIMARY KEY,
  kind        VARCHAR2(64) NOT NULL,   -- health, retry, fallback, rate_limit, etc.
  run_id      NUMBER REFERENCES generations(run_id) ON DELETE SET NULL,
  provider_id NUMBER REFERENCES providers(provider_id) ON DELETE SET NULL,
  model_id    NUMBER REFERENCES models(model_id) ON DELETE SET NULL,
  data        CLOB CHECK (data IS JSON),
  at          TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX ix_metrics_kind_at ON metrics_events(kind, at);

-- Metrics events ID sequence + trigger
CREATE SEQUENCE metrics_events_seq START WITH 1 NOCACHE;
CREATE OR REPLACE TRIGGER metrics_events_bi
BEFORE INSERT ON metrics_events FOR EACH ROW
WHEN (NEW.event_id IS NULL)
BEGIN
  :NEW.event_id := metrics_events_seq.NEXTVAL;
END;
/

-- QoL triggers for updated_at
CREATE OR REPLACE TRIGGER trg_entities_upd
BEFORE UPDATE ON world_entities FOR EACH ROW
BEGIN
  :NEW.updated_at := SYSTIMESTAMP;
END;
/

CREATE OR REPLACE TRIGGER trg_relations_upd
BEFORE UPDATE ON world_relations FOR EACH ROW
BEGIN
  :NEW.updated_at := SYSTIMESTAMP;
END;
/
