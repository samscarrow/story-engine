Messaging contracts and topics (v1)

- Version: v1
- Purpose: define message envelopes, topics, contracts, and DLQs for workers.

Topics

- Plot
  - `plot.request` → request to generate a plot outline
  - `plot.done` → plot outline generated successfully

- Scene
  - `scene.request` → request to generate a scene for a beat
  - `scene.done` → scene created

- Dialogue
  - `dialogue.request` → request to generate character dialogue
  - `dialogue.done` → dialogue created

- Evaluation
  - `evaluation.request` → request to evaluate content against criteria
  - `evaluation.done` → evaluation produced

Dead-letter queues (DLQ)

- Per-topic DLQ uses `dlq.<topic>` naming.
  - Examples:
    - `dlq.plot.request`
    - `dlq.scene.request`
    - `dlq.dialogue.request`
    - `dlq.evaluation.request`

Retry policy

- Max retries: 3 (adapter-enforced when available). In-memory bus does not retry automatically.
- On validation errors: do not retry; route to corresponding DLQ.

Contracts

- `plot.request` (schema):
  - `job_id` (string, required)
  - `prompt` (string, required)
  - `constraints` (object, required)

- `plot.done` (schema):
  - `job_id` (string, required)
  - `outline_id` (string, required)
  - `outline_ref` (string, required)

- `scene.request` (schema):
  - `job_id` (string, required)
  - `outline_id` (string, optional)
  - `beat_name` (string, optional)
  - `prompt` (string, required)
  - `characters` (array, required)
  - `constraints` (object, required)

- `scene.done` (schema):
  - `job_id` (string, required)
  - `scene_id` (string, required)
  - `scene_description` (string, required)

- `dialogue.request` (schema):
  - `job_id` (string, required)
  - `scene_id` (string, optional)
  - `character_id` (string, required)
  - `opening_line` (string, optional)
  - `context` (object, required)

- `dialogue.done` (schema):
  - `job_id` (string, required)
  - `scene_id` (string, optional)
  - `text` (string, required)

- `evaluation.request` (schema):
  - `job_id` (string, required)
  - `content` (string, required)
  - `criteria` (array, optional)
  - `options` (object, required)

- `evaluation.done` (schema):
  - `job_id` (string, required)
  - `evaluation_text` (string, required)

Enforcement

- Code registry: `story_engine.core.core.contracts.topics.VALIDATORS` maps topics to `validate(payload)`.
- In-memory bus:
  - Validates on `publish()` when a validator exists; on failure routes to `dlq.<topic>` if subscribed, else raises.
  - Validates again before invoking a subscriber; on failure routes to `dlq.<topic>` if subscribed, else raises.
- Services should still validate on boundaries (defense-in-depth) when constructing messages.

Adapter Parity (RabbitMQ)

- Type/topic normalization: the adapter normalizes `Message.type` to the publish `topic` and annotates mismatch in `headers`.
- Validation on publish: if validation fails, publishes an enriched diagnostic message to `dlq.<topic>` and does not send the original to the main queue.
- Validation on consume: validates before invoking the handler; on failure, publishes enriched DLQ diagnostics and acknowledges the original.
- Processing retries: for non-validation errors, retries the handler with exponential backoff up to `max_retries`; after exhaustion, publishes to DLQ and acknowledges the original.
- DLQ payload includes `original` (payload), `error` (string), and optionally `attempts`; headers include `validation_error` or `retries_exhausted` flags.

Correlation

- `Message.correlation_id` should carry `job_id` for end-to-end tracing.
- `Message.causation_id` should reference the triggering message `id`.
