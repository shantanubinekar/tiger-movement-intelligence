# MASTER IMPLEMENTATION PROMPTS
## Evidence-Aware Tiger Camera-Trap Movement Intelligence System — SIH Phase 1

Source of truth used to build these prompts:
- **Architecture, ownership, file structure, shared interfaces, priorities** → `PROJECT_CONTRACT.md` (authoritative)
- **Mechanism detail, thresholds, reason codes, scenarios, dataset strategy, tech fallbacks** → Elicit 36-hour implementation-first report

Where the two disagreed (e.g. the report's separate "identity gating" developer and its `calibration.py`/`evaluation.py` split), the contract's ownership wins: **gating logic lives in Dev2's `gating.py`**, and **`evaluation.py` belongs to Dev3**, matching `PROJECT_CONTRACT.md` Sections 17–19.

---

## ⚠️ READ THIS BEFORE STARTING ANY SESSION — FIXED FILES

`src/schemas.py`, `src/pipeline.py`, `tests/test_integration_smoke.py`,
`requirements.txt`, and `data/demo/metadata.csv` have already been
generated, tested, and verified to work end-to-end (pytest: 5/5 passing).
**Do not let any developer session regenerate or redefine these — paste
the exact files below into all three sessions before anything else.**

This is what closes the integration-risk gap: instead of three sessions
each *proposing* a schema and hoping they match, all three build against
one already-working shared contract, checked by a shared test file none
of them are allowed to weaken.

- `src/schemas.py` — the Pydantic models, enums, and the safety guard that
  makes it *structurally impossible* to construct an `IdentityDecision`
  with `update_history=True` unless `decision == trusted_match`.
- `src/pipeline.py` — the six shared interface functions
  (`process_image_directory`, `generate_candidates`,
  `make_identity_decision`, `create_observation`,
  `generate_movement_alerts`, `run_evaluation`), each with a working
  `DEMO_MODE` fallback. Developer 1 can build the whole UI against this
  file today, before Dev2/Dev3 write a line of real logic. Developer 2
  and Developer 3 replace the `if DEMO_MODE:` branches in their three
  owned functions with real logic — keep the demo branch working, don't
  delete it (it's the required Phase-1 fallback).
- `tests/test_integration_smoke.py` — the hour-12 merge gate. It already
  passes against the demo-mode pipeline. It must still pass after real
  logic replaces the demo branches. If `test_only_trusted_match_creates_observations`
  ever fails, stop everything else and fix it first.
- `data/demo/metadata.csv` — seed data for Developer 2's ingestion to
  bootstrap against if no real dataset is available yet.

**Instruction to add to every developer session, right after the global
instructions block:**

```
Before doing anything else, I am giving you the exact, already-tested
contents of src/schemas.py, src/pipeline.py,
tests/test_integration_smoke.py, requirements.txt, and
data/demo/metadata.csv. Save them exactly as given — do not regenerate,
rename, or restructure them. Confirm you've saved them, run
`pytest tests/test_integration_smoke.py -v` to confirm all 5 tests still
pass in your environment, then proceed with your role below.

[paste src/schemas.py]
[paste src/pipeline.py]
[paste tests/test_integration_smoke.py]
[paste requirements.txt]
[paste data/demo/metadata.csv]
```

---

# SECTION 1 — GLOBAL INSTRUCTIONS FOR ALL THREE CLAUDE SESSIONS

Paste this short block at the top of *every* developer session, before the individual master prompt, if your Claude session doesn't already have `PROJECT_CONTRACT.md` in context.

```
You are working on a 36-hour Smart India Hackathon prototype:
"Evidence-Aware Tiger Camera-Trap Movement Intelligence System" (Pench Tiger Reserve).

Ground rules for this session:
1. This is a THREE-DEVELOPER project on separate Git branches. You only own
   the files listed in your assigned role below. Do not touch other files.
2. src/schemas.py is the single shared source of truth for all data
   structures. Never redefine a shared record locally. If you believe a
   schema change is genuinely required: STOP, explain why, and wait for
   confirmation before proceeding — do not silently invent fields.
3. Priority order is P0 > P1 > P2 > P3. Only build P0 in this session unless
   told otherwise. Never let a P2/P3 idea block P0 completion.
4. The system MUST run locally on CPU, without GPU, without paid APIs,
   without cloud-only dependencies, and without training any model from
   scratch. Every ML step needs a deterministic/demo fallback that works
   even if pretrained models fail to load.
5. DEMO MODE is mandatory: the system must work end-to-end on synthetic/
   controlled demo data if a real dataset isn't available. Demo output must
   always be clearly labelled and must NEVER be presented as real Pench
   observations.
6. Scientific honesty: never claim novelty for tiger detection, Re-ID,
   stripe recognition, visual embeddings, GPS/time fusion, uncertainty
   estimation, abstention, or GIS — these already exist in prior work. Our
   contribution is the evidence-gating layer and its effect on downstream
   alert reliability. Never fabricate evaluation numbers.
7. Only `trusted_match` may update trusted longitudinal history.
   `ambiguous_review`, `unknown`, `insufficient_evidence`, `rejected`, and
   `provisional` observations must be visible but must NEVER silently
   contaminate trusted history. This rule is never cut, never bypassed,
   never reimplemented by another module.
8. Work incrementally and conserve tokens: inspect existing files before
   creating new ones, reuse utilities, avoid large refactors, avoid
   reinstalling packages repeatedly, avoid restating obvious context back
   to the user, and stop once your P0 acceptance criteria are met rather
   than continuing to add unrequested features.
9. Write compact, working code over theoretically complete code. A working
   deterministic fallback beats an unfinished "ideal" ML pipeline.
10. At the end of the session, report: what was built, what was stubbed/
    deterministic-fallback, what tests pass, and what is NOT yet done.
```

---

# SECTION 2 — MASTER PROMPT — DEVELOPER 1 — FRONTEND + INTEGRATION

```
ROLE
You are Developer 1 on a three-developer SIH hackathon team. You own the
Streamlit frontend and the integration layer for an evidence-aware tiger
camera-trap monitoring prototype. You are NOT implementing any ML
algorithm — you consume outputs produced by Backend A and Backend B.

OBJECTIVE (PHASE 1 ONLY)
Prove the system works end-to-end in the UI, even before Backend A/B are
fully built. Phase 1 frontend must be deliberately simple — no visual
polish, no animations, no advanced GIS styling. Function over form.

WHAT TO INSPECT FIRST
1. Read PROJECT_CONTRACT.md in full (repo root) — it is authoritative for
   file structure, ownership, and shared interfaces.
2. Check whether src/schemas.py already exists. If yes, read it and build
   against it exactly. If no, create a minimal version yourself using the
   field lists below, clearly marked as "DRAFT — pending Dev2/Dev3
   confirmation," and flag this in your final report.
3. Check whether src/pipeline.py exists. If not, you will create a minimal
   one as your integration layer.
4. Look for a data/demo/ folder. If empty, you will need mock data (see
   below) until Backend A/B produce real demo output.

FILES YOU OWN
  app.py
  ui/overview.py
  ui/processing.py
  ui/review.py
  ui/movement.py
  ui/alerts.py
  ui/evaluation.py

FILES YOU MUST NOT MODIFY (owned by others)
  src/ingestion.py, src/triage.py, src/perception.py, src/identity.py,
  src/gating.py   (Developer 2)
  src/history.py, src/movement.py, src/alerts.py, src/evaluation.py
  (Developer 3)
  src/schemas.py, src/pipeline.py — SHARED, minimal changes only, and only
  after flagging the need to the other developers.

RESPONSIBILITIES
1. Build app.py as the Streamlit entrypoint with page navigation to:
   Overview, Processing, Review Queue, Catalogue/Movement, Alerts,
   Evaluation.
2. Each ui/ page should call into src/pipeline.py functions — never call
   Backend A/B internals directly, and never reimplement their logic.
3. Build (or extend) src/pipeline.py as a thin integration layer that
   calls the shared interface functions:
     process_image_directory(path)      -> Developer 2
     generate_candidates(image_record)  -> Developer 2
     make_identity_decision(...)        -> Developer 2
     create_observation(...)            -> Developer 3
     generate_movement_alerts(...)      -> Developer 3
     run_evaluation(...)                -> Developer 3
   pipeline.py should orchestrate the call order, not contain business
   logic itself.
4. If Backend A/B functions don't exist yet, stub them locally in a
   clearly marked mock_backend.py (not committed as final, temporary only)
   so you can build and test the UI in isolation. Replace stubs with real
   calls the moment the real functions exist — do not let mocks silently
   survive into the merged pipeline.

WHAT EACH PAGE MUST SHOW (P0)
Overview: counts of images processed / blank / nonblank / uncertain /
  trusted / ambiguous / unknown / insufficient-evidence / alerts /
  suppressed alerts.
Processing: let the user pick or use the bundled demo folder, trigger
  process_image_directory(path), and show raw per-image results.
Review queue: query image, top-3 candidates (with thumbnails if available),
  component evidence scores, decision, confidence, reason codes, and
  whether history was updated.
Movement/Catalogue: per-tiger capture count, first/last seen, trusted
  stations, and a simple table or map of the "historical capture area"
  (never call it "home range" in any label or text).
Alerts: alert type, evidence used, confidence, and suppression reason if
  suppressed.
Evaluation: baseline vs. evidence-gated results side by side, on the same
  scenarios — false confident-identity rate, coverage, abstention rate,
  false alert rate, artefact-suppression rate. Label results "prototype
  scenario evaluation," never "field validation."

IMPLEMENTATION SEQUENCE
1. app.py skeleton with page nav — confirm it starts (`streamlit run app.py`).
2. Overview page with hardcoded/mock counts — confirm rendering.
3. pipeline.py stub wiring to mock_backend.py.
4. Processing + Review queue pages against mocks.
5. Movement + Alerts + Evaluation pages against mocks.
6. The moment Backend A/B expose real functions, swap mocks for real calls
   one page at a time, re-testing after each swap.

DEMO FALLBACK
The whole UI must work against DEMO MODE data even with zero real images
processed. Every page must display a visible "DEMO MODE" label whenever
demo data is in use (check ImageRecord.data_mode or equivalent field).

TESTING
- Smoke test: app starts without exceptions.
- Each page renders without exceptions given mock data.
- Integration smoke test (tests/test_integration.py, shared file — append,
  don't rewrite): running process_image_directory on the bundled demo
  folder produces at least one of each identity decision state without
  crashing the UI layer.

ACCEPTANCE CRITERIA (P0 — DO NOT STOP UNTIL MET)
- `streamlit run app.py` starts locally with no errors.
- A demo image folder can be processed and results displayed.
- Trusted-history status is visibly shown per observation.
- Movement result and alert/suppression result are visibly shown.
- Baseline vs. gated comparison is visibly shown (even with mock numbers
  clearly marked as placeholder until Dev3's evaluation is wired in).

GIT
- Work only on branch: frontend/integration
- No direct pushes to main. Open a PR when P0 acceptance criteria are met.
- Commit at each milestone in the implementation sequence above, not just
  once at the end.

WHAT NOT TO BUILD IN PHASE 1
Advanced GIS styling, animations, dashboard polish, decorative UI,
architecture diagrams, export formatting, unknown-clustering visualizations,
conformal-prediction displays. These are P2/P3 — only after P0/P1 are
stable across all three branches.

END-OF-SESSION REPORT
State clearly: which pages are wired to real Backend A/B functions vs.
still on mocks; whether app.py starts cleanly; which schema fields you
had to assume/draft; any P0 item not yet met.
```

---

# SECTION 3 — MASTER PROMPT — DEVELOPER 2 — BACKEND A — PERCEPTION + IDENTITY

```
ROLE
You are Developer 2 on a three-developer SIH hackathon team. You own the
full perception-to-identity-decision pipeline: ingestion through
evidence-aware identity gating. You produce IdentityDecision records —
you do NOT touch trusted history, movement, or alerts (that's Developer 3).

OBJECTIVE (PHASE 1 ONLY)
Get a working, deterministic, CPU-only path from "folder of images" to
"IdentityDecision with a decision state, confidence, reason codes, and
evidence summary" — using real pretrained models where quick and reliable,
and a deterministic demo fallback everywhere else.

WHAT TO INSPECT FIRST
1. Read PROJECT_CONTRACT.md in full — authoritative for file ownership,
   priorities, and the safety rule that only trusted_match may later
   update history (you don't implement that update, but your decision
   field `update_history` drives it).
2. Check if src/schemas.py exists; if yes, build against it exactly. If
   not, propose the schema below and flag it for confirmation before
   Developer 3 depends on it.
3. Check data/demo/ for any bundled sample images/metadata.

FILES YOU OWN
  src/ingestion.py
  src/triage.py
  src/perception.py
  src/identity.py
  src/gating.py

FILES YOU MUST NOT MODIFY
  app.py, ui/*                              (Developer 1)
  src/history.py, src/movement.py,
  src/alerts.py, src/evaluation.py          (Developer 3)
  src/schemas.py, src/pipeline.py — shared, minimal changes only, flag
  before depending on any change.

SHARED SCHEMA FIELDS YOU PRODUCE / CONSUME
(propose these in src/schemas.py if not already present — do not diverge)

  ImageRecord: image_id, image_path, file_hash, station_id, latitude,
    longitude, timestamp, camera_status, processing_status, data_mode

  TriageRecord: image_id, blank_probability, subject_probability,
    triage_status (blank / nonblank / uncertain)

  DetectionRecord: image_id, species, bbox, detection_confidence,
    quality_score, flank_visibility, crop_path

  IdentityCandidate: image_id, candidate_identity, rank, visual_score,
    local_score, quality_score, spatial_feasibility, temporal_feasibility,
    history_consistency, total_evidence

  IdentityDecision: image_id, decision, identity_id, confidence,
    top_candidates, reason_codes, evidence_summary, update_history
    decision ∈ {trusted_match, ambiguous_review, unknown,
                insufficient_evidence, non_tiger, blank}

RESPONSIBILITIES / IMPLEMENTATION SEQUENCE
1. src/ingestion.py
   - ingest_folder(path) -> list[ImageRecord]
   - Recursively scan directory, compute file hash, read available
     metadata (station_id/lat/lon/timestamp/camera_status if present in a
     sidecar CSV, else demo-generated), preserve originals — NEVER delete.
   - Fallback: if no metadata CSV, assign synthetic demo metadata and mark
     data_mode="demo".

2. src/triage.py
   - triage_image(record) -> TriageRecord
   - Use a lightweight blank/non-blank classifier if quick to set up;
     fallback: filename/demo label or a simple brightness+edge heuristic.
   - Blank images get quarantined (moved/flagged), never deleted.

3. src/perception.py
   - detect_subject(record) -> DetectionRecord
   - generate_embedding(crop) -> list[float]
   - Recommended order: pretrained torchvision ResNet-50/ConvNeXt first;
     OpenCLIP only if installation is quick and reliable; wildlife-specific
     embedding only if it installs in minutes; otherwise deterministic
     demo crop + deterministic pseudo-embedding (e.g. seeded from file hash)
     so the pipeline never blocks on a missing model.
   - Compute quality features: blur (Laplacian variance), brightness,
     contrast, crop area, flank visibility proxy.
   - A poor image must produce insufficient evidence downstream, not a
     forced identity — encode this via low quality_score, not by guessing.

4. src/identity.py
   - generate_candidates(embedding, catalogue) -> list[IdentityCandidate]
   - Use sklearn.NearestNeighbors with cosine similarity before reaching
     for FAISS. Return top-3 candidates plus the top-1/top-2 margin.
   - This function must NOT decide trusted/ambiguous/unknown — it only
     produces ranked evidence.

5. src/gating.py
   - compute_evidence(candidate, context) -> evidence score E, using:
       E = 0.55*V + 0.15*Q + 0.15*S + 0.10*T + 0.05*H
     where V=visual similarity, Q=quality/flank score, S=spatial
     feasibility, T=temporal feasibility, H=history consistency.
     Make all five weights configurable constants — never hardcode them
     as "final." State clearly in a comment: these are prototype
     heuristics, not scientifically validated parameters.
   - make_identity_decision(evidence, context) -> IdentityDecision using
     configurable thresholds, suggested starting point:
       E >= 0.80 and no major conflict -> trusted_match
       0.55 <= E < 0.80               -> ambiguous_review
       E < 0.55                       -> unknown (or rejected)
       severe quality/camera/metadata failure -> insufficient_evidence
   - Populate reason_codes from a fixed vocabulary, e.g.
     LOW_VISUAL_MARGIN, POOR_IMAGE_QUALITY, FLANK_NOT_VISIBLE,
     CAMERA_RELOCATED, MISSING_TIMESTAMP, MISSING_LOCATION,
     INSUFFICIENT_HISTORY, TRAVEL_SPEED_IMPLAUSIBLE, NEW_STATION.
   - update_history must be True only when decision == trusted_match.
     This is the one rule you must never weaken, bypass, or leave to
     another module's judgment.
   - Also implement basic unknown-individual handling: store unknown
     embeddings separately; if several are mutually similar, assign a
     provisional id like NEW-001 — do NOT add it to the trusted catalogue.

REQUIRED EXPOSED INTERFACE (per PROJECT_CONTRACT.md Section 19)
  process_image_directory(path)       # orchestrates ingest->triage->
                                       # perception->candidates for a folder
  generate_candidates(image_record)
  make_identity_decision(...)
These three are what Developer 1's pipeline.py and Developer 3 will call.
Keep their signatures stable once agreed; if you must change one, stop and
flag it per the global rules.

DEMO FALLBACK
Every function above must work with zero real images and zero installed
pretrained models — falling back to deterministic, seeded synthetic
values so process_image_directory() never crashes and never blocks on a
missing dependency. Mark all such outputs with data_mode="demo".

TESTING (tests/, append — don't rewrite shared test files)
- test_schemas.py: IdentityDecision only allows the six defined states.
- test_perception.py: quality features return sane ranges (0–1 or defined
  bounds) on a real and a synthetic image.
- test_gating.py:
    - E >= 0.80 with no conflicts -> trusted_match, update_history=True
    - low E -> unknown/ambiguous, update_history=False
    - severe quality failure -> insufficient_evidence, update_history=False
- Camera-relocation / poor-quality / missing-metadata scenarios each
  produce the correct reason code.

ACCEPTANCE CRITERIA (P0)
- process_image_directory(demo_folder) runs to completion on CPU with no
  GPU, no paid API, no model training, and no crash even with zero
  pretrained models available.
- Blank images are quarantined, not deleted.
- Every IdentityDecision has confidence, reason_codes, evidence_summary,
  top_candidates, and update_history correctly set per the rule above.
- pytest passes for all tests you own.

GIT
- Branch: backend-perception
- Commit at each of the 5 implementation-sequence milestones.
- Do not modify src/history.py, src/movement.py, src/alerts.py,
  src/evaluation.py, app.py, or ui/*.

WHAT NOT TO BUILD IN PHASE 1
Local stripe/flank matching beyond a simple proxy, sophisticated unknown
clustering, calibration curves, conformal prediction, FAISS (unless
sklearn genuinely can't keep up), training any model from scratch.

END-OF-SESSION REPORT
State: which perception steps used a real pretrained model vs. the
deterministic demo fallback; final gating thresholds used; pytest results;
any schema field you needed but wasn't in schemas.py.
```

---

# SECTION 4 — MASTER PROMPT — DEVELOPER 3 — BACKEND B — HISTORY + MOVEMENT + EVALUATION

```
ROLE
You are Developer 3 on a three-developer SIH hackathon team. You own
trusted longitudinal history, movement analysis, alert generation/
suppression, and the baseline-vs-proposed evaluation. You consume
IdentityDecision records produced by Developer 2 — you do not re-decide
identity, and you do not build a second hidden classifier.

OBJECTIVE (PHASE 1 ONLY)
Turn a stream of IdentityDecision records into: (a) a trusted-only
observation history, (b) basic movement-deviation detection, (c) alerts
that are correctly generated or suppressed, and (d) a baseline vs.
evidence-gated comparison on the same set of scenarios.

WHAT TO INSPECT FIRST
1. Read PROJECT_CONTRACT.md in full — Section 5 (critical safety rule) and
   Sections 14–15 (alert types and suppression) apply directly to you.
2. Check src/schemas.py for Observation and MovementAlert definitions; if
   absent, propose the fields below and flag for confirmation.
3. Check whether Developer 2's IdentityDecision output already exists (real
   or mock) to build/test against.

FILES YOU OWN
  src/history.py
  src/movement.py
  src/alerts.py
  src/evaluation.py

FILES YOU MUST NOT MODIFY
  app.py, ui/*                                        (Developer 1)
  src/ingestion.py, src/triage.py, src/perception.py,
  src/identity.py, src/gating.py                       (Developer 2)
  src/schemas.py, src/pipeline.py — shared, minimal changes only, flag
  before depending on any change.

SHARED SCHEMA FIELDS YOU PRODUCE / CONSUME

  Observation: observation_id, image_id, identity_id, station_id,
    latitude, longitude, timestamp, identity_confidence,
    observation_status, camera_status, quality_score

  MovementAlert: alert_id, identity_id, alert_type, confidence, status,
    evidence_observation_ids, explanation, suppression_reason
    alert_type ∈ {NEW_STATION, OUTSIDE_HISTORICAL_AREA, UNUSUAL_TRAVEL,
                  PROLONGED_ABSENCE, BUFFER_OR_VILLAGE_ADJACENT,
                  POSSIBLE_DISPERSAL, INSUFFICIENT_EVIDENCE,
                  CAMERA_OR_SURVEY_ARTEFACT}

CRITICAL SAFETY RULE — NEVER WEAKEN THIS
Only IdentityDecision.decision == "trusted_match" (equivalently
update_history == True) may create/update a trusted Observation.
ambiguous_review, unknown, insufficient_evidence, rejected, and
provisional decisions may be stored/displayed but must never silently
enter the trusted longitudinal history used for movement analysis.

RESPONSIBILITIES / IMPLEMENTATION SEQUENCE
1. src/history.py
   - update_trusted_history(decision) -> Observation | None
     Returns None (and does nothing to trusted history) unless
     decision.update_history is True. This function is the enforcement
     point for the safety rule — put an explicit guard clause at the top,
     not buried logic.
   - Maintain per-tiger: capture_count, first_seen, last_seen,
     trusted_stations, activity_centroid, historical capture area
     (a simple bounding polygon/convex hull over trusted stations — call
     it "historical capture area," never "home range" or "validated home
     range" anywhere in code, comments, or UI-facing strings).
   - compute_individual_summary(identity_id) -> IndividualSummary with the
     fields above plus camera-effort history and last-seen duration.

2. src/movement.py
   - Compute distance between successive trusted observations for a given
     identity_id, and implied travel speed.
   - Detect: new station (first trusted appearance at a station), outside
     historical capture area, unusually large travel distance/time,
     proximity to a buffer/village-adjacent station (if such stations are
     flagged in metadata), prolonged absence (no trusted observation for
     longer than a configurable threshold while cameras remained active).

3. src/alerts.py
   - generate_alerts(observations, station_context) -> list[MovementAlert]
   - Generate an alert ONLY when ALL of the following hold:
       - identity is trusted;
       - camera was active at the time;
       - station was not recently relocated;
       - image quality was adequate;
       - there is sufficient historical data for that individual;
       - the deviation exceeds a configurable threshold.
   - Otherwise: downgrade to INSUFFICIENT_EVIDENCE, or suppress the alert
     entirely with a filled-in suppression_reason (e.g. "camera relocated
     14 days ago," "only one trusted observation on record," "camera
     inactive during absence window").
   - Every alert's explanation field must plainly state which evidence
     (identity confidence, station history, camera status, deviation
     size) produced it — this is what the Developer 1 Alerts page renders.
   - Alerts must distinguish, in the explanation/status: likely biological
     signal vs. likely observation/survey artefact vs. insufficient
     evidence vs. human review required.

4. src/evaluation.py
   - Implement both pipelines over the same set of scenarios/observations:
       BASELINE (always-assign): every IdentityCandidate top-1 becomes a
         trusted observation unconditionally, feeding directly into
         movement + alerts.
       PROPOSED (evidence-gated): only IdentityDecision.decision ==
         trusted_match feeds trusted history, exactly as implemented
         above.
   - run_evaluation(records) -> EvaluationReport with, where labels
     permit: false confident-identity rate, coverage, abstention/review
     rate, false movement-alert rate, alert precision, artefact-
     suppression rate, percentage of observations withheld from trusted
     history.
   - Do not fabricate numbers. If a metric can't be computed on available
     demo data, report it as "not computable on current data" rather than
     inventing a plausible-looking value.
   - Frame all output as "prototype scenario evaluation," never "field
     validation" or "Pench performance."

REQUIRED EXPOSED INTERFACE (per PROJECT_CONTRACT.md Section 19)
  create_observation(...)          # wraps update_trusted_history
  generate_movement_alerts(...)    # wraps movement.py + alerts.py
  run_evaluation(...)
Keep these signatures stable once agreed with Developer 1; flag before
changing.

REQUIRED PHASE 1 SCENARIOS (minimum set — build a small scenario runner,
not a giant scenario engine)
1. normal trusted observation; 2. new station; 3. camera relocation;
4. prolonged absence with active cameras; 5. prolonged absence with
inactive cameras; 6. missing GPS/timestamp; 7. an ambiguous_review case
that a baseline pipeline would have wrongly trusted (this is the single
most important scenario — it's the direct evidence for your research
question).

DEMO FALLBACK
All of the above must run against synthetic demo IdentityDecision/
Observation records if Developer 2's real output isn't ready yet — build
a small local demo-decision generator for standalone testing, but replace
it with real Developer 2 output the moment it's available.

TESTING (tests/, append — don't rewrite shared test files)
- test_history.py: a non-trusted decision never creates/updates an
  Observation; a trusted decision does.
- test_alerts.py: camera relocation suppresses an alert that would
  otherwise fire; a single isolated observation does not trigger
  OUTSIDE_HISTORICAL_AREA; a genuine large jump with sufficient history
  does trigger an alert with a non-empty explanation.
- test_integration.py (shared — append only): baseline vs. proposed
  pipelines run on the same demo scenario set and produce a comparison
  report without crashing.

ACCEPTANCE CRITERIA (P0)
- Non-trusted decisions provably never contaminate trusted history
  (covered by a passing test, not just informal checking).
- A simulated new-station or unusual-travel event is detected.
- A camera-relocation false alert is suppressed with a stated reason.
- Baseline and evidence-gated pipelines can be run and compared on the
  same scenario set.
- pytest passes for all tests you own.

GIT
- Branch: backend-history
- Commit at each of the 4 implementation-sequence milestones.
- Do not modify src/ingestion.py, src/triage.py, src/perception.py,
  src/identity.py, src/gating.py, app.py, or ui/*.

WHAT NOT TO BUILD IN PHASE 1
Sophisticated seasonal modelling, a scientifically validated home-range
method, advanced GIS, conformal prediction bands, unseen-camera/temporal
splits (P1/P2 stretch — only if P0 is stable with time remaining).

END-OF-SESSION REPORT
State: pytest results; confirmation the safety-rule test passes; which
scenarios from the required list are covered; baseline vs. gated numbers
produced (and which, if any, are "not computable on current data" rather
than invented).
```
