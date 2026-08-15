# PROJECT CONTRACT
## Evidence-Aware Tiger Camera-Trap Movement Intelligence System

Version: 1.0
Status: MVP CONTRACT

---

## 1. PROJECT OBJECTIVE

Build a working prototype for:

Evidence-aware camera-trap triage, individual tiger identity
decision-making, trusted longitudinal history, movement analysis,
and alert management.

The project does NOT claim to invent:

- tiger detection;
- tiger Re-ID;
- stripe recognition;
- visual embeddings;
- GPS/time fusion;
- uncertainty estimation;
- abstention;
- GIS;
- home-range analysis.

The central experimental question is:

> Does an identity-gating layer, calibrated for the cost of false
> identity assignments, reduce downstream movement-alert errors
> compared with an always-assign pipeline under camera-trap domain
> shift and survey artefacts?

---

# 2. DEVELOPMENT PRIORITY

Everything must be developed in this order:

P0 — MAKE IT WORK

P1 — MAKE THE EXPERIMENT CREDIBLE

P2 — ADD RESEARCH/ROBUSTNESS FEATURES

P3 — BEAUTIFY AND PREPARE FOR JUDGING

If time becomes limited, P2 and P3 features are removed first.

P0 must never be sacrificed for advanced features.

---

# 3. THREE-DEVELOPER ARCHITECTURE

## Developer 1 — FRONTEND + INTEGRATION

Responsible for:

- Streamlit application;
- UI pages;
- displaying backend outputs;
- review queue;
- evidence panels;
- movement visualization;
- alerts;
- evaluation dashboard;
- final integration.

Developer 1 does NOT implement ML algorithms.

---

## Developer 2 — BACKEND A
## PERCEPTION + IDENTITY

Responsible for:

IMAGE
→ INGESTION
→ TRIAGE
→ DETECTION/CROP
→ QUALITY
→ EMBEDDING
→ CANDIDATE GENERATION
→ IDENTITY GATING

Outputs:

IdentityDecision

---

## Developer 3 — BACKEND B
## HISTORY + MOVEMENT + EVALUATION

Responsible for:

IdentityDecision
→ Observation
→ Trusted History
→ Movement Analysis
→ Alerts

Also responsible for:

Baseline vs Evidence-Gated evaluation.

---

# 4. INTEGRATION CONTRACT

All developers MUST use:

src/schemas.py

as the single source of truth.

Do not independently redefine data structures.

Core data flow:

ImageRecord
    ↓
TriageRecord
    ↓
DetectionRecord
    ↓
IdentityCandidate
    ↓
IdentityDecision
    ↓
Observation
    ↓
MovementAlert

---

# 5. CRITICAL SAFETY RULE

ONLY:

trusted_match

may update trusted longitudinal history.

These MUST NOT update trusted history:

- ambiguous_review;
- unknown;
- insufficient_evidence;
- rejected;
- provisional.

Uncertain observations may be displayed and reviewed but cannot
silently contaminate trusted history.

---

# 6. P0 MVP

The first working version must:

1. process an image folder;
2. create image records;
3. perform blank/nonblank triage;
4. produce detection/crop or deterministic demo equivalent;
5. generate candidate identities;
6. produce identity decisions;
7. route uncertain cases;
8. update trusted history only for trusted matches;
9. calculate basic movement deviation;
10. generate or suppress an alert;
11. explain the decision;
12. compare always-assign vs evidence-gated behaviour;
13. run without GPU.

---

# 7. DEMO-FIRST RULE

The system MUST work without a real tiger dataset.

Create a clearly labelled DEMO MODE.

Demo mode may use:

- synthetic metadata;
- controlled scenarios;
- deterministic candidate scores;
- sample images;
- simulated movement observations.

Demo data MUST NEVER be presented as real Pench observations.

Real public datasets can be integrated later.

Development must NOT stop while waiting for a perfect dataset.

---

# 8. BASELINE

Implement:

IMAGE
→ IDENTITY PREDICTION
→ ALWAYS ASSIGN
→ HISTORY
→ MOVEMENT
→ ALERT

This is the baseline.

---

# 9. PROPOSED SYSTEM

Implement:

IMAGE
→ CANDIDATE GENERATION
→ IDENTITY GATING
→ TRUSTED HISTORY ONLY
→ MOVEMENT
→ ALERT / SUPPRESS / REVIEW

---

# 10. EVALUATION

Compare the baseline and proposed system on the same scenarios.

Measure where labels permit:

- false confident identity rate;
- coverage;
- abstention/review rate;
- selective risk;
- false movement-alert rate;
- alert precision;
- artefact suppression;
- observations prevented from entering trusted history.

Never fabricate results.

---

# 11. IDENTITY STATES

Allowed identity decisions:

- trusted_match
- ambiguous_review
- unknown
- insufficient_evidence
- non_tiger
- blank

Every decision must contain:

- confidence;
- reason codes;
- evidence summary;
- top candidates;
- update_history.

---

# 12. PROTOTYPE EVIDENCE SCORE

Initial configurable heuristic:

E =
0.55V +
0.15Q +
0.15S +
0.10T +
0.05H

where:

V = visual evidence
Q = image quality
S = spatial feasibility
T = temporal feasibility
H = history consistency

These weights are prototype heuristics.

They are NOT scientifically validated parameters.

Thresholds must be configurable.

---

# 13. MOVEMENT TERMINOLOGY

Unless a scientifically validated home-range method is implemented,
use:

"historical capture area"

instead of:

"validated home range".

Never claim behavioural change from one isolated observation.

---

# 14. ALERT TYPES

Support:

- NEW_STATION
- OUTSIDE_HISTORICAL_AREA
- UNUSUAL_TRAVEL
- PROLONGED_ABSENCE
- BUFFER_OR_VILLAGE_ADJACENT
- POSSIBLE_DISPERSAL
- INSUFFICIENT_EVIDENCE
- CAMERA_OR_SURVEY_ARTEFACT

Alerts must distinguish:

- likely biological signal;
- likely observation artefact;
- insufficient evidence;
- human review required.

---

# 15. ALERT SUPPRESSION

Downgrade or suppress alerts when:

- identity confidence is insufficient;
- camera was relocated;
- camera effort is insufficient;
- image quality is poor;
- history is too short;
- metadata are missing;
- only one isolated observation exists.

---

# 16. TECHNOLOGY

Preferred:

- Python 3.11+
- Streamlit
- pandas
- NumPy
- Pydantic
- SQLite
- OpenCV
- Pillow
- scikit-learn
- Plotly
- PyTorch/torchvision

Optional:

- OpenCLIP
- Ultralytics YOLO
- GeoPandas
- Shapely
- pyproj
- Folium/PyDeck
- FAISS

No component may require a GPU.

No paid API.

No cloud-only dependency.

No large-model training from scratch.

Every ML component needs a deterministic/demo fallback.

---

# 17. REPOSITORY STRUCTURE

project/

├── PROJECT_CONTRACT.md
├── README.md
├── requirements.txt
├── app.py
│
├── data/
│   └── demo/
│
├── src/
│   ├── schemas.py
│   ├── pipeline.py
│   │
│   ├── ingestion.py
│   ├── triage.py
│   ├── perception.py
│   ├── identity.py
│   ├── gating.py
│   │
│   ├── history.py
│   ├── movement.py
│   ├── alerts.py
│   └── evaluation.py
│
├── ui/
│   ├── overview.py
│   ├── processing.py
│   ├── review.py
│   ├── movement.py
│   ├── alerts.py
│   └── evaluation.py
│
└── tests/

---

# 18. OWNERSHIP

Developer 1 owns:

- app.py
- ui/
- frontend/integration code

Developer 2 owns:

- ingestion.py
- triage.py
- perception.py
- identity.py
- gating.py

Developer 3 owns:

- history.py
- movement.py
- alerts.py
- evaluation.py

Shared:

- schemas.py
- pipeline.py

Shared files must have minimal changes.

Any schema change must be communicated to all developers before merging.

---

# 19. SHARED INTERFACES

Developer 2 must expose:

process_image_directory(path)

generate_candidates(image_record)

make_identity_decision(...)

Developer 3 must expose:

create_observation(...)

generate_movement_alerts(...)

run_evaluation(...)

Developer 1 must consume these interfaces rather than reimplementing
their logic.

---

# 20. INTEGRATION ORDER

Integration happens in this order:

1. schemas;
2. demo data;
3. backend A;
4. backend B;
5. pipeline;
6. frontend;
7. baseline comparison;
8. advanced features;
9. UI polish.

Do NOT build the complete UI before the backend pipeline works.

---

# 21. TESTING

Every developer must create unit tests.

At minimum:

- schema validation;
- candidate generation;
- identity decisions;
- trusted-only history;
- unknown handling;
- camera relocation;
- alert suppression;
- baseline vs gated comparison.

Run:

pytest

before merging.

---

# 22. GIT RULES

main is protected.

Developers work only on their assigned branches.

No direct pushes to main.

All changes enter main through Pull Requests.

Do not force-push shared branches.

Do not rewrite another developer's code unnecessarily.

---

# 23. FEATURE PRIORITY

P0:

- complete end-to-end pipeline;
- demo mode;
- schemas;
- identity gating;
- trusted history;
- movement;
- alerts;
- baseline;
- evaluation;
- CPU fallback.

P1:

- better embeddings;
- better detection;
- image quality;
- camera relocation;
- review queue;
- stress tests.

P2:

- unknown clustering;
- calibration;
- unseen-camera evaluation;
- temporal evaluation;
- advanced spatial analysis.

P3:

- conformal prediction;
- sophisticated GIS;
- animations;
- advanced visual polish.

---

# 24. CUT RULE

If time is running out:

CUT:

1. conformal prediction;
2. advanced GIS;
3. sophisticated clustering;
4. animations;
5. decorative UI.

DO NOT CUT:

1. identity gating;
2. trusted-only history;
3. baseline comparison;
4. alert suppression;
5. explanations;
6. demo fallback;
7. tests.

---

# 25. DEFINITION OF DONE

The prototype is considered functional when:

- application starts locally;
- images can be processed;
- blank images are quarantined;
- candidates are generated;
- uncertain identities can abstain;
- trusted identities update history;
- uncertain identities cannot contaminate history;
- movement deviations are detected;
- artefact alerts can be suppressed;
- decisions are explainable;
- baseline and gated systems can be compared;
- tests pass;
- system works without GPU;
- demo data are clearly labelled.

---

# 26. SCIENTIFIC HONESTY

Never claim:

- invented tiger Re-ID;
- invented stripe recognition;
- invented visual embeddings;
- invented GPS/time fusion;
- field validation without field data;
- real Pench performance without real Pench evaluation;
- scientifically validated home ranges from prototype data.

The project's defensible contribution is the evaluation of an
auditable identity-gating layer and its effect on downstream
longitudinal movement-alert reliability.

---

# 27. GOLDEN RULE

The objective is NOT:

"assign an identity to every image."

The objective is:

> Know when the evidence is strong enough for an observation to
> influence a downstream management conclusion.
