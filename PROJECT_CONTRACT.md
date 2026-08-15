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

```mermaid
flowchart LR
    P0["P0<br/>Make it work"] --> P1["P1<br/>Make the experiment credible"]
    P1 --> P2["P2<br/>Add research / robustness features"]
    P2 --> P3["P3<br/>Beautify & prepare for judging"]

    style P0 fill:#2e7d32,color:#fff,stroke:#1b5e20
    style P1 fill:#1565c0,color:#fff,stroke:#0d47a1
    style P2 fill:#8e24aa,color:#fff,stroke:#4a148c
    style P3 fill:#616161,color:#fff,stroke:#212121
```

If time becomes limited, **P2 and P3 features are removed first**.

**P0 must never be sacrificed for advanced features.**

---

# 3. THREE-DEVELOPER ARCHITECTURE

```mermaid
flowchart TB
    subgraph D1["Developer 1 — Frontend + Integration"]
        UI[Streamlit app · UI pages · review queue<br/>evidence panels · movement viz · alerts<br/>evaluation dashboard · final integration]
    end

    subgraph D2["Developer 2 — Backend A: Perception + Identity"]
        direction LR
        A1[Ingestion] --> A2[Triage] --> A3[Detection / Crop] --> A4[Quality] --> A5[Embedding] --> A6[Candidate Generation] --> A7[Identity Gating]
    end

    subgraph D3["Developer 3 — Backend B: History + Movement + Evaluation"]
        direction LR
        B1[Observation] --> B2[Trusted History] --> B3[Movement Analysis] --> B4[Alerts]
    end

    A7 -->|IdentityDecision| B1
    D2 -.consumed by.-> D1
    D3 -.consumed by.-> D1

    style D1 fill:#fff3e0,stroke:#e65100
    style D2 fill:#e3f2fd,stroke:#0d47a1
    style D3 fill:#e8f5e9,stroke:#1b5e20
```

**Developer 1** does NOT implement ML algorithms.

**Developer 2** outputs: `IdentityDecision`

**Developer 3** is also responsible for **Baseline vs Evidence-Gated evaluation**.

---

# 4. INTEGRATION CONTRACT

All developers MUST use:

`src/schemas.py`

as the single source of truth.

Do not independently redefine data structures.

### Core data flow

```mermaid
flowchart LR
    A[ImageRecord] --> B[TriageRecord] --> C[DetectionRecord] --> D[IdentityCandidate] --> E[IdentityDecision] --> F[Observation] --> G[MovementAlert]

    style A fill:#eceff1,stroke:#37474f
    style B fill:#eceff1,stroke:#37474f
    style C fill:#eceff1,stroke:#37474f
    style D fill:#eceff1,stroke:#37474f
    style E fill:#ffe0b2,stroke:#e65100
    style F fill:#eceff1,stroke:#37474f
    style G fill:#ffcdd2,stroke:#b71c1c
```

---

# 5. CRITICAL SAFETY RULE

```mermaid
flowchart LR
    T[trusted_match] -->|MAY update| H[(Trusted Longitudinal History)]

    subgraph Blocked["MUST NOT update trusted history"]
        direction LR
        S1[ambiguous_review]
        S2[unknown]
        S3[insufficient_evidence]
        S4[rejected]
        S5[provisional]
    end

    Blocked -.blocked.-> H

    style T fill:#2e7d32,color:#fff,stroke:#1b5e20
    style H fill:#0d47a1,color:#fff,stroke:#0d47a1
    style Blocked fill:#ffebee,stroke:#c62828
```

**ONLY** `trusted_match` may update trusted longitudinal history.

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

**Demo data MUST NEVER be presented as real Pench observations.**

Real public datasets can be integrated later.

Development must NOT stop while waiting for a perfect dataset.

---

# 8. BASELINE

```mermaid
flowchart LR
    I1[Image] --> P1[Identity Prediction] --> A1[Always Assign] --> H1[History] --> M1[Movement] --> AL1[Alert]

    style A1 fill:#ffcdd2,stroke:#b71c1c
```

This is the baseline.

---

# 9. PROPOSED SYSTEM

```mermaid
flowchart LR
    I2[Image] --> C2[Candidate Generation] --> G2[Identity Gating] --> H2[Trusted History Only] --> M2[Movement] --> D2{Alert / Suppress / Review}

    style G2 fill:#c8e6c9,stroke:#1b5e20
    style H2 fill:#c8e6c9,stroke:#1b5e20
    style D2 fill:#fff3e0,stroke:#e65100
```

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

**Never fabricate results.**

---

# 11. IDENTITY STATES

```mermaid
flowchart TD
    Img[Incoming Image] --> Decision{Identity Decision}
    Decision --> TM[trusted_match]
    Decision --> AR[ambiguous_review]
    Decision --> UK[unknown]
    Decision --> IE[insufficient_evidence]
    Decision --> NT[non_tiger]
    Decision --> BL[blank]

    style TM fill:#2e7d32,color:#fff
    style AR fill:#f9a825,color:#000
    style UK fill:#ef6c00,color:#fff
    style IE fill:#c62828,color:#fff
    style NT fill:#616161,color:#fff
    style BL fill:#9e9e9e,color:#fff
```

Every decision must contain:

- confidence;
- reason codes;
- evidence summary;
- top candidates;
- `update_history`.

---

# 12. PROTOTYPE EVIDENCE SCORE

Initial configurable heuristic:

```
E = 0.55V + 0.15Q + 0.15S + 0.10T + 0.05H
```

| Symbol | Meaning |
|---|---|
| V | visual evidence |
| Q | image quality |
| S | spatial feasibility |
| T | temporal feasibility |
| H | history consistency |

These weights are **prototype heuristics**. They are **NOT** scientifically validated parameters.

Thresholds must be configurable.

---

# 13. MOVEMENT TERMINOLOGY

Unless a scientifically validated home-range method is implemented,
use:

> "historical capture area"

instead of:

> "validated home range".

Never claim behavioural change from one isolated observation.

---

# 14. ALERT TYPES

```mermaid
flowchart TD
    Alerts[Alert Types] --> A1[NEW_STATION]
    Alerts --> A2[OUTSIDE_HISTORICAL_AREA]
    Alerts --> A3[UNUSUAL_TRAVEL]
    Alerts --> A4[PROLONGED_ABSENCE]
    Alerts --> A5[BUFFER_OR_VILLAGE_ADJACENT]
    Alerts --> A6[POSSIBLE_DISPERSAL]
    Alerts --> A7[INSUFFICIENT_EVIDENCE]
    Alerts --> A8[CAMERA_OR_SURVEY_ARTEFACT]
```

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

**Preferred:**

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

**Optional:**

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

```
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
```

---

# 18. OWNERSHIP

```mermaid
flowchart TB
    subgraph Dev1["Developer 1"]
        O1[app.py]
        O2[ui/]
        O3[frontend / integration code]
    end

    subgraph Dev2["Developer 2"]
        O4[ingestion.py]
        O5[triage.py]
        O6[perception.py]
        O7[identity.py]
        O8[gating.py]
    end

    subgraph Dev3["Developer 3"]
        O9[history.py]
        O10[movement.py]
        O11[alerts.py]
        O12[evaluation.py]
    end

    subgraph Shared["Shared (minimal changes only)"]
        O13[schemas.py]
        O14[pipeline.py]
    end

    style Shared fill:#fff9c4,stroke:#f57f17
```

Any schema change must be communicated to all developers before merging.

---

# 19. SHARED INTERFACES

```mermaid
flowchart LR
    subgraph Dev2API["Developer 2 exposes"]
        F1["process_image_directory(path)"]
        F2["generate_candidates(image_record)"]
        F3["make_identity_decision(...)"]
    end

    subgraph Dev3API["Developer 3 exposes"]
        F4["create_observation(...)"]
        F5["generate_movement_alerts(...)"]
        F6["run_evaluation(...)"]
    end

    Dev2API --> Dev1[Developer 1 UI]
    Dev3API --> Dev1
```

Developer 1 must consume these interfaces rather than reimplementing
their logic.

---

# 20. INTEGRATION ORDER

```mermaid
flowchart LR
    S1["1. Schemas"] --> S2["2. Demo data"] --> S3["3. Backend A"] --> S4["4. Backend B"] --> S5["5. Pipeline"] --> S6["6. Frontend"] --> S7["7. Baseline comparison"] --> S8["8. Advanced features"] --> S9["9. UI polish"]

    style S1 fill:#1565c0,color:#fff
    style S2 fill:#1565c0,color:#fff
    style S3 fill:#2e7d32,color:#fff
    style S4 fill:#2e7d32,color:#fff
    style S5 fill:#6a1b9a,color:#fff
    style S6 fill:#e65100,color:#fff
    style S7 fill:#e65100,color:#fff
    style S8 fill:#757575,color:#fff
    style S9 fill:#757575,color:#fff
```

**Do NOT build the complete UI before the backend pipeline works.**

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

```
pytest
```

before merging.

---

# 22. GIT RULES

```mermaid
gitGraph
    commit id: "main: initial"
    branch dev1-frontend
    branch dev2-backend-a
    branch dev3-backend-b
    checkout dev2-backend-a
    commit id: "perception + identity"
    checkout dev3-backend-b
    commit id: "history + movement"
    checkout dev1-frontend
    commit id: "streamlit UI"
    checkout main
    merge dev2-backend-a tag: "PR review"
    merge dev3-backend-b tag: "PR review"
    merge dev1-frontend tag: "PR review"
```

`main` is protected.

- Developers work only on their assigned branches.
- No direct pushes to `main`.
- All changes enter `main` through Pull Requests.
- Do not force-push shared branches.
- Do not rewrite another developer's code unnecessarily.

---

# 23. FEATURE PRIORITY

```mermaid
flowchart TB
    subgraph P0["P0 — must have"]
        p0a[Complete end-to-end pipeline]
        p0b[Demo mode]
        p0c[Schemas]
        p0d[Identity gating]
        p0e[Trusted history]
        p0f[Movement]
        p0g[Alerts]
        p0h[Baseline]
        p0i[Evaluation]
        p0j[CPU fallback]
    end

    subgraph P1["P1 — should have"]
        p1a[Better embeddings]
        p1b[Better detection]
        p1c[Image quality]
        p1d[Camera relocation]
        p1e[Review queue]
        p1f[Stress tests]
    end

    subgraph P2["P2 — nice to have"]
        p2a[Unknown clustering]
        p2b[Calibration]
        p2c[Unseen-camera evaluation]
        p2d[Temporal evaluation]
        p2e[Advanced spatial analysis]
    end

    subgraph P3["P3 — stretch"]
        p3a[Conformal prediction]
        p3b[Sophisticated GIS]
        p3c[Animations]
        p3d[Advanced visual polish]
    end

    P0 --> P1 --> P2 --> P3

    style P0 fill:#c8e6c9,stroke:#1b5e20
    style P1 fill:#bbdefb,stroke:#0d47a1
    style P2 fill:#e1bee7,stroke:#4a148c
    style P3 fill:#eeeeee,stroke:#616161
```

---

# 24. CUT RULE

If time is running out:

**CUT (in order):**

1. conformal prediction;
2. advanced GIS;
3. sophisticated clustering;
4. animations;
5. decorative UI.

**DO NOT CUT:**

1. identity gating;
2. trusted-only history;
3. baseline comparison;
4. alert suppression;
5. explanations;
6. demo fallback;
7. tests.

---

# 25. DEFINITION OF DONE

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

The objective is **NOT**:

> "assign an identity to every image."

The objective is:

> Know when the evidence is strong enough for an observation to
> influence a downstream management conclusion.
