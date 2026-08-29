# Model Lifecycle

## Stages

```mermaid
stateDiagram-v2
    [*] --> Development
    Development --> Validation
    Development --> Archived
    Validation --> Staging
    Validation --> Development
    Validation --> Archived
    Staging --> Production
    Staging --> Validation
    Staging --> Archived
    Production --> Archived
```

## Promotion Rules

| Transition | Requirements |
|-----------|-------------|
| Development → Validation | Manual promotion |
| Validation → Staging | PR-AUC ≥ 0.5 |
| Staging → Production | `validation_passed = true`, PR-AUC ≥ 0.5 |
| Production → Archived | Automatic when new model promoted |

**Production promotion requires explicit validation.** A candidate cannot reach production without passing validation checks.

## Registration

```bash
modelops model register <mlflow_run_id> --version 1.0.0
```

Stores:
- Artifact path
- Metrics
- Training configuration
- Git commit
- Lineage ID

## Rollback

```bash
modelops model rollback
```

Demotes current production to archived, promotes most recent archived version to production.

## Retraining Workflow

```mermaid
graph LR
    A[Monitoring Event] --> B[Create Candidate]
    B --> C[Train Model]
    C --> D[Validate vs Production]
    D --> E{Passed?}
    E -->|Yes| F[Stage for Review]
    E -->|No| G[Archive]
    F --> H[Manual Production Promotion]
```

Retraining does **not** automatically deploy. Human approval required for production replacement.

## Lineage

Every model traces back to:
- Dataset version
- Preprocessing version
- Experiment run ID
- Git commit
- Configuration file

Query: `GET /lineage/{record_id}`
