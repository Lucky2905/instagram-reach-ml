# Instagram Reach ML

Predict post reach (regression) and classify reach tier (low/medium/high) using Scikit-learn, Flask, and a Chart.js dashboard — structured with 4 Gang-of-Four design patterns.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (5,000 rows)
python src/data/generator.py

# 3. Train both models
python src/train.py

# 4. Start the Flask server
python app.py

# 5. Open dashboard
# http://localhost:5000
```

## Project Structure

```
instagram-reach-ml/
├── app.py                        Flask server entry point
├── config.py                     Central configuration
├── requirements.txt
├── src/
│   ├── train.py                  Training CLI (python src/train.py)
│   ├── data/
│   │   ├── generator.py          Synthetic data generator (5,000 rows)
│   │   ├── loader.py             DataLoader (CSV or auto-generate)
│   │   └── feature_engineer.py  Feature encoding
│   ├── patterns/
│   │   ├── factory.py            Factory Pattern  → ModelFactory
│   │   ├── strategy.py           Strategy Pattern → PreprocessingContext
│   │   ├── observer.py           Observer Pattern → TrainingSubject
│   │   └── decorator.py         Decorator Pattern → CV + FeatureSelection
│   ├── models/
│   │   ├── regressor.py          ReachRegressor (LinearRegression)
│   │   └── classifier.py        TierClassifier (RandomForest)
│   ├── training/
│   │   └── trainer.py           Training pipeline orchestrator
│   └── api/
│       ├── app.py               Flask app factory
│       └── routes.py            REST endpoints
├── dashboard/
│   ├── templates/index.html     Interactive web dashboard
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/                        Generated CSV stored here
├── saved_models/                Trained .pkl files
└── tests/
    ├── test_pipeline.py
    ├── test_models.py
    └── test_api.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/health` | Model load status |
| POST | `/predict` | Single prediction |
| POST | `/predict-batch` | Batch CSV predictions |
| GET | `/metrics` | Training history |
| POST | `/train` | Trigger training via API |

### POST /predict

```json
// Request
{
  "likes": 1500, "comments": 80, "shares": 25, "saves": 60,
  "hashtag_count": 12, "post_type": "reel",
  "hour_of_day": 19, "day_of_week": 5,
  "follower_count": 50000, "account_age_days": 730
}

// Response
{
  "predicted_reach": 34200,
  "reach_tier": "high",
  "confidence": 0.91
}
```

## Design Patterns

| Pattern | File | Purpose |
|---------|------|---------|
| **Factory** | `src/patterns/factory.py` | `ModelFactory.create("random_forest")` — decouples callers from sklearn |
| **Strategy** | `src/patterns/strategy.py` | Swap `normalize` ↔ `standardize` ↔ `robust` with zero code changes |
| **Observer** | `src/patterns/observer.py` | Training events logged/persisted without touching model code |
| **Decorator** | `src/patterns/decorator.py` | `CrossValidationDecorator` adds `.cv_score()` to any model |

## Swap Preprocessing Strategy (zero code change)

```bash
# Change via CLI flag only — train.py is untouched
python src/train.py --strategy normalize
python src/train.py --strategy robust
```

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Success Criteria

| Check | Target |
|-------|--------|
| Dataset | 5,000 rows, 0 nulls |
| R² Score | ≥ 0.75 |
| Accuracy | ≥ 0.80 |
| API response | < 200ms |
| Dashboard charts | 3 Chart.js visualisations |
