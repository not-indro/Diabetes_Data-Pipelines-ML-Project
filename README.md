## 2. Core Modeling Task

### Modeling Question

**Can we predict whether a patient will be readmitted within 30 days of discharge?**

### Why This Matters

* Hospitals face financial and quality penalties for high readmission rates.
* Early identification of high-risk patients enables targeted follow-ups and care coordination.
* This problem balances **clinical relevance** with **operational impact**.

### Target Engineering

* `readmitted` is converted to a binary label:

  * `<30` → 1 (early readmission)
  * `NO`, `>30` → 0

---

## 3. Modeling Approach

### Feature Handling

* Identifier columns (`encounter_id`, `patient_nbr`) are removed.
* Sparse administrative features are dropped using the feature dictionary.
* Admission and discharge IDs are mapped to descriptive categories using `IDS_mapping.csv`.

### Preprocessing

* Numerical features: median imputation
* Categorical features: most-frequent imputation + one-hot encoding
* Implemented via a single sklearn `Pipeline` to prevent leakage.

### Model Choice

* **Logistic Regression** with `class_weight='balanced'` is used as a baseline.
* Chosen for:

  * interpretability
  * stability on noisy healthcare data
  * ease of deployment

### Evaluation

* Primary metric: **Recall**
* Rationale: missing a high-risk patient is more costly than a false positive.
* Confusion matrix and classification report are used for error analysis.

---

## 4. Additional Business Questions

### 1. Which patients are likely to become high-utilization patients?

**Value:** Enables proactive care management and cost reduction.
**Approach:** Aggregate encounters at the patient level and model long-term utilization risk.

### 2. What factors drive prolonged hospital stays?

**Value:** Improves bed planning and staffing decisions.
**Approach:** Regression on `time_in_hospital` with diagnosis and admission context.

### 3. How do medication changes impact readmission risk?

**Value:** Helps refine discharge planning and follow-up strategies.
**Approach:** Analyze medication change indicators using interpretable models.

---

## 5. Minimal Pipeline & Service

### What’s Included

* Cleaned data persisted to SQLite
* Trained model saved as an artifact
* Lightweight FastAPI service with:

  * `GET /health`
  * `POST /predict`

This mirrors how models are typically deployed internally without external hosting.

---

## 6. How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:

* clean the data
* train the model
* save the model artifact
* persist cleaned data to SQLite

### 3. Start the API

```bash
uvicorn api:app --reload
```

Service will be available at:

```
http://localhost:8000
```

---

## 7. Example API Request / Response

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{ "status": "ok" }
```

---

### Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '[
  {
    "age": "[60-70)",
    "gender": "Male",
    "time_in_hospital": 4,
    "num_lab_procedures": 45,
    "num_medications": 13,
    "number_inpatient": 1,
    "admission_type": "Emergency",
    "discharge_disposition": "Discharged to home"
  }
]'
```

### Response

```json
[
  {
    "prediction": 1,
    "risk_score": 0.67
  }
]
```
