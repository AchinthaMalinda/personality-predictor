# 🧠 Personality Predictor — Introvert vs Extrovert

A machine learning model that predicts whether a person is an **Introvert** or **Extrovert** based on behavioral characteristics, trained using XGBoost and deployed on Hugging Face Spaces.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **91.6%** |
| ROC-AUC | **95.8%** |
| 5-Fold CV | **93.1% ± 1.7%** |

---

## 🗂️ Dataset

The dataset contains 2,900 records with 7 behavioral features:

| Feature | Description |
|---------|-------------|
| `Time_spent_Alone` | Hours spent alone per day (0–11) |
| `Stage_fear` | Has stage fear — Yes / No |
| `Social_event_attendance` | Social events attended per month (0–10) |
| `Going_outside` | Days going outside per week (0–7) |
| `Drained_after_socializing` | Feels drained after socializing — Yes / No |
| `Friends_circle_size` | Number of close friends (0–20) |
| `Post_frequency` | Social media posts per week (0–10) |

📥 **Dataset:** [Google Drive](https://drive.google.com/file/d/1GNx9VkdrFP5Tz3XsuBzscXbiYJeamBfS/view)

---

## 🤖 Model

- **Algorithm:** XGBoost Classifier
- **Preprocessing:** Label encoding for Yes/No columns only (XGBoost handles missing values natively)
- **Key Feature:** `Drained_after_socializing` — 91% feature importance

---

## 🚀 Live Demo

👉 [https://huggingface.co/spaces/AchinthaMalinda/personality-predictor](https://huggingface.co/spaces/AchinthaMalinda/personality-predictor)

---

## 📡 Public API

**Endpoint:**
```
POST https://achinthamalinda-personality-predictor.hf.space/run/predict
```

**Request body (JSON):**
```json
{
  "data": [5, "No", 5, 3, "No", 8, 4]
}
```

**Field order in `data` array:**
```
time_spent_alone, stage_fear, social_event_attendance, going_outside,
drained_after_socializing, friends_circle_size, post_frequency
```

**Response:**
```json
{
  "data": ["🎉 **Extrovert**\nConfidence: 92.3%"]
}
```

**Example — Python:**
```python
import requests

url = "https://achinthamalinda-personality-predictor.hf.space/run/predict"

payload = {
    "data": [9, "Yes", 1, 1, "Yes", 3, 2]
}

response = requests.post(url, json=payload)
print(response.json())
# → {"data": ["🪴 **Introvert**\nConfidence: 91.2%"]}
```

**Example — curl:**
```bash
curl -X POST \
  https://achinthamalinda-personality-predictor.hf.space/run/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [9, "Yes", 1, 1, "Yes", 3, 2]}'
```

---

## 📁 Repository Structure

```
personality-predictor/
├── notebook.ipynb        # Data preparation & model training
├── app.py                # Gradio app deployed on Hugging Face Spaces
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Tech Stack

- **Model:** XGBoost
- **Data:** Pandas, NumPy, Scikit-learn
- **Deployment:** Hugging Face Spaces + Gradio
- **Training:** Google Colab + Google Drive

---

## 👤 Author

**Achinthа Malinda**
📧 achinthamalinda329@gmail.com
