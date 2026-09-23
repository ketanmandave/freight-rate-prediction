### Freight Rate Prediction

Goal
Predict freight rates for 12,000 future loads.

Validation
Time-based holdout designed to simulate future-rate prediction.

Model
<selected after experiments>

Key features
Lane, distance, weight, equipment, temporal features,
market/quote information where available.

Run
pip install -r requirements.txt
python src/train.py
python src/predict.py
python score.py ...