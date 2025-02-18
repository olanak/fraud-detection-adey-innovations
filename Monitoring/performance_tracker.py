from prometheus_client import start_http_server, Gauge
import time

class PerformanceMonitor:
    def __init__(self):
        self.prediction_count = Gauge('fraud_predictions_total', 'Total predictions')
        self.fraud_count = Gauge('fraud_positive_total', 'Fraudulent transactions')
        self.response_time = Gauge('api_response_seconds', 'API response time')
        self.data_drift = Gauge('data_drift_score', 'Overall data drift score')

    def log_prediction(self, is_fraud: bool):
        self.prediction_count.inc()
        if is_fraud:
            self.fraud_count.inc()

    def log_response_time(self, duration: float):
        self.response_time.set(duration)

    def update_drift_score(self, score: float):
        self.data_drift.set(score)

# Start metrics server on port 8000
start_http_server(8000)