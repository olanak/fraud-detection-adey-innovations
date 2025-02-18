import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.features = reference_data.columns

    def calculate_drift(self, current_data: pd.DataFrame):
        results = {}
        for feature in self.features:
            ref = self.reference[feature]
            curr = current_data[feature]
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(ref, curr)
            
            # PSI calculation
            psi = self._calculate_psi(ref, curr)
            
            results[feature] = {
                'ks_stat': ks_stat,
                'psi': psi,
                'threshold_exceeded': psi > 0.2 or ks_stat > 0.1
            }
        return results

    def _calculate_psi(self, ref, curr, bins=10):
        # Calculate Population Stability Index
        ref_percents = np.histogram(ref, bins=bins)[0]/len(ref)
        curr_percents = np.histogram(curr, bins=bins)[0]/len(curr)
        return np.sum((curr_percents - ref_percents) * np.log(curr_percents/ref_percents))