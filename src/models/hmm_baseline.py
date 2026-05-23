"""Hidden Markov Model baseline for RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - exercised only without optional dependency
    GaussianHMM = None


@dataclass
class HMMRULPredictor:
    """Gaussian HMM baseline that maps hidden degradation states to normalized RUL."""

    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 200
    random_state: int = 42
    model: object | None = field(default=None, init=False)
    state_rul_: np.ndarray | None = field(default=None, init=False)

    def _make_model(self, n_components: int | None = None):
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required for HMMRULPredictor")
        return GaussianHMM(
            n_components=n_components or self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    @staticmethod
    def _as_sequences(sequences: list[np.ndarray]) -> list[np.ndarray]:
        prepared = [np.asarray(seq, dtype=float) for seq in sequences if len(seq) > 0]
        if not prepared:
            raise ValueError("At least one non-empty sequence is required")
        feature_count = prepared[0].shape[1]
        for seq in prepared:
            if seq.ndim != 2 or seq.shape[1] != feature_count:
                raise ValueError("All sequences must have shape (T, n_features)")
        return prepared

    @staticmethod
    def _default_true_rul(length: int) -> np.ndarray:
        if length <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(1.0, 0.0, length, dtype=float)

    def fit(self, sequences: list[np.ndarray], lengths: list[int] | None = None) -> "HMMRULPredictor":
        """Fit the HMM and calibrate hidden states to normalized RUL."""
        prepared = self._as_sequences(sequences)
        inferred_lengths = [len(seq) for seq in prepared]
        if lengths is not None and list(lengths) != inferred_lengths:
            raise ValueError("lengths must match provided sequence lengths")

        x_train = np.vstack(prepared)
        y_train = np.concatenate([self._default_true_rul(len(seq)) for seq in prepared])

        self.model = self._make_model()
        self.model.fit(x_train, inferred_lengths)
        posterior = self.model.predict_proba(x_train)

        state_rul = np.zeros(self.n_components, dtype=float)
        for state_idx in range(self.n_components):
            weights = posterior[:, state_idx]
            if float(weights.sum()) <= 1e-12:
                state_rul[state_idx] = 0.0
            else:
                state_rul[state_idx] = float(np.average(y_train, weights=weights))
        self.state_rul_ = np.clip(state_rul, 0.0, 1.0)
        return self

    def predict_rul(self, sequence: np.ndarray) -> np.ndarray:
        """Predict normalized RUL for one degradation sequence."""
        if self.model is None or self.state_rul_ is None:
            raise RuntimeError("Call fit before predict_rul")
        x = np.asarray(sequence, dtype=float)
        posterior = self.model.predict_proba(x)
        return np.clip(posterior @ self.state_rul_, 0.0, 1.0)

    def evaluate(self, sequences: list[np.ndarray], true_ruls: list[np.ndarray]) -> dict[str, float]:
        """Evaluate HMM RUL predictions on multiple sequences."""
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for sequence, true_rul in zip(sequences, true_ruls):
            prediction = self.predict_rul(sequence)
            y_true = np.asarray(true_rul, dtype=float).reshape(-1)
            if prediction.shape[0] != y_true.shape[0]:
                raise ValueError("Prediction and target lengths must match")
            y_true_all.append(y_true)
            y_pred_all.append(prediction)

        y_true_concat = np.concatenate(y_true_all)
        y_pred_concat = np.concatenate(y_pred_all)
        mse = float(mean_squared_error(y_true_concat, y_pred_concat))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_concat, y_pred_concat))
        r2 = float(r2_score(y_true_concat, y_pred_concat))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def select_n_components(
        self,
        sequences: list[np.ndarray],
        n_range: tuple[int, int] = (2, 8),
    ) -> tuple[int, dict[int, float]]:
        """Select the number of states by BIC; returns best K and BIC table."""
        prepared = self._as_sequences(sequences)
        lengths = [len(seq) for seq in prepared]
        x_train = np.vstack(prepared)
        n_samples, n_features = x_train.shape

        scores: dict[int, float] = {}
        for n_components in range(n_range[0], n_range[1] + 1):
            model = self._make_model(n_components=n_components)
            model.fit(x_train, lengths)
            log_likelihood = float(model.score(x_train, lengths))
            n_params = self._parameter_count(n_components, n_features)
            scores[n_components] = -2.0 * log_likelihood + n_params * np.log(n_samples)

        best_k = min(scores, key=scores.get)
        return int(best_k), scores

    def _parameter_count(self, n_components: int, n_features: int) -> int:
        start_params = n_components - 1
        transition_params = n_components * (n_components - 1)
        mean_params = n_components * n_features
        if self.covariance_type == "full":
            cov_params = n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = n_components * n_features
        else:
            cov_params = n_components * n_features
        return int(start_params + transition_params + mean_params + cov_params)

