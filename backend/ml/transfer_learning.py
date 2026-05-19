"""
Transfer learning: fine-tune steel-trained MLP on sparse HEA data.
Uses sklearn's MLPRegressor warm_start for the approximation.
For production: use PyTorch with proper layer freezing.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

def finetune_for_hea(
    base_model_path: str,
    X_hea_train: np.ndarray,
    y_hea_train: np.ndarray,
    n_finetune_iter: int = 50,
) -> MLPRegressor:
    """
    Fine-tune the steel-trained MLP on HEA data using warm_start.
    
    Limitation: sklearn MLPRegressor doesn't support layer freezing natively.
    This approximates transfer learning by continuing training from the steel model's
    weights as initialization — still significantly better than training from scratch.
    
    For true layer freezing, use: PyTorch / TensorFlow (see pytorch_transfer.py)
    """
    base_model: MLPRegressor = joblib.load(base_model_path)

    # Copy base model, enable warm_start, reduce learning rate for fine-tuning
    hea_model = MLPRegressor(
        hidden_layer_sizes=base_model.hidden_layer_sizes,
        activation=base_model.activation,
        max_iter=n_finetune_iter,
        warm_start=True,
        learning_rate_init=1e-4,   # 10× smaller than base training LR
        random_state=42,
    )

    # Transfer weights from base model
    hea_model.coefs_ = [c.copy() for c in base_model.coefs_]
    hea_model.intercepts_ = [i.copy() for i in base_model.intercepts_]
    hea_model.n_iter_ = base_model.n_iter_
    hea_model.n_outputs_ = base_model.n_outputs_
    hea_model.out_activation_ = base_model.out_activation_

    hea_model.fit(X_hea_train, y_hea_train)
    return hea_model
