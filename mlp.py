import pandas as pd
import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor
from skorch.callbacks import PrintLog
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Load already preprocessed datasets
df_train = pd.read_csv('train_preprocesado.csv')
df_val = pd.read_csv('val_preprocesado.csv')
df_test = pd.read_csv('test_preprocesado.csv')

# 2. Separate features and target
y_train = df_train['prezo_euros'].values.astype(np.float32)
y_val = df_val['prezo_euros'].values.astype(np.float32)
X_train = df_train.drop(columns=['prezo_euros', 'id']).values.astype(np.float32)
X_val = df_val.drop(columns=['prezo_euros', 'id']).values.astype(np.float32)
X_test = df_test.drop(columns=['id']).values.astype(np.float32)

# 3. Define PyTorch module for MLP
def create_module(input_dim=10, hidden_units=[100, 50], activation=nn.ReLU):
    layers = []
    in_dim = input_dim
    for h in hidden_units:
        layers.append(nn.Linear(in_dim, h))
        layers.append(activation())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Flatten(start_dim=0))

    return nn.Sequential(*layers)

# 4. Wrap with skorch NeuralNetRegressor (verbose=1 already prints train/valid loss per epoch)
input_dim = X_train.shape[1]
net = NeuralNetRegressor(
    module=create_module,
    module__input_dim=input_dim,
    max_epochs=50,
    lr=0.001,
    batch_size=32,
    optimizer=torch.optim.Adam,
    criterion=nn.MSELoss,
    device=device,
    verbose=1,  # shows train and valid loss each epoch
)

# 5. Hyperparameter distribution for RandomizedSearchCV Hyperparameter distribution for RandomizedSearchCV
dist_params = {
    'module__hidden_units': [[50], [100], [100, 50], [100, 100, 50]],
    'module__activation': [nn.ReLU, nn.Tanh],
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'optimizer__weight_decay': [0, 1e-4, 1e-3]
}

search = RandomizedSearchCV(
    net,
    param_distributions=dist_params,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    verbose=2,
    refit=True,
)

# 6. Run hyperparameter search on training set
search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# 7. Evaluate on validation set
y_val_pred = search.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.2f}")

# 8. Retrain on train+validation
a = np.concatenate([X_train, X_val], axis=0)
b = np.concatenate([y_train, y_val], axis=0)
best_net = search.best_estimator_
best_net.set_params(max_epochs=50)
best_net.fit(a, b)

# 9. Predict on test and save predictions
test_preds = best_net.predict(X_test)
out = df_test[['id']].copy()
out['prezo_euros'] = test_preds
out.to_csv('test_predictions.csv', index=False)

# 10. Export the trained model
joblib.dump(best_net, 'mlp_price_model.pkl')
print("Model and predictions saved.")
