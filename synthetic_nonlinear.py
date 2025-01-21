import numpy as np
import shap
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import umap
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from prettytable import PrettyTable
import argparse
import csv
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("nonlinear_shap")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def appro_slp(X, Z):
    device_ = device()
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device_)
    Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device_)
    slp = nn.Linear(X_tensor.shape[1], Z_tensor.shape[1], bias=True).to(device_)
    optimizer = torch.optim.AdamW(slp.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(1000):
        optimizer.zero_grad()
        Z_pred = slp(X_tensor)
        loss = loss_fn(Z_pred, Z_tensor)
        if loss.item() < 1e-4:
            break
        loss.backward()
        optimizer.step()
    return slp.weight.detach().cpu().numpy()


def generate_data(n_samples, n_features, noise=0.1):
    X = np.random.rand(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + noise * np.random.randn(n_samples)
    return X, y


def dr_umap(X, n_components):
    umap_model = umap.UMAP(n_components=n_components)
    Z = umap_model.fit_transform(X)
    return Z


def shap_alloc(model, Z_test, dFdx, id_interest):
    d = len(dFdx[0])
    r = len(dFdx)
    explainer = shap.LinearExplainer(model, Z_test)
    shap_values = explainer(Z_test)
    z_shap = shap_values[id_interest].values

    alpha_min, alpha_max = pow(10, -5), pow(10, 5)

    def unexplained_portion(z_shap, idx, alpha, beta):
        return z_shap[idx] * (1 - np.sum([math.tanh(beta + alpha * dFdx[idx][j]) for j in range(d)]))

    def calculate_beta_bounds(dFdx):
        dFdx_positive_min = np.min([dFdx[i][j] for i in range(r) for j in range(d) if dFdx[i][j] > 0], initial=np.inf)
        dFdx_negative_max = np.max([dFdx[i][j] for i in range(r) for j in range(d) if dFdx[i][j] < 0], initial=-np.inf)
        return -alpha_min * dFdx_positive_min, -alpha_max * dFdx_negative_max

    def objective(params):
        alphas, beta = params[:-1], params[-1]
        return sum(unexplained_portion(z_shap, i, alphas[i], beta) ** 2 for i in range(r))

    def find_optimal_params():
        beta_lower, beta_upper = calculate_beta_bounds(dFdx)
        initial_params = [0] * r + [0]
        bounds = [(alpha_min, alpha_max)] * r + [(beta_lower, beta_upper)]
        result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds)
        return result.x[:-1], result.x[-1]

    optimal_alphas, optimal_beta = find_optimal_params()

    x_shap = [
        sum(z_shap[i] * math.tanh(optimal_beta + optimal_alphas[i] * dFdx[i][j]) for i in range(r))
        for j in range(d)
    ]

    explained_portion = 1 - np.abs(np.abs(np.sum(z_shap)) - np.abs(np.sum(x_shap))) / np.abs(np.sum(z_shap))
    return explained_portion


def experiments(n_samples, n_features, run_time):
    results = []
    for _ in tqdm(range(run_time), desc="Experiments"):
        X, y = generate_data(n_samples, n_features, noise=0.1)
        Z = dr_umap(X, n_components=n_features // 3)
        dFdx = appro_slp(X, Z)
        Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)
        model = LinearRegression()
        model.fit(Z_train, y_train)

        explained_percentages = [
            shap_alloc(model, Z_test, dFdx, id_interest=i) for i in range(len(Z_test))
        ]
        results.append(np.mean(explained_percentages))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Allocation experiments.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples for data generation.")
    parser.add_argument("--n_features", type=int, nargs="+", default=list(range(10, 110, 10)),
                        help="Number of features to experiment with.")
    parser.add_argument("--run_time", type=int, default=100, help="Number of runs per feature set.")
    parser.add_argument("--to_csv", type=str, default="results.csv", help="Output CSV file for results.")
    args = parser.parse_args()

    logger.info(f"Running experiments on {device()}.")
    table = PrettyTable()
    table.field_names = ["Features", "Mean", "Std. Dev"]
    results_csv = []

    for n_features in args.n_features:
        logger.info(f"Starting experiments for {n_features} features.")
        result = experiments(args.n_samples, n_features, args.run_time)
        mean_result = np.mean(result)
        std_result = np.std(result)
        table.add_row([n_features, mean_result, std_result])
        results_csv.append([n_features, mean_result, std_result])
        print(table)

    with open(args.to_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Features", "Mean", "Std. Dev"])
        csv_writer.writerows(results_csv)

    logger.info(f"Results saved to {args.to_csv}.")
