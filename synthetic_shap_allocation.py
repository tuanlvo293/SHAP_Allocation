import numpy as np
import shap
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import argparse
import logging
from prettytable import PrettyTable
from tqdm import tqdm
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SHAP_Experiments")


def generate_synthetic_dataset(n_samples, n_features, noise=0.1):
    logger.debug("Generating synthetic dataset.")
    X = np.random.rand(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + noise * np.random.randn(n_samples)
    return X, y


def DR(X, explained_variance):
    logger.debug("Performing dimensionality reduction.")
    pca_full = PCA()
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= explained_variance) + 1
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    loading_matrix = (pca.components_).T
    return Z, loading_matrix, n_components


def SHAP_Allocation(model, Z_test, dFdx, id_interest):
    d = len(dFdx[0])
    r = len(dFdx)
    explainer = shap.LinearExplainer(model, Z_test)
    shap_values = explainer(Z_test)
    z_SHAP = shap_values[id_interest].values

    alpha_min, alpha_max = pow(10, -5), pow(10, 5)

    def unexplained_portion(z_SHAP, idx, alpha, beta):
        return z_SHAP[idx] * (1 - np.sum([math.tanh(beta + alpha * dFdx[idx][j]) for j in range(d)]))

    def calculate_beta_bounds(dFdx):
        dFdx_positive_min = np.min([dFdx[i][j] for i in range(r) for j in range(d) if dFdx[i][j] > 0], initial=np.inf)
        dFdx_negative_max = np.max([dFdx[i][j] for i in range(r) for j in range(d) if dFdx[i][j] < 0], initial=-np.inf)

        beta_lower = -alpha_min * dFdx_positive_min
        beta_upper = -alpha_max * dFdx_negative_max

        return beta_lower, beta_upper

    def objective(params):
        alphas = params[:-1]
        beta = params[-1]
        total = sum(unexplained_portion(z_SHAP, i, alphas[i], beta) ** 2 for i in range(r))
        return total

    def find_optimal_params():
        beta_lower, beta_upper = calculate_beta_bounds(dFdx)
        initial_params = [0] * r + [0]
        bounds = [(alpha_min, alpha_max)] * r + [(beta_lower, beta_upper)]

        result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds)
        optimal_alphas = result.x[:-1]
        optimal_beta = result.x[-1]
        return optimal_alphas, optimal_beta

    optimal_alphas, optimal_beta = find_optimal_params()

    x_SHAP = []
    for j in range(d):
        temp = sum(z_SHAP[i] * math.tanh(optimal_beta + optimal_alphas[i] * dFdx[i][j]) for i in range(r))
        x_SHAP.append(temp)
    explained_portion = 1 - np.abs(np.abs(np.sum(z_SHAP)) - np.abs(np.sum(x_SHAP))) / np.abs(np.sum(z_SHAP))
    return explained_portion


def single_run(n_samples, n_features):
    X, y = generate_synthetic_dataset(n_samples, n_features, noise=0.1)
    Z, loading_matrix, r = DR(X, explained_variance=0.90)
    dFdx = [loading_matrix[:, i] for i in range(r)]
    Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)
    model = LinearRegression()
    model.fit(Z_train, y_train)
    res = [SHAP_Allocation(model, Z_test, dFdx, id_interest=id) for id in range(len(Z_test))]
    return np.mean(res)


def experiments(n_features, run_time):
    logger.info(f"Running experiments with {n_features} features for {run_time} runs.")
    RES = []
    for _ in tqdm(range(run_time), desc="Experiment"):
        n_samples = 100
        X, y = generate_synthetic_dataset(n_samples, n_features, noise=0.1)
        Z, loading_matrix, r = DR(X, explained_variance=0.90)
        dFdx = [loading_matrix[:, i] for i in range(r)]
        Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2)
        model = LinearRegression()
        model.fit(Z_train, y_train)
        res = [SHAP_Allocation(model, Z_test, dFdx, id_interest=id) for id in range(len(Z_test))]
        RES.append(np.mean(res))
    return RES


def experiments_parallel(n_features, run_time, n_workers):
    logger.info(f"Running experiments with {n_features} features for {run_time} runs using {n_workers} workers.")
    n_samples = 100

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(single_run, n_samples, n_features) for _ in range(run_time)]
        results = [future.result() for future in tqdm(as_completed(futures), total=run_time, desc="Parallel rolling")]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Allocation experiments.")
    parser.add_argument("--n_features", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help="Range of features to experiment with.")
    parser.add_argument("--run_time", type=int, default=10, help="Number of runs per feature set.")
    parser.add_argument("--to_csv", type=str, default="results.csv", help="Output file to save results as CSV.")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of workers to use for parallel experiments.")
    args = parser.parse_args()

    table = PrettyTable()
    table.field_names = ["Number of features", "Mean", "Std. Dev"]
    csv_table = []

    for n_features in args.n_features:
        logger.info(f"Starting experiments for {n_features} features.")
        if args.n_workers >= 2:
            result = experiments_parallel(n_features=n_features, run_time=args.run_time, n_workers=args.n_workers)
        else:
            result = experiments(n_features=n_features, run_time=args.run_time)
        mean_result = np.mean(result, axis=0)
        std_dev = np.std(result, axis=0)
        table.add_row([n_features, mean_result, std_dev])
        csv_table.append([n_features, mean_result, std_dev])

    print(table)

    with open(args.to_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["n_features", "mean", "std"])
        csv_writer.writerows(csv_table)

    logger.info(f"Results saved to {args.to_csv}.")
