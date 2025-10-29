"""
Statistical Pattern Recognition
Computer Homework #1
Alireza Mahdizadeh 40416814


BME Classifier Project
Exercise_2 implementation in Python

Structure expected:
Exercise_1/
   Data/
       Iris-Data_dat.txt
       Liquid-Data_dat.txt
       Normal-Data-Training_dat.txt
       Normal-Data-Testing_dat.txt
   main_a.py  <-- this file
   main_b.py
   main_c.py
   

How to run:
    python main.py

The script will:
 - Implement a Bayes Minimum Error (BME) classifier using ML estimates
 - Run leave-one-out cross validation on the Iris dataset and print:
     * Means and covariances for the first classifier (when sample 1 is left out)
     * A list of misclassified samples with required details
     * Confusion matrix and per-class error rates
 - Train and test a classifier for Liquid dataset (train on full data, report confusion matrix via 5-fold CV)
 - Train on Normal training file and evaluate on Normal testing file

Notes:
 - The code expects the Data/ files to exist according to the structure above.
 - Covariance matrices are regularized slightly to avoid singularity.

"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold

np.set_printoptions(precision=4, suppress=True)

# --------------------------- Utility functions ---------------------------

def load_dataset(file_path, expected_cols=None):
    """Load dataset; supports whitespace or comma separated files."""
    # Ensure absolute path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, header=None, sep=r'\s+') # updated to avoid FutureWarning
    if expected_cols is not None and df.shape[1] != expected_cols:
        raise ValueError(f"Unexpected column count in {file_path}: got {df.shape[1]}, expected {expected_cols}")
    labels = df.iloc[:,0].astype(int).to_numpy()
    X = df.iloc[:,1:].to_numpy(dtype=float)
    return labels, X


def ml_mean_cov(X):
    """Maximum likelihood estimate of mean and covariance from samples X (n x d).
    ML covariance uses (1/n) normalization (not unbiased 1/(n-1)).
    Regularize covariance slightly to avoid singular matrices.
    Returns mu (d,), Sigma (d,d).
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty class for ML estimation")
    mu = X.mean(axis=0)
    Xm = X - mu
    Sigma = (Xm.T @ Xm) / n
    # regularization
    eps = 1e-6 * np.eye(Sigma.shape[0])
    Sigma_reg = Sigma + eps
    return mu, Sigma_reg


def discriminant_mvnormal(x, mu, Sigma, prior=1.0):
    """Compute discriminant function g(x) = log p(x|class) + log prior (up to additive constant)
    For multivariate normal: log p = -0.5*log|2pi Sigma| -0.5 (x-mu)^T Sigma^{-1} (x-mu)
    We omit the -0.5*log(2pi) constant since same for all classes of same dim.
    """
    d = x.shape[0]
    # use quadratic term and log determinant
    try:
        L = np.linalg.cholesky(Sigma)
        # compute logdet from cholesky
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        # fallback
        inv = np.linalg.pinv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(Sigma) + 1e-12))
    diff = x - mu
    quad = diff.T @ inv @ diff
    log_prior = np.log(prior)
    g = -0.5 * logdet - 0.5 * quad + log_prior
    return float(g)


def predict_bme(x, params, priors=None):
    """Given x (d,), and params dict class->(mu,Sigma), priors dict optional, return predicted class and g values dict."""
    best_k = None
    best_g = -np.inf
    g_vals = {}
    for k, (mu, Sigma) in params.items():
        prior = 1.0 if priors is None else priors.get(k, 1.0)
        g = discriminant_mvnormal(x, mu, Sigma, prior)
        g_vals[k] = g
        if g > best_g:
            best_g = g
            best_k = k
    return best_k, g_vals


# --------------------------- Iris leave-one-out ---------------------------

def iris_leave_one_out(data_path):
    labels, X = load_dataset(data_path)
    n = X.shape[0]
    classes = np.unique(labels)
    K = len(classes)
    assert n == 150 and K == 3, "Iris dataset unexpected size"

    # Group indices by class assuming dataset arranged class1 (1..50), class2 (51..100), class3 (101..150)
    class_indices = {k: np.where(labels == k)[0] for k in classes}

    confusion = np.zeros((K, K), dtype=int)  # rows actual, cols predicted (1-indexed classes mapped to 0..K-1)
    misclassified_records = []

    # We'll also capture the first classifier's parameter estimates (when sample 0 is left out)
    first_params = None

    for i in range(n):
        # leave out sample i
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        labels_train = labels[train_mask]
        X_train = X[train_mask]

        # estimate params per class
        params = {}
        for k in classes:
            idx = np.where(labels_train == k)[0]
            Xk = X_train[idx]
            mu_k, Sigma_k = ml_mean_cov(Xk)
            params[k] = (mu_k, Sigma_k)

        # store first classifier estimates (i == 0 means left out first sample)
        if i == 0:
            first_params = params.copy()

        # classify the held-out sample
        x_test = X[i]
        true_label = int(labels[i])
        pred_label, g_vals = predict_bme(x_test, params)

        confusion[true_label - 1, pred_label - 1] += 1

        if pred_label != true_label:
            # sample number (out of 50) within its class: compute index within class ordering
            # find its position among samples of its class (1-based). Since original file arranged by class blocks,
            # but to be robust we compute within class order of appearance
            indices_of_class = np.where(labels == true_label)[0]
            # find which position i is in this list
            pos_in_class = int(np.where(indices_of_class == i)[0][0]) + 1
            record = {
                'sample_global_index': int(i+1),
                'sample_no_within_class': pos_in_class,
                'actual_class': int(true_label),
                'predicted_class': int(pred_label),
                'g_actual': g_vals[true_label],
                'g_predicted': g_vals[pred_label]
            }
            misclassified_records.append(record)

    # compute error rates per class
    class_errors = {}
    for idx, k in enumerate(classes):
        total = confusion[idx, :].sum()
        correct = confusion[idx, idx]
        err = 1.0 - (correct / total) if total > 0 else 0.0
        class_errors[int(k)] = err

    results = {
        'confusion': confusion,
        'misclassified': misclassified_records,
        'class_errors': class_errors,
        'first_params': first_params
    }
    return results


# --------------------------- Liquid dataset (example evaluation) ---------------------------

def liquid_evaluate(data_path, n_splits=5):
    labels, X = load_dataset(data_path)
    classes = np.unique(labels)
    K = len(classes)

    # Use KFold CV (stratification not implemented; simple CV for demonstration)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    confusion_sum = np.zeros((K,K), dtype=int)

    for train_idx, test_idx in kf.split(X):
        labels_train, X_train = labels[train_idx], X[train_idx]
        labels_test, X_test = labels[test_idx], X[test_idx]

        params = {}
        for k in classes:
            Xk = X_train[labels_train == k]
            mu_k, Sigma_k = ml_mean_cov(Xk)
            params[k] = (mu_k, Sigma_k)

        for j, x in enumerate(X_test):
            true = int(labels_test[j])
            pred, _ = predict_bme(x, params)
            confusion_sum[true-1, pred-1] += 1

    class_errors = {}
    for idx, k in enumerate(classes):
        total = confusion_sum[idx,:].sum()
        correct = confusion_sum[idx, idx]
        err = 1.0 - (correct / total) if total > 0 else 0.0
        class_errors[int(k)] = err

    return {
        'confusion': confusion_sum,
        'class_errors': class_errors
    }


# --------------------------- Normal dataset (train/test) ---------------------------

def normal_train_test(train_path, test_path):
    labels_train, X_train = load_dataset(train_path)
    labels_test, X_test = load_dataset(test_path)
    classes = np.unique(labels_train)

    params = {}
    priors = {}
    total_train = labels_train.shape[0]
    for k in classes:
        Xk = X_train[labels_train == k]
        mu_k, Sigma_k = ml_mean_cov(Xk)
        params[k] = (mu_k, Sigma_k)
        priors[k] = Xk.shape[0] / total_train

    K = len(classes)
    confusion = np.zeros((K, K), dtype=int)

    for j, x in enumerate(X_test):
        true = int(labels_test[j])
        pred, _ = predict_bme(x, params, priors=priors)
        confusion[true-1, pred-1] += 1

    class_errors = {}
    for idx, k in enumerate(classes):
        total = confusion[idx,:].sum()
        correct = confusion[idx, idx]
        err = 1.0 - (correct / total) if total > 0 else 0.0
        class_errors[int(k)] = err

    return {
        'confusion': confusion,
        'class_errors': class_errors,
        'params': params,
        'priors': priors
    }


def main(data_dir='Data'):
    iris_file = os.path.join(data_dir, 'Iris-Data_dat.txt')
    liquid_file = os.path.join(data_dir, 'Liquid-Data_dat.txt')
    normal_train = os.path.join(data_dir, 'Normal-Data-Training_dat.txt')
    normal_test = os.path.join(data_dir, 'Normal-Data-Testing_dat.txt')

    # Create Part A results folder
    out_dir = os.path.join('results', 'part_a')
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------- Iris LOOCV ---------------------------
    print('\n=== Iris: Leave-one-out evaluation ===')
    iris_results = iris_leave_one_out(iris_file)

    print('\n-- First classifier parameters (when first sample is left out) --')
    for k, (mu, Sigma) in iris_results['first_params'].items():
        print(f'Class {k}:')
        print(' mean =', np.round(mu, 6))
        print(' cov  =\n', np.round(Sigma, 6))

    print('\n-- Misclassified samples (leave-one-out) --')
    if len(iris_results['misclassified']) == 0:
        print('No misclassifications')
    else:
        for rec in iris_results['misclassified']:
            print(f"Global sample #{rec['sample_global_index']}: sample no.{rec['sample_no_within_class']} from class {rec['actual_class']} incorrectly classified to class {rec['predicted_class']} with g_actual={rec['g_actual']:.6f}, g_pred={rec['g_predicted']:.6f}")

    print('\n-- Confusion matrix (rows: actual class 1..3, cols: predicted 1..3) --')
    print(iris_results['confusion'])
    print('\n-- Class error rates --')
    for k, err in iris_results['class_errors'].items():
        print(f'Class {k} error rate: {err*100:.2f}%')

    # Save Iris results
    if len(iris_results['misclassified']) > 0:
        pd.DataFrame(iris_results['misclassified']).to_csv(
            os.path.join(out_dir, 'iris_misclassified_loocv.csv'), index=False
        )
    pd.DataFrame(iris_results['confusion']).to_csv(
        os.path.join(out_dir, 'iris_confusion.csv'), index=False
    )

    # --------------------------- Liquid evaluation ---------------------------
    print('\n=== Liquid dataset evaluation (5-fold CV) ===')
    liquid_results = liquid_evaluate(liquid_file, n_splits=5)

    print('\nConfusion matrix (Liquid):')
    print(liquid_results['confusion'])
    print('\nClass errors (Liquid):')
    for k, err in liquid_results['class_errors'].items():
        print(f'Class {k} error rate: {err*100:.2f}%')

    # Save Liquid confusion matrix
    pd.DataFrame(liquid_results['confusion']).to_csv(
        os.path.join(out_dir, 'liquid_confusion.csv'), index=False
    )

    # --------------------------- Normal evaluation ---------------------------
    print('\n=== Normal dataset: train on training file, evaluate on testing file ===')
    normal_results = normal_train_test(normal_train, normal_test)

    print('\nConfusion matrix (Normal):')
    print(normal_results['confusion'])
    print('\nClass errors (Normal):')
    for k, err in normal_results['class_errors'].items():
        print(f'Class {k} error rate: {err*100:.2f}%')

    # Save Normal confusion matrix
    pd.DataFrame(normal_results['confusion']).to_csv(
        os.path.join(out_dir, 'normal_confusion.csv'), index=False
    )

    print(f"\nAll Part A results saved to {out_dir}/")


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'Data'

    # Create Part A results folder
    os.makedirs(os.path.join('results', 'part_a'), exist_ok=True)

    # Optional: log output to file as before
    log_path = os.path.join('results', 'part_a', 'output_main_a.txt')

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(log_path, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
            main(data_dir)
        finally:
            sys.stdout = original_stdout

    print(f"\nFull output log saved to {log_path}")
