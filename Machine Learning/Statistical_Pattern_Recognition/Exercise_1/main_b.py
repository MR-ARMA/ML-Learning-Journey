"""
Statistical Pattern Recognition
Computer Homework #1
Alireza Mahdizadeh 40416814


BME Classifier - Part B
Liquid dataset leave-one-out evaluation
"""

import os
import sys
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

# --------------------------- Utility functions ---------------------------

def load_dataset(file_path, expected_cols=None):
    """Load dataset; supports whitespace or comma separated files."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    if expected_cols is not None and df.shape[1] != expected_cols:
        raise ValueError(f"Unexpected column count in {file_path}: got {df.shape[1]}, expected {expected_cols}")
    labels = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    return labels, X


def ml_mean_cov(X):
    """Maximum likelihood estimate of mean and covariance from samples X (n x d)."""
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty class for ML estimation")
    mu = X.mean(axis=0)
    Xm = X - mu
    Sigma = (Xm.T @ Xm) / n
    # regularization to avoid singularity
    eps = 1e-6 * np.eye(Sigma.shape[0])
    return mu, Sigma + eps


def discriminant_mvnormal(x, mu, Sigma, prior=1.0):
    """Discriminant function for multivariate normal with optional prior."""
    try:
        L = np.linalg.cholesky(Sigma)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(Sigma) + 1e-12))
    diff = x - mu
    quad = diff.T @ inv @ diff
    g = -0.5 * logdet - 0.5 * quad + np.log(prior)
    return float(g)


def predict_bme(x, params, priors=None):
    """Predict class for x using given parameters."""
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


# --------------------------- Liquid leave-one-out ---------------------------

def liquid_leave_one_out(data_path):
    labels, X = load_dataset(data_path)
    n = X.shape[0]
    classes = np.unique(labels)
    K = len(classes)

    confusion = np.zeros((K, K), dtype=int)
    misclassified_records = []
    first_params = None

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        labels_train, X_train = labels[train_mask], X[train_mask]

        # estimate parameters for each class
        params = {}
        for k in classes:
            Xk = X_train[labels_train == k]
            mu_k, Sigma_k = ml_mean_cov(Xk)
            params[k] = (mu_k, Sigma_k)

        # store first classifier parameters
        if i == 0:
            first_params = params.copy()

        # classify the left-out sample
        x_test = X[i]
        true_label = int(labels[i])
        pred_label, g_vals = predict_bme(x_test, params)
        confusion[true_label - 1, pred_label - 1] += 1

        if pred_label != true_label:
            indices_of_class = np.where(labels == true_label)[0]
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
        class_errors[int(k)] = 1.0 - (correct / total) if total > 0 else 0.0

    results = {
        'confusion': confusion,
        'misclassified': misclassified_records,
        'class_errors': class_errors,
        'first_params': first_params
    }
    return results


# --------------------------- Main runner ---------------------------

def main(data_dir='Data'):
    liquid_file = os.path.join(data_dir, 'Liquid-Data_dat.txt')

    print('\n=== Liquid: Leave-one-out evaluation ===')
    liquid_results = liquid_leave_one_out(liquid_file)

    print('\n-- First classifier parameters (when first sample is left out) --')
    for k, (mu, Sigma) in liquid_results['first_params'].items():
        print(f'Class {k}:')
        print(' mean =', np.round(mu, 6))
        print(' cov  =\n', np.round(Sigma, 6))

    print('\n-- Misclassified samples (leave-one-out) --')
    if len(liquid_results['misclassified']) == 0:
        print('No misclassifications')
    else:
        for rec in liquid_results['misclassified']:
            print(f"Global sample #{rec['sample_global_index']}: sample no.{rec['sample_no_within_class']} from class {rec['actual_class']} incorrectly classified to class {rec['predicted_class']} with g_actual={rec['g_actual']:.6f}, g_pred={rec['g_predicted']:.6f}")

    print('\n-- Confusion matrix (rows: actual class, cols: predicted class) --')
    print(liquid_results['confusion'])
    print('\n-- Class error rates --')
    for k, err in liquid_results['class_errors'].items():
        print(f'Class {k} error rate: {err*100:.2f}%')

    # Save results
    out_dir = os.path.join('results', 'part_b')
    os.makedirs(out_dir, exist_ok=True)

    if len(liquid_results['misclassified']) > 0:
        pd.DataFrame(liquid_results['misclassified']).to_csv(
            os.path.join(out_dir, 'liquid_misclassified_loocv.csv'), index=False
        )

    pd.DataFrame(liquid_results['confusion']).to_csv(
        os.path.join(out_dir, 'liquid_confusion_loocv.csv'), index=False
    )

    print(f"\nResults for Liquid dataset saved to {out_dir}/")

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'Data'
    main(data_dir)
