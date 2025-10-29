"""
Statistical Pattern Recognition
Computer Homework #1
Alireza Mahdizadeh 40416814


BME Classifier - Part C
Normal dataset: train on Training file, test on Testing file
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.set_printoptions(precision=4, suppress=True)

# --------------------------- Utility functions ---------------------------

def load_dataset(file_path, expected_cols=None):
    """Load dataset; supports whitespace-separated files."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    if expected_cols is not None and df.shape[1] != expected_cols:
        raise ValueError(f"Unexpected column count in {file_path}: got {df.shape[1]}, expected {expected_cols}")
    labels = df.iloc[:,0].astype(int).to_numpy()
    X = df.iloc[:,1:].to_numpy(dtype=float)
    return labels, X


def ml_mean_cov(X):
    """Maximum likelihood estimate of mean and covariance from samples X."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xm = X - mu
    Sigma = (Xm.T @ Xm) / n
    # regularization
    eps = 1e-6 * np.eye(Sigma.shape[0])
    return mu, Sigma + eps


def discriminant_mvnormal(x, mu, Sigma, prior=1.0):
    """Discriminant function for multivariate normal."""
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
    g = -0.5*logdet - 0.5*quad + np.log(prior)
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


# --------------------------- BME classifier on Normal dataset ---------------------------

def normal_train_test(train_file, test_file):
    labels_train, X_train = load_dataset(train_file)
    labels_test, X_test = load_dataset(test_file)
    classes = np.unique(labels_train)

    # ML estimates for each class
    params = {}
    priors = {}
    n_train = labels_train.shape[0]
    for k in classes:
        Xk = X_train[labels_train == k]
        mu_k, Sigma_k = ml_mean_cov(Xk)
        params[k] = (mu_k, Sigma_k)
        priors[k] = Xk.shape[0]/n_train

    # classify test samples
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    misclassified = []
    pred_labels = []
    for j, x in enumerate(X_test):
        true_label = int(labels_test[j])
        pred_label, g_vals = predict_bme(x, params, priors)
        pred_labels.append(pred_label)
        confusion[true_label-1, pred_label-1] += 1
        if pred_label != true_label:
            record = {
                'sample_index': int(j+1),
                'actual_class': true_label,
                'predicted_class': pred_label,
                'g_actual': g_vals[true_label],
                'g_predicted': g_vals[pred_label]
            }
            misclassified.append(record)

    # empirical error rates
    class_errors = {}
    for idx, k in enumerate(classes):
        total = confusion[idx,:].sum()
        correct = confusion[idx, idx]
        class_errors[int(k)] = 1.0 - correct/total if total > 0 else 0.0
    overall_error = 1.0 - np.trace(confusion)/np.sum(confusion)

    # theoretical error rate
    # known parameters: μ1=[0,0], Σ1=[[2,0],[0,2]], μ2=[4,0], Σ2=[[2,0],[0,2]], equal priors
    mu1 = np.array([0,0]); Sigma1 = np.array([[2,0],[0,2]])
    mu2 = np.array([4,0]); Sigma2 = np.array([[2,0],[0,2]])
    # Bayes decision boundary: g1(x) = g2(x)
    # Use Monte Carlo integration to estimate theoretical error
    n_mc = 100000
    # class 1 error (P(class1 -> class2))
    Xmc1 = np.random.multivariate_normal(mu1, Sigma1, n_mc)
    g1_vals = np.array([discriminant_mvnormal(x, mu1, Sigma1) for x in Xmc1])
    g2_vals = np.array([discriminant_mvnormal(x, mu2, Sigma2) for x in Xmc1])
    err1 = np.mean(g2_vals > g1_vals)
    # class 2 error
    Xmc2 = np.random.multivariate_normal(mu2, Sigma2, n_mc)
    g1_vals2 = np.array([discriminant_mvnormal(x, mu1, Sigma1) for x in Xmc2])
    g2_vals2 = np.array([discriminant_mvnormal(x, mu2, Sigma2) for x in Xmc2])
    err2 = np.mean(g1_vals2 > g2_vals2)
    theoretical_error = 0.5*(err1+err2)

    results = {
        'confusion': confusion,
        'class_errors': class_errors,
        'overall_error': overall_error,
        'misclassified': misclassified,
        'params': params,
        'priors': priors,
        'theoretical_error': theoretical_error,
        'X_train': X_train,
        'labels_train': labels_train,
        'X_test': X_test,
        'labels_test': labels_test,
        'pred_labels': np.array(pred_labels)
    }
    return results


# --------------------------- Plot function ---------------------------

def plot_normal_results(results, out_dir):
    X_train = results['X_train']; y_train = results['labels_train']
    X_test = results['X_test']; y_test = results['labels_test']
    pred_labels = results['pred_labels']

    plt.figure(figsize=(8,6))
    # Training samples
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], marker='o', c='blue', label='Train Class1')
    plt.scatter(X_train[y_train==2,0], X_train[y_train==2,1], marker='s', c='green', label='Train Class2')
    # Testing samples
    plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1], marker='^', c='cyan', label='Test Class1')
    plt.scatter(X_test[y_test==2,0], X_test[y_test==2,1], marker='v', c='lime', label='Test Class2')
    # Misclassified test samples
    mis = np.array([rec['sample_index']-1 for rec in results['misclassified']])
    if len(mis) > 0:
        plt.scatter(X_test[mis,0], X_test[mis,1], facecolors='none', edgecolors='red', s=100, label='Misclassified')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Normal Dataset: BME Classification')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'normal_bme_plot.png'))
    plt.show()


# --------------------------- Main runner ---------------------------

def main(data_dir='Data'):
    train_file = os.path.join(data_dir, 'Normal-Data-Training_dat.txt')
    test_file = os.path.join(data_dir, 'Normal-Data-Testing_dat.txt')

    out_dir = os.path.join('results', 'part_c')
    os.makedirs(out_dir, exist_ok=True)

    results = normal_train_test(train_file, test_file)

    print('\nConfusion matrix (rows: actual, cols: predicted):')
    print(results['confusion'])
    print('\nClass errors:')
    for k, err in results['class_errors'].items():
        print(f'Class {k} error rate: {err*100:.2f}%')
    print(f'\nOverall empirical error rate: {results["overall_error"]*100:.2f}%')
    print(f'Theoretical Bayes error rate: {results["theoretical_error"]*100:.2f}%')

    print('\nMisclassified test samples:')
    if len(results['misclassified']) == 0:
        print('None')
    else:
        for rec in results['misclassified']:
            print(f"Sample #{rec['sample_index']}: actual={rec['actual_class']}, predicted={rec['predicted_class']}, g_actual={rec['g_actual']:.6f}, g_predicted={rec['g_predicted']:.6f}")

    # save misclassified and confusion matrix
    if len(results['misclassified']) > 0:
        pd.DataFrame(results['misclassified']).to_csv(os.path.join(out_dir, 'normal_misclassified.csv'), index=False)
    pd.DataFrame(results['confusion']).to_csv(os.path.join(out_dir, 'normal_confusion.csv'), index=False)

    # plot
    plot_normal_results(results, out_dir)
    print(f'\nPlot and CSV files saved in {out_dir}/')


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'Data'
    main(data_dir)
