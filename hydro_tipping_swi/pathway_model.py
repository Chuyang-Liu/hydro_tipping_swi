from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from .pathway_labels import build_labels_3class_gate, build_labels_4class, CLASS_ORDER_3


def load_training_df(
    csv_path: str | Path,
    recharge_scale_m_per_yr: float,
    permeability_col: str = 'Permeability [m^2]',
    recharge_col: str = 'GWR_precp_rate',
    slr_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if recharge_col in df.columns and 'recharge_eff_m_per_yr' not in df.columns:
        df['recharge_eff_m_per_yr'] = pd.to_numeric(df[recharge_col], errors='coerce') * recharge_scale_m_per_yr
    if permeability_col in df.columns and 'log_permeability_m2' not in df.columns:
        K = pd.to_numeric(df[permeability_col], errors='coerce')
        df['log_permeability_m2'] = np.where(K > 0, np.log10(K), np.nan)
    if slr_col is not None and slr_col in df.columns and 'slr_m' not in df.columns:
        df['slr_m'] = pd.to_numeric(df[slr_col], errors='coerce')
    return df


def split_train_val_test_stratified(
    X: pd.DataFrame, y: pd.Series, train_frac: float, val_frac: float, test_frac: float, random_seed: int = 42
):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError('Fractions must sum to 1.')
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(1 - train_frac), stratify=y, random_state=random_seed)
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=random_seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_and_train_gate2stage(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_iter_search: int = 35,
    n_jobs_tune: int = 1,
    random_seed: int = 42,
    model_out: Optional[str | Path] = None,
):
    y3 = build_labels_3class_gate(df)
    keep = y3.notna()
    X = df.loc[keep, feature_cols].copy()
    y = y3.loc[keep].copy()

    valid = np.isfinite(X).all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_stratified(X, y, random_seed=random_seed)

    base = RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs_tune, class_weight='balanced_subsample')
    param_dist = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [5, 8, 12, None],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
    }
    search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter_search, cv=5, scoring='f1_macro', random_state=random_seed, n_jobs=n_jobs_tune, verbose=0)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_val_pred = best.predict(X_val)
    y_test_pred = best.predict(X_test)

    bundle = {
        'model': best,
        'feature_cols': feature_cols,
        'class_order_3': CLASS_ORDER_3,
        'best_params': search.best_params_,
        'val_report': classification_report(y_val, y_val_pred, output_dict=True),
        'test_report': classification_report(y_test, y_test_pred, output_dict=True),
        'val_confusion': confusion_matrix(y_val, y_val_pred, labels=CLASS_ORDER_3),
        'test_confusion': confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER_3),
    }
    if model_out is not None:
        model_out = Path(model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_out)
    return bundle


def load_model_bundle(path: str | Path):
    return joblib.load(path)
