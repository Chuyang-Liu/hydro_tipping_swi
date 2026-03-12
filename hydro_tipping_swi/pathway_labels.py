from __future__ import annotations

import re
import numpy as np
import pandas as pd

K_TIP_DEFAULT = 1.58e-12
F_L_DEFAULT = 0.40
F_V_DEFAULT = 0.60

CLASS_ORDER_4 = ['SWI not accelerated', 'Lateral-dominated', 'Mixed', 'Vertical-dominated']
CLASS_TO_ID_4 = {c: i + 1 for i, c in enumerate(CLASS_ORDER_4)}
ID_TO_CLASS_4 = {v: k for k, v in CLASS_TO_ID_4.items()}
CLASS_ORDER_3 = ['Lateral-dominated', 'Mixed', 'Vertical-dominated']
CLASS_TO_ID_3 = {c: i for i, c in enumerate(CLASS_ORDER_3)}
ID_TO_CLASS_3 = {v: k for k, v in CLASS_TO_ID_3.items()}


def parse_slr_m_from_pf_df_label(s: str | None) -> float:
    if s is None:
        return 0.0
    s = str(s).strip()
    if s.lower() == 'historical':
        return 0.0
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*m', s)
    return float(m.group(1)) if m else 0.0


def classify_pathway_from_fvert(f_vert: np.ndarray | pd.Series, f_l: float = F_L_DEFAULT, f_v: float = F_V_DEFAULT):
    fv = np.asarray(f_vert, dtype='float64')
    out = np.empty(fv.shape, dtype=object)
    out[:] = np.nan
    valid = np.isfinite(fv)
    out[valid & (fv <= f_l)] = 'Lateral-dominated'
    out[valid & (fv > f_l) & (fv < f_v)] = 'Mixed'
    out[valid & (fv >= f_v)] = 'Vertical-dominated'
    return out


def build_labels_4class(
    df: pd.DataFrame,
    permeability_col: str = 'Permeability [m^2]',
    fvert_col: str = 'f_vert',
    k_tip: float = K_TIP_DEFAULT,
    f_l: float = F_L_DEFAULT,
    f_v: float = F_V_DEFAULT,
) -> pd.Series:
    if fvert_col not in df.columns:
        raise ValueError(f"Training df missing '{fvert_col}' column.")
    K = df[permeability_col].astype(float).values
    fv = df[fvert_col].astype(float).values
    out = np.empty(len(df), dtype=object); out[:] = np.nan
    valid = np.isfinite(K) & np.isfinite(fv)
    out[valid & (K < k_tip)] = 'SWI not accelerated'
    out[valid & (K >= k_tip)] = classify_pathway_from_fvert(fv[valid & (K >= k_tip)], f_l=f_l, f_v=f_v)
    return pd.Series(out, index=df.index, name='pathway_class_4')


def build_labels_3class_gate(
    df: pd.DataFrame,
    permeability_col: str = 'Permeability [m^2]',
    fvert_col: str = 'f_vert',
    k_tip: float = K_TIP_DEFAULT,
    f_l: float = F_L_DEFAULT,
    f_v: float = F_V_DEFAULT,
) -> pd.Series:
    K = df[permeability_col].astype(float).values
    fv = df[fvert_col].astype(float).values
    out = np.empty(len(df), dtype=object); out[:] = np.nan
    valid = np.isfinite(K) & np.isfinite(fv) & (K >= k_tip)
    out[valid] = classify_pathway_from_fvert(fv[valid], f_l=f_l, f_v=f_v)
    return pd.Series(out, index=df.index, name='pathway_class_3')
