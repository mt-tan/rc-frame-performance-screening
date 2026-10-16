"""
tbdy_filters.py — Makale 3: mevcut bina adaylarının ön-analiz filtreleri.

Orijinal (modules_orijinal/tbdy_filters.py) üzerine farklar:
  * Güçlü kolon-zayıf kiriş FİLTRESİ KALDIRILDI (kullanıcı: mevcut bina).
    ΣMc/ΣMb oranı yalnız tanımlayıcı olarak hesaplanır (scwb_ratio) (K-14).
  * Kolon kesit alt sınırı b, h ≥ 0.25 m (kullanıcı) (K-10).
  * fc ∈ [15, 35] MPa (kullanıcı) (K-04).
  * Kiriş net açıklığı L_min − col_H (düzlem içi derinlik) (K-10).
  * Eksenel yük oranı ≤ 0.60 korunur (kullanıcı).
  * Proje boyuna donatı oranı sınırları korunur (kullanıcı, madde 12);
    gerçekleşen donatı κ ile ayrıca azaltılır (filtre dışında).
"""

import math
import config


def filter_min_dimensions(p):
    if p['col_B'] < config.COL_B_MIN - 1e-9:
        return False, f"col_B={p['col_B']:.3f} m < {config.COL_B_MIN} m"
    if p['col_H'] < config.COL_H_RANGE[0] - 1e-9:
        return False, f"col_H={p['col_H']:.3f} m < {config.COL_H_RANGE[0]} m"
    if p['beam_b'] < 0.25 - 1e-9:
        return False, f"beam_b={p['beam_b']:.3f} m < 0.25 m"
    L_gov = p.get('L_max', p['Lbay'])
    if p['beam_h'] < L_gov / 12 - 1e-9:
        return False, f"beam_h={p['beam_h']:.3f} m < L_max/12={L_gov/12:.3f} m"
    return True, ''


def filter_reinforcement_ratios(p):
    if not (0.010 - 1e-9 <= p['rho_col'] <= 0.040 + 1e-9):
        return False, f"proje ρ_col={p['rho_col']:.4f} [0.010, 0.040] dışında"
    if not (0.0033 - 1e-9 <= p['rho_beam_top'] <= 0.020):
        return False, f"ρ_beam_top={p['rho_beam_top']:.4f} sınır dışı"
    if not (0.0033 - 1e-9 <= p['rho_beam_bot'] <= 0.020):
        return False, f"ρ_beam_bot={p['rho_beam_bot']:.4f} sınır dışı"
    if p['rho_beam_bot'] < 0.5 * p['rho_beam_top'] - 1e-9:
        return False, "ρ_beam_bot < 0.5·ρ_beam_top"
    if not (0.02 <= p['cover'] <= 0.05):
        return False, f"cover={p['cover']*100:.1f} cm [2, 5] dışında"
    return True, ''


def estimate_column_axial_load(p):
    """Kaba eksenel yük: en alt orta kolon ≈ 1.3 × ortalama (orijinal ile aynı)."""
    w_combined = p['G'] + config.n_combo * p['Q']
    n_story = p['N_story']
    n_axes = p['N_bay'] + 1
    W_width = p.get('W_total', p['N_bay'] * p['Lbay'])
    W_total_kN = n_story * W_width * w_combined
    Nd_kN = (W_total_kN / n_axes) * 1.3
    Ac_m2 = p['col_B'] * p['col_H']
    Pn_kN = Ac_m2 * p['fck']
    n_ratio = Nd_kN / Pn_kN if Pn_kN > 0 else float('inf')
    return Nd_kN, Pn_kN, n_ratio


def filter_axial_load(p, limit=0.60):
    Nd, Pn, n_ratio = estimate_column_axial_load(p)
    if n_ratio > limit:
        return False, f"Nd/(Ac·fc) = {n_ratio:.3f} > {limit:.2f}"
    return True, ''


def filter_hinge_geometry(p):
    """HingeRadau: 2·Lp < serbest boy. Kolon Lp = 0.5·col_H (düzlem içi),
    kiriş Lp = 0.5·beam_h, kiriş serbest açıklığı = L_min − col_H."""
    Lp_col = 0.5 * p['col_H']
    if 2 * Lp_col >= p['h1']:
        return False, f"2·Lp_col={2*Lp_col:.3f} ≥ h1={p['h1']:.3f}"
    Lp_beam = 0.5 * p['beam_h']
    L_clear_beam = p.get('L_min', p['Lbay']) - p['col_H']
    if 2 * Lp_beam >= L_clear_beam:
        return False, f"2·Lp_beam={2*Lp_beam:.3f} ≥ L_clear={L_clear_beam:.3f}"
    return True, ''


def filter_long_feasible(p):
    if not p.get('col_long_feasible', True):
        return False, "kolon boyuna donatı: %1-4 bandı tutturulamadı"
    if not p.get('beam_long_feasible', True):
        return False, "kiriş boyuna donatı koşulu sağlanamadı"
    return True, ''


def filter_material(p):
    fc_MPa = p['fc'] / config.MPa
    lo, hi = config.FC_RANGE_MPA
    if not (lo - 1e-9 <= fc_MPa <= hi + 1e-9):
        return False, f"fc={fc_MPa:.1f} MPa [{lo}, {hi}] dışında"
    return True, ''


# ----------------------------------------------------------------------
# Tanımlayıcı (filtre DEĞİL): kapasite tabanlı ΣMc/ΣMb (gerçekleşen donatı)
# ----------------------------------------------------------------------

def _beam_capacities(p):
    b, h, cov = p['beam_b'], p['beam_h'], p['cover']
    fc, fy = p['fc'], p['fy']
    phi = p.get('beam_phi_long', 0.018)
    petr = p.get('beam_phi_etr', 0.010)
    k = p.get('kappa_beam', 1.0)
    Abar = math.pi * phi ** 2 / 4.0 * k
    As_top = p['beam_n_long_top'] * Abar
    As_bot = p['beam_n_long_bot'] * Abar
    d = h - cov - petr - phi / 2.0

    def Mn(As):
        a = As * fy / (0.85 * fc * b)
        return As * fy * (d - a / 2.0)
    return Mn(As_top), Mn(As_bot)


def _column_capacity(p, N_kN):
    b, h, cov = p['col_B'], p['col_H'], p['cover']
    fc, fy = p['fc'], p['fy']
    phi = p.get('col_phi_long', 0.020)
    petr = p.get('col_phi_etr', 0.010)
    k = p.get('kappa_col', 1.0)
    Abar = math.pi * phi ** 2 / 4.0 * k
    As_t = p['col_n_long_x'] * Abar
    dd = h - 2.0 * (cov + petr + phi / 2.0)
    a = max(N_kN, 0.0) / (0.85 * fc * b)
    a = min(a, 0.8 * h)
    Mn = As_t * fy * dd + N_kN * (h / 2.0 - a / 2.0)
    return max(Mn, 0.5 * As_t * fy * dd)


def scwb_ratio(p):
    """En kritik iç düğümde ΣMc/ΣMb (yaklaşık, tanımlayıcı)."""
    Mn_neg, Mn_pos = _beam_capacities(p)
    SMb = Mn_neg + Mn_pos
    if SMb <= 0:
        return float('nan')
    Nbase, _, _ = estimate_column_axial_load(p)
    Ns = int(p['N_story'])
    worst = float('inf')
    for L in range(1, max(Ns, 2)):
        N_below = Nbase * (Ns - L + 1) / Ns
        N_above = Nbase * (Ns - L) / Ns
        SMc = _column_capacity(p, N_below) + _column_capacity(p, N_above)
        worst = min(worst, SMc / SMb)
    return worst


ALL_FILTERS = [
    ('min_dimensions',       filter_min_dimensions),
    ('long_feasible',        filter_long_feasible),
    ('reinforcement_ratios', filter_reinforcement_ratios),
    ('material',             filter_material),
    ('hinge_geometry',       filter_hinge_geometry),
    ('axial_load_60pct',     filter_axial_load),
]


def filter_all(p):
    _, _, axial_ratio = estimate_column_axial_load(p)
    for name, fn in ALL_FILTERS:
        ok, reason = fn(p)
        if not ok:
            return False, name, reason, axial_ratio
    return True, 'accepted', '', axial_ratio


def filter_batch(samples):
    accepted, rejected = [], []
    for s in samples:
        ok, fname, reason, axial = filter_all(s)
        if ok:
            accepted.append(s)
        else:
            rejected.append({'name': s['name'], 'filter': fname, 'reason': reason,
                             'axial_ratio': round(axial, 3)})
    return accepted, rejected


def reject_summary(rejected):
    from collections import Counter
    c = Counter(r['filter'] for r in rejected)
    return [{'filter': k, 'count': v} for k, v in c.most_common()]
