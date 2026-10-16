"""
target_displacement.py — Makale 3: TBEC 2018 Madde 5.6 hedef deplasman (tek modlu itme).

Hesap zinciri orijinal ile AYNI (modules_orijinal/target_displacement.py:
çatı-normalize Γ1, a* = V/M*, çıkan dalda eşit alan ikilineerleştirme,
Ke = 0.6·a*peak sekantı, CR Denk. 5.16). Farklar (K-02):
  * Spektrum artık saha/zemin/düzeye göre dışarıdan verilir (config.site_spectrum).
  * İkilineerleştirme bir kez yapılır; her deprem düzeyi için δt ayrı hesaplanır
    (yapı kapasite eğrisi düzeyden bağımsızdır).
  * I = 1.0 (TBEC 15.4.2).
"""

import numpy as np
import openseespy.opensees as ops
import config


def compute_modal_parameters(model_info, mode=1):
    ops.eigen(mode)
    roof = model_info['node_ids'][(model_info['N_story'], 0)]
    phi_raw_roof = ops.nodeEigenvector(roof, mode, 1)
    if phi_raw_roof == 0:
        return None
    scale = 1.0 / phi_raw_roof
    L, M, Mt = 0.0, 0.0, 0.0
    for (s, a), nid in model_info['node_ids'].items():
        if s == 0:
            continue
        m = ops.nodeMass(nid, 1)
        ph = ops.nodeEigenvector(nid, mode, 1) * scale
        L += m * ph
        M += m * ph ** 2
        Mt += m
    G1 = L / M if M > 0 else float('nan')
    Ms = L ** 2 / M if M > 0 else float('nan')
    return {'L1': L, 'M1': M, 'Gamma1': G1, 'M_star': Ms, 'M_total': Mt,
            'mass_participation': Ms / Mt if Mt > 0 else float('nan'), 'phi_roof': 1.0}


def tbdy_design_spectrum(T, SDS, SD1, TA, TB, TL=6.0):
    if T <= TA:
        return (0.4 + 0.6 * T / TA) * SDS
    if T <= TB:
        return SDS
    if T <= TL:
        return SD1 / T
    return SD1 * TL / T ** 2


def bilinearize_modal_curve(delta_star, a_star_g, Ke_ratio=0.60):
    """Orijinal ile aynı (çıkan dal, eşit alan, Ke = 0.6·a_peak sekantı)."""
    delta_star = np.asarray(delta_star, float)
    a_star_g = np.asarray(a_star_g, float)
    i_peak = int(np.argmax(a_star_g))
    a_peak = float(a_star_g[i_peak])
    if i_peak < 1 or a_peak <= 0:
        return None
    asc_d, asc_a = delta_star[:i_peak + 1], a_star_g[:i_peak + 1]
    d_peak = float(asc_d[-1])
    if d_peak <= 0:
        return None
    i06 = int(np.argmin(np.abs(asc_a - Ke_ratio * a_peak)))
    if asc_d[i06] <= 0:
        pos = np.where(asc_d > 0)[0]
        if len(pos) == 0:
            return None
        i06 = int(pos[0])
    Ke = float(asc_a[i06] / asc_d[i06])
    if Ke <= 0:
        return None
    A_real = float(np.trapezoid(asc_a, asc_d))
    b, c = -2.0 * Ke * d_peak, 2.0 * Ke * A_real
    disc = b * b - 4.0 * c
    ay = None
    if disc >= 0:
        sd = np.sqrt(disc)
        for cand in ((-b - sd) / 2.0, (-b + sd) / 2.0):
            if 0 < cand <= a_peak * 1.001:
                ay = min(cand, a_peak)
                break
    if ay is None:
        ay = a_peak
    dy = ay / Ke
    if dy <= 0:
        return None
    Te = 2.0 * np.pi * np.sqrt(dy / (ay * config.g))
    return {'delta_y_star': dy, 'a_y_star': ay, 'Ke_star': Ke, 'Te': Te, 'a_peak_star': a_peak}


def capacity_bilinear(pushover_result, modal):
    """Pushover → modal koordinat → ikilineer (düzeyden bağımsız)."""
    if modal is None or not np.isfinite(modal.get('Gamma1', np.nan)) \
            or abs(modal.get('Gamma1', 0.0)) < 1e-6:
        return None
    d_star = np.asarray(pushover_result['disp']) / modal['Gamma1']
    a_star = np.asarray(pushover_result['V_base']) / modal['M_star'] / config.g
    return bilinearize_modal_curve(d_star, a_star)


def target_for_spectrum(bi, modal, spec):
    """TBEC 5.6: δt = Γ1·CR·Sde(Te) (çatı-normalize)."""
    Te, ay = bi['Te'], bi['a_y_star']
    Sae = tbdy_design_spectrum(Te, spec['SDS'], spec['SD1'], spec['TA'], spec['TB'], spec['TL'])
    Sde = Sae * config.g * (Te / (2.0 * np.pi)) ** 2
    Ry = Sae / ay if ay > 0 else float('nan')
    CR = 1.0 if Te >= spec['TB'] else max(1.0, (1.0 + (Ry - 1.0) * spec['TB'] / Te) / Ry)
    Sd_inel = CR * Sde
    return {'Sae_Te_g': Sae, 'Sde_Te': Sde, 'Ry': Ry, 'CR': CR, 'Sd_inel': Sd_inel,
            'delta_t': modal['Gamma1'] * Sd_inel,
            'SDS': spec['SDS'], 'SD1': spec['SD1'], 'TA': spec['TA'], 'TB': spec['TB']}
