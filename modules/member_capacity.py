"""
member_capacity.py — Makale 3 (YENİ): eleman kapasiteleri ve kesme kontrolleri.

Mevcut malzeme dayanımlarıyla (TBEC 15.2.12-b: malzeme katsayısı yok,
bilgi düzeyi katsayısı 1.0):
  * Mr: taşıma gücü momenti (pekleşmesiz), dikdörtgen gerilme bloğu,
    εcu = 0.0035, donatı elastik-tam plastik (TBEC 15.4.9-b; k1 TS 500).
  * Ve: kapasite kesmesi — kolon (Mra + Mrü)/ln (TBEC 7.3.7.3; 15.5.2.2-a
    gereği Mp yerine Mr); kiriş Vdy + (Mri + Mrj)/ln (7.4.5), Vdy = w·ln/2,
    w = G + 0.3Q (K-12).
  * Vr: TS 500 — Vcr = 0.65·fctm·bw·d·(1 + γ·N/Ac), Vc = 0.8·Vcr,
    Vw = (Asw/s)·fyw·d, Vr = Vc + Vw ≤ 0.22·fc·Aw; fctm = 0.35·√fc.
    Kesme kolları kanca tipinden etkilenmez (15.7.1.3 yalnız ρsh içindir).
  * Ön-analiz gevreklik (15.5.2.2): Ve > Vr veya Ve > 0.85·bw·d·√fc.
  * 15.7.1.4 sınır azaltma katsayısı: rv = Ve/(bw·d·fctm);
    rv ≤ 0.65 → 1.0, rv ≥ 1.30 → 0.5, arada doğrusal.
  * Analiz sonrası (kuvvet kontrollü) gevrek kırılma: V_talep(δt) > Vr(N(δt)).
Birimler: m, kN, kPa.
"""

import math
import numpy as np
import config

ES_KPA = 2.0e8
EPS_CU = 0.0035


def _k1(fc_MPa):
    return float(np.clip(0.85 - 0.006 * (fc_MPa - 25.0), 0.70, 0.85))


def moment_capacity(b, h, layers, fc_kPa, fy_kPa, N_kN=0.0):
    """Dikdörtgen kesit Mr (kN·m), basınç yüzü y = +h/2.
    layers: [(y_i, A_i)], y kesit merkezinden (+ basınç yüzüne doğru).
    N_kN: eksenel kuvvet (basınç +)."""
    fc_MPa = fc_kPa / 1000.0
    k1 = _k1(fc_MPa)
    ys = np.array([y for y, _ in layers])
    As = np.array([A for _, A in layers])
    d_i = h / 2.0 - ys                         # basınç yüzünden derinlik

    def forces(c):
        a = min(k1 * c, h)
        Cc = 0.85 * fc_kPa * b * a
        eps = EPS_CU * (c - d_i) / c           # basınç +
        sig = np.clip(ES_KPA * eps, -fy_kPa, fy_kPa)
        Fs = As * sig
        return Cc, a, Fs

    def resid(c):
        Cc, _, Fs = forces(c)
        return Cc + Fs.sum() - N_kN

    lo, hi = 1e-5, 10.0 * h
    if resid(lo) > 0:
        c = lo
    elif resid(hi) < 0:
        c = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0:
                hi = mid
            else:
                lo = mid
        c = 0.5 * (lo + hi)
    Cc, a, Fs = forces(c)
    M = Cc * (h / 2.0 - a / 2.0) + float(np.sum(Fs * (h / 2.0 - d_i)))
    return max(M, 0.0)


def column_layers(p):
    """Kolon donatı katmanları (y, A): B yüzleri ±yb'de n_x, H yüzlerinde ara çubuklar."""
    H, cover = p['col_H'], p['cover']
    yb = H / 2.0 - cover - p['col_phi_etr'] - p['col_phi_long'] / 2.0
    Ab = math.pi * p['col_phi_long'] ** 2 / 4.0 * p.get('kappa_col', 1.0)
    nx, ny = int(p['col_n_long_x']), int(p['col_n_long_y'])
    layers = [(yb, nx * Ab), (-yb, nx * Ab)]
    if ny > 2:
        for y in np.linspace(yb, -yb, ny)[1:-1]:
            layers.append((float(y), 2 * Ab))
    return layers


def beam_layers(p):
    h, cover = p['beam_h'], p['cover']
    yb = h / 2.0 - cover - p['beam_phi_etr'] - p['beam_phi_long'] / 2.0
    Ab = math.pi * p['beam_phi_long'] ** 2 / 4.0 * p.get('kappa_beam', 1.0)
    return [(yb, p['beam_n_long_top'] * Ab), (-yb, p['beam_n_long_bot'] * Ab)]


def fctm_kPa(fc_kPa):
    return config.FCTM_COEF * math.sqrt(fc_kPa / 1000.0) * 1000.0


def shear_strength(bw, h, d, fc_kPa, fyw_kPa, Asw, s, N_kN=0.0):
    """TS 500 Vr (kN), mevcut dayanımlar. N basınç +."""
    # TS 500: Vcr = 0.65·fctd·bw·d·(1 + γ·Nd/Ac), γ = 0.07 (basınç), −0.3 (çekme);
    # Nd/Ac MPa cinsinden (kPa/1000).
    Ac = bw * h
    sig_MPa = N_kN / Ac / 1000.0
    gam = 0.07 if N_kN >= 0 else 0.3
    Vcr = 0.65 * fctm_kPa(fc_kPa) * bw * d * (1.0 + gam * sig_MPa)
    Vc = max(0.0, 0.8 * Vcr)
    Vw = Asw / s * fyw_kPa * d
    return min(Vc + Vw, 0.22 * fc_kPa * bw * d)


def shear_factor(rv):
    lo, hi = config.SHEAR_RATIO_LIMITS
    if rv <= lo:
        return 1.0
    if rv >= hi:
        return config.SHEAR_RATIO_MIN_FACTOR
    return 1.0 - (1.0 - config.SHEAR_RATIO_MIN_FACTOR) * (rv - lo) / (hi - lo)


def member_static_capacities(p, model_info, gravity_axial):
    """
    gravity_axial: {eid: N_kN (basınç +)} gravite analizi sonrası.
    Returns {eid: dict(Mr..., Ve, Vr0, rv, fv, brittle_pre, d, bw, Asw, s)}.
    """
    fc, fy = p['fc'], p['fy']
    out = {}
    # Kolon geometrisi
    Bc, Hc = p['col_B'], p['col_H']
    d_c = Hc - p['cover'] - p['col_phi_etr'] - p['col_phi_long'] / 2.0
    Asw_c = p['col_n_etr_y'] * math.pi * p['col_phi_etr'] ** 2 / 4.0   # düzlem içi kesmeye paralel kollar
    s_c = p['col_s_etr']
    lay_c = column_layers(p)
    # Kiriş geometrisi
    bb, hb = p['beam_b'], p['beam_h']
    d_b = hb - p['cover'] - p['beam_phi_etr'] - p['beam_phi_long'] / 2.0
    Asw_b = p['beam_n_etr_y'] * math.pi * p['beam_phi_etr'] ** 2 / 4.0
    s_b = p['beam_s_etr']
    lay_b = beam_layers(p)
    # Pozitif moment (alt lif çekme): basınç üstte → katmanlar olduğu gibi.
    # Negatif moment (üst lif çekme): katmanlar ters çevrilir.
    Mr_pos_b = moment_capacity(bb, hb, lay_b, fc, fy, 0.0)
    Mr_neg_b = moment_capacity(bb, hb, [(-y, A) for y, A in lay_b], fc, fy, 0.0)
    w = p['G'] + config.n_combo * p['Q']
    fct = fctm_kPa(fc)
    lim_shear_sec = lambda bw, d: 0.85 * bw * d * math.sqrt(fc / 1000.0) * 1000.0

    for eid, meta in model_info['elem_meta'].items():
        if meta['type'] == 'col':
            N = gravity_axial.get(eid, 0.0)
            Mr = moment_capacity(Bc, Hc, lay_c, fc, fy, N)
            ln = max(meta['L'] - hb, 0.3 * meta['L'])
            Ve = 2.0 * Mr / ln
            Vr0 = shear_strength(Bc, Hc, d_c, fc, fy, Asw_c, s_c, N)
            rv = Ve / (Bc * d_c * fct)
            brittle = bool(Ve > Vr0 or Ve > lim_shear_sec(Bc, d_c))
            out[eid] = {'type': 'col', 'N_G': N, 'Mr': Mr, 'ln': ln, 'Ve': Ve, 'Vr0': Vr0,
                        'rv': rv, 'fv': shear_factor(rv), 'brittle_pre': brittle,
                        'bw': Bc, 'h': Hc, 'd': d_c, 'Asw': Asw_c, 's': s_c}
        else:
            ln = max(meta['L'] - Hc, 0.3 * meta['L'])
            Vdy = w * ln / 2.0
            Ve = Vdy + (Mr_neg_b + Mr_pos_b) / ln
            Vr0 = shear_strength(bb, hb, d_b, fc, fy, Asw_b, s_b, 0.0)
            rv = Ve / (bb * d_b * fct)
            brittle = bool(Ve > Vr0 or Ve > lim_shear_sec(bb, d_b))
            out[eid] = {'type': 'beam', 'N_G': 0.0, 'Mr_pos': Mr_pos_b, 'Mr_neg': Mr_neg_b,
                        'ln': ln, 'Ve': Ve, 'Vr0': Vr0, 'rv': rv, 'fv': shear_factor(rv),
                        'brittle_pre': brittle, 'bw': bb, 'h': hb, 'd': d_b,
                        'Asw': Asw_b, 's': s_b}
    return out


def shear_capacity_at(cap, fc_kPa, fyw_kPa, N_kN):
    return shear_strength(cap['bw'], cap['h'], cap['d'], fc_kPa, fyw_kPa,
                          cap['Asw'], cap['s'], N_kN)
