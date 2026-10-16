# -*- coding: utf-8 -*-
"""
features_m3.py — Makale 3: ML girdi blokları.

BASE (tasarım/saha girdileri): mühendisin binadan topladığı bilgi (geometri, kesitler,
  malzeme, gerçekleşen donatı, etriye, kanca, yükler, saha SDS/SD1, zemin).
A2 (kapalı form, analiz YOK) — her düzey için:
  * TpA = 0.1·HN^0.75 (4.7.3.4); Sae(TpA), Sde(TpA)
  * Γ_lin: doğrusal mod şekli φi = Hi/HN, kat kütleleri (G + 0.3Q)·W_total/g
  * θ_A2 = Γ_lin·CR·Sde(TpA)/HN  (eşit yerdeğiştirme; CR TBEC 5.16 ile, Ry = Sae/(Vmech/W))
  * Vmech: min(1. kat kolon mekanizması Σ2Mr/(h1 − hb), kiriş mekanizması
    [Σ_kirişler(Mr+ + Mr−) + Σ_taban kolonları Mr]/H_eff), H_eff = Σ m H²/Σ m H
  * R_A2 = Vmech/(W·Sae(TpA))  (dayanım/talep)
  * ωwe, εc(GÖ) kolon/kiriş (Denk. 5.4); kolon/kiriş Ve/Vr ve rv (yaklaşık N ile)
  * ΣMc/ΣMb (scwb_ratio), eksenel yük oranı tahmini
Yaklaşık kolon eksenel kuvveti: kat başına (G+0.3Q)·kolon etki genişliği (iç kolon
  ort. açıklık, kenar kolon yarısı) × üstteki kat sayısı.
"""

import math
import numpy as np

import config
import materials
import member_capacity as MC
from target_displacement import tbdy_design_spectrum

BASE_NUM = ['N_story', 'N_bay', 'L_mean', 'L_max', 'L_min', 'W_total', 'h1', 'htyp', 'Htot',
            'col_B', 'col_H', 'col_aspect', 'beam_b', 'beam_h', 'cover', 'fc_MPa', 'fy_MPa',
            'G', 'Q', 'kappa_col', 'kappa_beam', 'rho_col_actual', 'rho_beam_top_actual',
            'rho_beam_bot_actual', 'phi_long_col_mm', 'phi_long_beam_mm', 'col_phi_etr',
            'col_s_etr', 'col_n_etr_x', 'col_n_etr_y', 'beam_phi_etr', 'beam_s_etr', 'hook_deg']


def base_features(p, lev=None):
    f = {k: float(p[k]) for k in BASE_NUM if k in p}
    if 'Htot' not in p:
        f['Htot'] = p['h1'] + (p['N_story'] - 1) * p['htyp']
    f['plain_bars'] = float(bool(p.get('plain_bars', p.get('steel_grade') == 'S220')))
    f['soil'] = float(config.SOIL_ORDINAL[p['soil_class']])
    levs = [lev] if lev else config.HAZARD_LEVELS
    for L in levs:
        sp = config.site_spectrum(p['site'], L, p['soil_class'])
        sfx = '' if lev else f'_{L}'
        f['SDS' + sfx] = sp['SDS']
        f['SD1' + sfx] = sp['SD1']
    return f


def a2_features(p):
    fc, fy = p['fc'], p['fy']
    N, nb = int(p['N_story']), int(p['N_bay'])
    H = p['h1'] + (N - 1) * p['htyp']
    w = p['G'] + config.n_combo * p['Q']
    Wf = w * p['W_total']                       # kat ağırlığı (kN)
    W = Wf * N
    Hs = np.array([p['h1'] + i * p['htyp'] for i in range(N)])
    m = np.full(N, Wf / config.g)
    phi = Hs / H
    Gam = float((m * phi).sum() / (m * phi ** 2).sum())
    Heff = float((m * Hs ** 2).sum() / (m * Hs).sum())
    TpA = 0.1 * H ** 0.75
    hook = config.HOOK_EFFECTIVENESS[int(p['hook_deg'])]
    mc = materials.mander_tbdy_confined(
        p['col_B'], p['col_H'], p['cover'], fc, fy, p['col_phi_long'], p['col_n_long_x'],
        p['col_n_long_y'], p['col_phi_etr'], p['col_s_etr'], p['col_n_etr_x'], p['col_n_etr_y'],
        p['eps_su'], hook)
    mb = materials.mander_tbdy_confined(
        p['beam_b'], p['beam_h'], p['cover'], fc, fy, p['beam_phi_long'],
        max(p['beam_n_long_top'], p['beam_n_long_bot']), 2, p['beam_phi_etr'], p['beam_s_etr'],
        p['beam_n_etr_x'], p['beam_n_etr_y'], p['eps_su'], hook)
    lay_c, lay_b = MC.column_layers(p), MC.beam_layers(p)
    Lavg = p['W_total'] / nb
    # 1. kat kolon eksenel kuvveti (yaklaşık): iç kolon etki genişliği = ort. açıklık
    N_int = w * Lavg * N
    N_ext = 0.5 * N_int
    Mr_int = MC.moment_capacity(p['col_B'], p['col_H'], lay_c, fc, fy, N_int)
    Mr_ext = MC.moment_capacity(p['col_B'], p['col_H'], lay_c, fc, fy, N_ext)
    Mr_bp = MC.moment_capacity(p['beam_b'], p['beam_h'], lay_b, fc, fy, 0.0)
    Mr_bn = MC.moment_capacity(p['beam_b'], p['beam_h'], [(-y, A) for y, A in lay_b], fc, fy, 0.0)
    hcl = max(p['h1'] - p['beam_h'], 0.3 * p['h1'])
    V_col = (2 * (nb - 1) * Mr_int + 2 * 2 * Mr_ext) / hcl
    V_beam = (N * nb * (Mr_bp + Mr_bn) + (nb - 1) * Mr_int + 2 * Mr_ext) / Heff
    Vmech = min(V_col, V_beam)
    # kesme oranları (Ve/Vr, rv) — iç kolon ve en uzun açıklıktaki kiriş
    d_c = p['col_H'] - p['cover'] - p['col_phi_etr'] - p['col_phi_long'] / 2
    Asw_c = p['col_n_etr_y'] * math.pi * p['col_phi_etr'] ** 2 / 4
    Ve_c = 2 * Mr_int / hcl
    Vr_c = MC.shear_strength(p['col_B'], p['col_H'], d_c, fc, fy, Asw_c, p['col_s_etr'], N_int)
    d_b = p['beam_h'] - p['cover'] - p['beam_phi_etr'] - p['beam_phi_long'] / 2
    Asw_b = p['beam_n_etr_y'] * math.pi * p['beam_phi_etr'] ** 2 / 4
    ln_b = max(p['L_max'] - p['col_H'], 0.3 * p['L_max'])
    Ve_b = w * ln_b / 2 + (Mr_bp + Mr_bn) / ln_b
    Vr_b = MC.shear_strength(p['beam_b'], p['beam_h'], d_b, fc, fy, Asw_b, p['beam_s_etr'], 0.0)
    fct = MC.fctm_kPa(fc)
    f = {'A2_TpA': TpA, 'A2_Gamma_lin': Gam, 'A2_Vmech_W': Vmech / W,
         'A2_Vcolmech_W': V_col / W, 'A2_Vbeammech_W': V_beam / W,
         'A2_mech_beam_over_col': V_beam / V_col,
         'A2_omega_we_col': mc['omega_we'], 'A2_omega_we_beam': mb['omega_we'],
         'A2_eps_cGO_col': mc['eps_c_GO'], 'A2_eps_cGO_beam': mb['eps_c_GO'],
         'A2_eps_sGO': mc['eps_s_GO'],
         'A2_VeVr_col': Ve_c / Vr_c, 'A2_VeVr_beam': Ve_b / Vr_b,
         'A2_rv_col': Ve_c / (p['col_B'] * d_c * fct), 'A2_rv_beam': Ve_b / (p['beam_b'] * d_b * fct),
         'A2_fv_col': MC.shear_factor(Ve_c / (p['col_B'] * d_c * fct)),
         'A2_axial_ratio': N_int / (p['col_B'] * p['col_H'] * fc),
         'A2_scwb': p.get('scwb_ratio', float('nan'))}
    for lev in config.HAZARD_LEVELS:
        sp = config.site_spectrum(p['site'], lev, p['soil_class'])
        Sae = tbdy_design_spectrum(TpA, sp['SDS'], sp['SD1'], sp['TA'], sp['TB'], sp['TL'])
        Sde = Sae * config.g * (TpA / (2 * math.pi)) ** 2
        R = Vmech / (W * Sae)
        Ry = max(1.0 / R, 1.0) if R > 0 else 1.0
        CR = 1.0 if TpA >= sp['TB'] else max(1.0, (1.0 + (Ry - 1.0) * sp['TB'] / TpA) / Ry)
        f[f'A2_Sae_{lev}'] = Sae
        f[f'A2_Sde_{lev}'] = Sde
        f[f'A2_R_{lev}'] = R
        f[f'A2_theta_{lev}'] = Gam * CR * Sde / H
    return f


# ----------------------------------------------------------------------
# K-43: A2s — kat düzeyinde kapalı form göstergeler (yapısal analiz yok)
# ----------------------------------------------------------------------
# Kat kesme talebi: Eşdeğer Deprem Yükü dağılımı (4.7.2), Sae(TpA), Ra = 1.
# Kolon ve kiriş uç momentleri: portal yöntemi (iç kolon V/nb, kenar kolon V/(2nb); dönüm noktası
#   kolon ortasında; iç düğüm momenti iki kirişe eşit paylaşılır). EKO = moment / Mr.
# Kat rijitliği ve ötelemesi: Muto D yöntemi (çatlamış EI: kolon 0.70, kiriş 0.35; 1. kat ankastre
#   taban a = (0.5 + k̄)/(2 + k̄), üst katlar a = k̄/(2 + k̄)); öteleme CR ile büyütülür.
# Kat mekanizma dayanımı: her kolon ucu min(Mr_kolon, düğümdeki kiriş kapasitesinin payı);
#   taban ucu Mr_kolon. Eksenel kuvvet: (G + 0.3Q)·etki boyu·üstteki kat sayısı (A2 ile aynı kabul).

def a2s_features(p):
    fc, fy = p['fc'], p['fy']
    N, nb = int(p['N_story']), int(p['N_bay'])
    bays = [float(x) for x in p['bay_lengths']]
    h = [p['h1']] + [p['htyp']] * (N - 1)
    Hs = np.cumsum(h)
    H = float(Hs[-1])
    w = p['G'] + config.n_combo * p['Q']
    Wf = w * p['W_total']
    W = Wf * N
    Lavg = p['W_total'] / nb
    Ec = 5000.0 * math.sqrt(fc / 1000.0) * 1000.0
    Ic = p['col_B'] * p['col_H'] ** 3 / 12.0
    Ib = p['beam_b'] * p['beam_h'] ** 3 / 12.0
    lay_c, lay_b = MC.column_layers(p), MC.beam_layers(p)
    Mr_bp = MC.moment_capacity(p['beam_b'], p['beam_h'], lay_b, fc, fy, 0.0)
    Mr_bn = MC.moment_capacity(p['beam_b'], p['beam_h'], [(-y, A) for y, A in lay_b], fc, fy, 0.0)
    Mb_int = Mr_bp + Mr_bn                       # iç düğüm: iki kiriş (biri pozitif, biri negatif)
    Mb_ext = 0.5 * (Mr_bp + Mr_bn)               # kenar düğüm: tek kiriş (iki yönün ortalaması)
    hcl = [max(hi - p['beam_h'], 0.3 * hi) for hi in h]
    # kat kolon eksenel kuvvetleri ve kapasiteleri (indis 0 = 1. kat)
    Nint = [w * Lavg * (N - i) for i in range(N)]
    Next = [0.5 * x for x in Nint]
    Mri = [MC.moment_capacity(p['col_B'], p['col_H'], lay_c, fc, fy, x) for x in Nint]
    Mre = [MC.moment_capacity(p['col_B'], p['col_H'], lay_c, fc, fy, x) for x in Next]
    # kolon kesmesi: Ve = 2Mr/ln, Vr(N) (TS 500, mevcut dayanımlar), 0.85·bw·d·√fc sınırı
    d_c = p['col_H'] - p['cover'] - p['col_phi_etr'] - p['col_phi_long'] / 2
    Asw_c = p['col_n_etr_y'] * math.pi * p['col_phi_etr'] ** 2 / 4
    Vri = [MC.shear_strength(p['col_B'], p['col_H'], d_c, fc, fy, Asw_c, p['col_s_etr'], x) for x in Nint]
    Vre = [MC.shear_strength(p['col_B'], p['col_H'], d_c, fc, fy, Asw_c, p['col_s_etr'], x) for x in Next]
    Vei = [2 * Mri[i] / hcl[i] for i in range(N)]
    Vee = [2 * Mre[i] / hcl[i] for i in range(N)]
    vlim = 0.85 * p['col_B'] * d_c * math.sqrt(fc / 1000.0) * 1000.0
    sh_int, sh_ext = (nb - 1) / nb, 1.0 / nb      # portal yöntemiyle kat kesmesindeki paylar
    brit_i = [Vei[i] > Vri[i] or Vei[i] > vlim for i in range(N)]
    brit_e = [Vee[i] > Vre[i] or Vee[i] > vlim for i in range(N)]
    brit_share = [sh_int * brit_i[i] + sh_ext * brit_e[i] for i in range(N)]
    # düğümlerde güçlü kolon oranı (çatı döşemesi hariç)
    scwb_i = [(Mri[i] + Mri[i + 1]) / Mb_int for i in range(N - 1)]
    scwb_e = [(Mre[i] + Mre[i + 1]) / Mb_ext for i in range(N - 1)]
    ok = [x >= 1.2 for x in scwb_i] * (nb - 1) + [x >= 1.2 for x in scwb_e] * 2
    # kat mekanizma dayanımı (kiriş sınırlı)
    Vcap = []
    for i in range(N):
        sh_top = 1.0 if i == N - 1 else 0.5     # çatı düğümünde kiriş kapasitesi tek kolona
        top_i, top_e = min(Mri[i], sh_top * Mb_int), min(Mre[i], sh_top * Mb_ext)
        if i == 0:
            bot_i, bot_e = Mri[0], Mre[0]
        else:
            bot_i, bot_e = min(Mri[i], 0.5 * Mb_int), min(Mre[i], 0.5 * Mb_ext)
        Vcap.append(((nb - 1) * (top_i + bot_i) + 2 * (top_e + bot_e)) / hcl[i])
    # Muto D yöntemi: kat rijitlikleri
    kc = [config.EI_FACTOR_COL * Ic / hi for hi in h]
    kbs = [config.EI_FACTOR_BEAM * Ib / L for L in bays]
    K = []
    for i in range(N):
        Dsum = 0.0
        for a in range(nb + 1):
            kb_side = (kbs[a - 1] if a > 0 else 0.0) + (kbs[a] if a < nb else 0.0)
            if i == 0:
                kbar = kb_side / kc[i]
                aa = (0.5 + kbar) / (2.0 + kbar)
            else:
                kbar = 2.0 * kb_side / (2.0 * kc[i])
                aa = kbar / (2.0 + kbar)
            Dsum += aa * 12.0 * Ec * config.EI_FACTOR_COL * Ic / h[i] ** 3
        K.append(Dsum)
    thy_c = [Mri[i] * hcl[i] / (6.0 * Ec * Ic) for i in range(N)]     # çift eğrilik akma dönmesi
    Lc_avg = max(Lavg - p['col_H'], 0.3 * Lavg)
    thy_b = 0.5 * (Mr_bp + Mr_bn) * Lc_avg / (6.0 * Ec * Ib)
    f = {'A2s_VrVe_min_int': min(Vri[i] / Vei[i] for i in range(N)),
         'A2s_VrVe_min_ext': min(Vre[i] / Vee[i] for i in range(N)),
         'A2s_brittle_share_max': max(brit_share),
         'A2s_brittle_story_frac': float(np.mean([s > 0 for s in brit_share])),
         'A2s_scwb_min_int': min(scwb_i),
         'A2s_scwb_min_ext': min(scwb_e),
         'A2s_scwb_ok_frac': float(np.mean(ok)),
         # zayıf kat oranı: çatı katını içeren çift hariç (çatı düğümünde kiriş kapasitesinin
         # tamamı tek kolona gider; N = 2 ise tek çift kullanılır)
         'A2s_eta_c_min': min(Vcap[i] / Vcap[i + 1] for i in range(max(N - 2, 1)))}
    # birim taban kesmesi için EDY kat kesmeleri (4.7.2) ve öteleme oranları
    m_i = np.full(N, Wf / config.g)
    dFN = 0.0075 * N
    Fu = (1.0 - dFN) * m_i * Hs / float((m_i * Hs).sum())
    Fu[-1] += dFN
    Vu = np.cumsum(Fu[::-1])[::-1]
    dr_u = np.array([Vu[i] / K[i] / h[i] for i in range(N)])
    f['A2s_eta_k_max'] = max([dr_u[i] / dr_u[i + 1] for i in range(N - 1)]
                             + [dr_u[i + 1] / dr_u[i] for i in range(N - 1)])
    TpA = 0.1 * H ** 0.75
    for lev in config.HAZARD_LEVELS:
        sp = config.site_spectrum(p['site'], lev, p['soil_class'])
        Sae = tbdy_design_spectrum(TpA, sp['SDS'], sp['SD1'], sp['TA'], sp['TB'], sp['TL'])
        V = max(W * Sae, 0.04 * W * sp['SDS']) * Vu        # kat kesme talepleri (kN)
        R = np.array(Vcap) / V
        Rg = float(R.min())
        Ry = max(1.0 / Rg, 1.0) if Rg > 0 else 1.0
        CR = 1.0 if TpA >= sp['TB'] else max(1.0, (1.0 + (Ry - 1.0) * sp['TB'] / TpA) / Ry)
        drift = CR * np.array([V[i] / K[i] / h[i] for i in range(N)])
        Vc_int, Vc_ext = V / nb, V / (2 * nb)
        ek_ci = np.array([Vc_int[i] * hcl[i] / 2 / Mri[i] for i in range(N)])
        ek_ce = np.array([Vc_ext[i] * hcl[i] / 2 / Mre[i] for i in range(N)])
        ek_csw = sh_int * ek_ci + sh_ext * ek_ce
        Mg = w * Lc_avg ** 2 / 12.0
        ek_b = []
        for i in range(N):                                   # kat i'nin üst döşemesi
            Mj = Vc_int[i] * h[i] / 2 + (Vc_int[i + 1] * h[i + 1] / 2 if i < N - 1 else 0.0)
            Ms = 0.5 * Mj * Lc_avg / Lavg                    # iç düğüm: iki kiriş, yüz momenti
            ek_b.append(max((Ms + Mg) / Mr_bn, max(Ms - Mg, 0.0) / Mr_bp))
        ek_b = np.array(ek_b)
        dem_brit = [sh_int * (min(Vei[i], Vc_int[i]) > min(Vri[i], vlim))
                    + sh_ext * (min(Vee[i], Vc_ext[i]) > min(Vre[i], vlim)) for i in range(N)]
        sfx = '_' + lev
        f['A2s_R_min' + sfx] = Rg
        f['A2s_R_1' + sfx] = float(R[0])
        f['A2s_R_2' + sfx] = float(R[1])
        f['A2s_R_upper_min' + sfx] = float(R[2:].min()) if N > 2 else float(R[1])
        f['A2s_R_argmin_rel' + sfx] = float(np.argmin(R)) / (N - 1)
        f['A2s_R_conc' + sfx] = Rg / float(R.mean())
        f['A2s_ekoc_sw_max' + sfx] = float(ek_csw.max())
        f['A2s_ekob_max' + sfx] = float(ek_b.max())
        f['A2s_ekob_mean' + sfx] = float(ek_b.mean())
        f['A2s_ekoc_over_ekob_max' + sfx] = float((ek_csw / ek_b).max())
        f['A2s_drift_max' + sfx] = float(drift.max())
        f['A2s_drift_1' + sfx] = float(drift[0])
        f['A2s_drift_argmax_rel' + sfx] = float(np.argmax(drift)) / (N - 1)
        f['A2s_mu_col_max' + sfx] = float(max(drift[i] / thy_c[i] for i in range(N)))
        f['A2s_mu_beam_max' + sfx] = float(drift.max() / thy_b)
        f['A2s_brittle_dem_share_max' + sfx] = float(max(dem_brit))
    return f
