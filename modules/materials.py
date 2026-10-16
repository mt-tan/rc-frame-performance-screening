"""
materials.py — Makale 3: OpenSeesPy malzemeleri + TBEC 2018 Ek 5A Mander modeli.

Orijinal (modules_orijinal/materials.py) üzerine farklar:
  * Mevcut bina: fce = fc, fywe = fyw (config.TBDY_MANDER oranları 1.0) (K-05).
  * εsu donatı sınıfına bağlı (S220: 0.12, S420: 0.08; Tablo 5A.1) → Mander
    εcu, εs(GÖ) = 0.4·εsu ve MinMax kopma sınırı sınıfa göre (K-06).
  * Kanca etkinliği η (135°: 1.0, 90°: 0.30; TBEC 15.7.1.3) enine donatı
    alanına uygulanır: hem ρsh (ωwe, Denk. 5.4c) hem Mander sargı basıncı ve
    εcu hesabında (K-07; yönetmelik yalnız 5.4d'yi anar, model tutarlılığı
    için sargı hesabına da uygulanmıştır).
  * Σai² (ke ve αse) tanımları orijinal ile aynı (Excel doğrulamalı D-52/D-53/D-67).
"""

import openseespy.opensees as ops
import numpy as np
import config


def mander_tbdy_confined(B_m, H_m, cover_m, fco_kPa, fyw_kPa,
                          phi_long_m, n_long_x, n_long_y,
                          phi_etr_m, s_etr_m, n_etr_x, n_etr_y,
                          eps_su, hook_eff=1.0):
    """
    B_m: z (düzlem dışı) boyutu, H_m: y (düzlem içi, eğilme derinliği) boyutu.
    n_long_x: B yüzlerindeki çubuk sayısı, n_long_y: H yüzlerindeki çubuk sayısı.
    n_etr_x: z'ye paralel kollar (ρx = n_etr_x·A/(s·ho)),
    n_etr_y: y'ye paralel kollar (ρy = n_etr_y·A/(s·bo)).
    fco_kPa, fyw_kPa: MEVCUT dayanımlar (beklenen çarpanı yok).
    hook_eff: etriye kanca etkinliği (TBEC 15.7.1.3).
    """
    eps_co = config.TBDY_MANDER['eps_co']
    fco = fco_kPa / config.MPa
    fyw = fyw_kPa / config.MPa

    bo = B_m - 2 * cover_m - phi_etr_m
    ho = H_m - 2 * cover_m - phi_etr_m

    n_long_total = 2 * n_long_x + 2 * max(0, n_long_y - 2)
    As_total = n_long_total * np.pi * phi_long_m ** 2 / 4.0
    A_core = bo * ho
    rho_cc = As_total / A_core

    aix = (bo - phi_etr_m - phi_long_m) / (n_long_x - 1) if n_long_x >= 2 else 0.0
    aiy = (ho - phi_etr_m - phi_long_m) / (n_long_y - 1) if n_long_y >= 2 else 0.0
    Sigma_ai2_ke = (2 * max(0, n_long_x - 1) * aix ** 2
                    + 2 * max(0, n_long_y - 1) * aiy ** 2)

    span_x = bo - phi_etr_m - phi_long_m
    span_y = ho - phi_etr_m - phi_long_m
    Sigma_aix2 = (span_x ** 2 * 4.0) if n_etr_y == 2 else (span_x ** 2 / (n_etr_y - 1) * 2.0)
    Sigma_aiy2 = (span_y ** 2 * 2.0) if n_etr_x == 2 else (span_y ** 2 / (n_etr_x - 1) * 2.0)
    Sigma_ai2_ase = Sigma_aix2 + Sigma_aiy2

    A_etr = np.pi * phi_etr_m ** 2 / 4.0 * hook_eff
    rho_x = (n_etr_x * A_etr) / (s_etr_m * ho)
    rho_y = (n_etr_y * A_etr) / (s_etr_m * bo)
    rho_s = rho_x + rho_y

    ke_num = (1.0 - Sigma_ai2_ke / (6.0 * bo * ho)) \
        * (1.0 - s_etr_m / (2.0 * bo)) * (1.0 - s_etr_m / (2.0 * ho))
    ke = max(0.0, min(1.0, ke_num / (1.0 - rho_cc)))

    fex = ke * rho_y * fyw
    fey = ke * rho_x * fyw
    fe = 0.5 * (fex + fey)
    fe_ratio = fe / fco
    lambda_c = -1.254 + 2.254 * np.sqrt(1.0 + 7.94 * fe_ratio) - 2.0 * fe_ratio
    fcc = lambda_c * fco
    eps_cc = eps_co * (1.0 + 5.0 * (lambda_c - 1.0))
    Ec = 5000.0 * np.sqrt(fco)
    eps_cu = 0.0035 + 1.4 * rho_s * fyw * eps_su / fcc

    alpha_se = (1.0 - Sigma_ai2_ase / (6.0 * bo * ho)) \
        * (1.0 - s_etr_m / (2.0 * bo)) * (1.0 - s_etr_m / (2.0 * ho))
    alpha_se = max(0.0, min(1.0, alpha_se))
    fce = config.TBDY_MANDER['fce_ratio'] * fco * config.KNOWLEDGE_FACTOR
    fywe = config.TBDY_MANDER['fywe_ratio'] * fyw * config.KNOWLEDGE_FACTOR
    rho_sh = min(rho_x, rho_y)
    omega_we = alpha_se * rho_sh * fywe / fce

    P = config.TBDY_PERFORMANCE_STRAINS
    eps_c_GO = min(P['eps_c_GO_base'] + P['eps_c_GO_coef'] * np.sqrt(max(0.0, omega_we)),
                   P['eps_c_GO_cap'])
    eps_s_GO = P['eps_s_GO_ratio'] * eps_su
    return {
        'bo_m': bo, 'ho_m': ho, 'A_core_m2': A_core,
        'rho_cc': rho_cc, 'rho_x': rho_x, 'rho_y': rho_y, 'rho_s': rho_s, 'rho_sh': rho_sh,
        'Sigma_ai2_ke_m2': Sigma_ai2_ke, 'Sigma_ai2_ase_m2': Sigma_ai2_ase,
        'ke': ke, 'fe_MPa': fe, 'lambda_c': lambda_c, 'fcc_MPa': fcc,
        'eps_cc': eps_cc, 'eps_cu': eps_cu, 'Ec_MPa': Ec,
        'fco_model_MPa': fco, 'fyw_model_MPa': fyw, 'hook_eff': hook_eff,
        'alpha_se': alpha_se, 'omega_we': omega_we, 'fce_MPa': fce, 'fywe_MPa': fywe,
        'eps_su': eps_su,
        'eps_c_SH': P['eps_c_SH'], 'eps_c_KH': P['KH_to_GO_ratio'] * eps_c_GO,
        'eps_c_GO': eps_c_GO,
        'eps_s_SH': P['eps_s_SH'], 'eps_s_KH': P['KH_to_GO_ratio'] * eps_s_GO,
        'eps_s_GO': eps_s_GO,
    }


def E_concrete(fc_kPa):
    return 5000.0 * np.sqrt(fc_kPa / config.MPa) * config.MPa


def create_concrete02_unconfined(tag, fc_kPa):
    fpc = -fc_kPa
    c = config.CONCRETE02
    ops.uniaxialMaterial('Concrete02', tag, fpc, c['epsc0_unconf'], 0.20 * fpc,
                         c['epscu_unconf'], c['lambda_param'],
                         c['ft_ratio'] * fc_kPa, c['Ets_ratio'] * E_concrete(fc_kPa))
    return tag


def create_concrete02_confined_from_mander(tag, fc_kPa, mander):
    c = config.CONCRETE02
    fpcc = -mander['fcc_MPa'] * config.MPa
    ops.uniaxialMaterial('Concrete02', tag, fpcc, -mander['eps_cc'], 0.20 * fpcc,
                         -mander['eps_cu'], c['lambda_param'],
                         c['ft_ratio'] * abs(fpcc), c['Ets_ratio'] * E_concrete(fc_kPa))
    return tag


def create_steel02(tag, fy_kPa, b, eps_su):
    p = config.STEEL02
    if config.USE_STEEL_RUPTURE:
        inner = tag + 90000
        ops.uniaxialMaterial('Steel02', inner, fy_kPa, p['E0'], b, p['R0'], p['cR1'], p['cR2'])
        ops.uniaxialMaterial('MinMax', tag, inner, '-min', -eps_su, '-max', eps_su)
    else:
        ops.uniaxialMaterial('Steel02', tag, fy_kPa, p['E0'], b, p['R0'], p['cR1'], p['cR2'])
    return tag


def setup_materials_with_mander(fc_kPa, mander, tag_offset, fy_kPa, steel_b, eps_su):
    tags = {'cover': tag_offset + 1, 'core': tag_offset + 2, 'steel': tag_offset + 3}
    create_concrete02_unconfined(tags['cover'], fc_kPa)
    create_concrete02_confined_from_mander(tags['core'], fc_kPa, mander)
    create_steel02(tags['steel'], fy_kPa, steel_b, eps_su)
    return tags
