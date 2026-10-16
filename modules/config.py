"""
config.py — Makale 3 (mevcut bina performans sınıflandırması) üretim konfigürasyonu.

Kaynak: 2R/2C üretim kodunun (modules_orijinal/config.py) mevcut bina
popülasyonuna uyarlanmış sürümü. Tüm değişiklikler docs/KARAR_KAYDI.md'de
gerekçeleriyle listelenir (K-xx numaraları kod yorumlarında geçer).

BİRİM SİSTEMİ: m, kN, ton, s, kPa (MPa girdisi `MPa` ile çarpılır).

Temel farklar (özet):
  * Tek saha (Nurdağı) yerine 8 saha (Kahramanmaraş çevresi, AFAD TDTH), her
    çerçeve için DD-1 ve DD-2 ayrı ayrı (K-01, K-02).
  * Mevcut bina malzeme kabulü: mevcut dayanım, bilgi düzeyi katsayısı 1.0
    (TBEC 15.2.12-b, 15.4.9-a, Tablo 15.1) → ωwe ve fiberlerde fce = fc,
    fywe = fywk (K-05).
  * Donatı sınıfı {S220, S420} (Tablo 5A.1), S220 düz donatı → donatı birim
    şekildeğiştirme talebi ×1.5 (TBEC 15.7.1.2) (K-06).
  * Etriye kanca tipi {135°, 90°}; 90° kancada ρsh'nin %30'u (TBEC 15.7.1.3) (K-07).
  * Donatı gerçekleşme katsayısı κ (TBEC 15.2.5.2) kolon ve kiriş için ayrı (K-08).
  * Çatlamış kesit rijitliği (TBEC 15.4.11, Tablo 4.2): kiriş 0.35, kolon 0.70 (K-09).
  * Dikdörtgen kolon: düzlem içi derinlik h ∈ [0.25, 1.00] m, h/b ∈ [1.0, 2.5],
    b ≥ 0.25 m; güçlü eksen düzlem içinde (K-10).
"""

import os
import numpy as np

# ============================================================
# 1. BİRİM SİSTEMİ ÇARPANLARI
# ============================================================
MPa = 1.0e3       # 1 MPa = 1000 kPa
GPa = 1.0e6       # 1 GPa = 1e6 kPa
cm  = 0.01
mm  = 0.001
g = 9.81          # m/s²

# ============================================================
# 2. SAHALAR — AFAD TDTH raporları (data/sites/L*-D*.pdf, 06.10.2026)
# ------------------------------------------------------------
# Ss, S1 harita spektral ivme katsayıları zemin sınıfından bağımsızdır
# (TBEC 2018 Madde 2.3.2.1, referans zemin VS30 = 760 m/s). Değerler
# raporlardan aynen alınmıştır; doğrulama: data/sites/P2_Spektrum_Hesabi_DD1_DD2.xlsx
# (ZA için hesap = AFAD çıktısı, en büyük fark 0.0005 yuvarlama).
# ============================================================
SITES = {
    'L1': {'lat': 37.438185, 'lon': 36.863235,
           'DD1': {'Ss': 2.185, 'S1': 0.619}, 'DD2': {'Ss': 1.112, 'S1': 0.295}},
    'L2': {'lat': 37.482316, 'lon': 36.659379,
           'DD1': {'Ss': 1.456, 'S1': 0.413}, 'DD2': {'Ss': 0.755, 'S1': 0.208}},
    'L3': {'lat': 37.697696, 'lon': 37.054736,
           'DD1': {'Ss': 1.444, 'S1': 0.411}, 'DD2': {'Ss': 0.758, 'S1': 0.211}},
    'L4': {'lat': 37.565605, 'lon': 37.366697,
           'DD1': {'Ss': 2.629, 'S1': 0.712}, 'DD2': {'Ss': 1.359, 'S1': 0.348}},
    'L5': {'lat': 38.000125, 'lon': 37.184462,
           'DD1': {'Ss': 1.815, 'S1': 0.491}, 'DD2': {'Ss': 0.825, 'S1': 0.223}},
    'L6': {'lat': 37.980651, 'lon': 36.585250,
           'DD1': {'Ss': 1.660, 'S1': 0.438}, 'DD2': {'Ss': 0.746, 'S1': 0.193}},
    'L7': {'lat': 37.614555, 'lon': 36.294910,
           'DD1': {'Ss': 1.480, 'S1': 0.377}, 'DD2': {'Ss': 0.719, 'S1': 0.184}},
    'L8': {'lat': 37.827112, 'lon': 36.758218,
           'DD1': {'Ss': 1.179, 'S1': 0.327}, 'DD2': {'Ss': 0.589, 'S1': 0.170}},
}
SITE_LIST = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
# Kullanıcı kararı (2026-10-06): saha-dışı test için L5 (ara değer) ve L4 (en
# yüksek tehlike, dış değer). Üretimde 8 saha eşit olasılıkla örneklenir;
# saha-dışı deney ML aşamasında bu iki sahayı eğitimden çıkarır (K-03).
HOLDOUT_SITES = ['L4', 'L5']
HAZARD_LEVELS = ['DD1', 'DD2']        # DD-3 yok (kullanıcı kararı)

# ============================================================
# 3. TBEC 2018 TABLO 2.1 / 2.2 — yerel zemin etki katsayıları
#    (resmi metinden doğrulandı: references/TBEC_2018.pdf, s. 8)
# ============================================================
TABLE_2_1_FS_BREAKPOINTS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
TABLE_2_1_FS = {
    'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    'ZB': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'ZC': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
    'ZD': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
    'ZE': [2.4, 1.7, 1.3, 1.1, 0.9, 0.8],
}
TABLE_2_2_F1_BREAKPOINTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
TABLE_2_2_F1 = {
    'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    'ZB': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    'ZC': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
    'ZD': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
    'ZE': [4.2, 3.3, 2.8, 2.4, 2.2, 2.0],
}
TL = 6.0                                   # s — TBEC 2018 Madde 2.3.4
SOIL_ORDINAL = {'ZA': 0, 'ZB': 1, 'ZC': 2, 'ZD': 3, 'ZE': 4}
SOIL_CLASSES = ['ZA', 'ZB', 'ZC', 'ZD', 'ZE']


def spectrum_params(Ss, S1, soil_class):
    """TBEC 2018 Madde 2.3.2-2.3.4: FS, F1 (doğrusal enterpolasyon, tablo
    dışında uç sütun), SDS = Ss·FS, SD1 = S1·F1, TB = SD1/SDS, TA = 0.2·TB.
    np.interp tablo uçlarında sabit kalır (≤ ilk, ≥ son sütun) → yönetmelik
    başlıklarındaki ≤ / ≥ anlamıyla birebir."""
    FS = float(np.interp(Ss, TABLE_2_1_FS_BREAKPOINTS, TABLE_2_1_FS[soil_class]))
    F1 = float(np.interp(S1, TABLE_2_2_F1_BREAKPOINTS, TABLE_2_2_F1[soil_class]))
    SDS = Ss * FS
    SD1 = S1 * F1
    TB = SD1 / SDS
    TA = 0.2 * TB
    return {'FS': FS, 'F1': F1, 'SDS': SDS, 'SD1': SD1, 'TA': TA, 'TB': TB, 'TL': TL}


def site_spectrum(site, level, soil_class):
    s = SITES[site][level]
    out = spectrum_params(s['Ss'], s['S1'], soil_class)
    out.update({'Ss': s['Ss'], 'S1': s['S1'], 'site': site, 'level': level,
                'soil_class': soil_class})
    return out


# ============================================================
# 4. SABİT YAPISAL KATSAYILAR
# ============================================================
BKS     = 3        # konut (Tablo 3.1)
I_E     = 1.0      # TBEC 15.4.2: mevcut bina değerlendirmesinde I = 1.0
n_combo = 0.30     # TBEC Tablo 4.3: konut n = 0.30

# ============================================================
# 5. MALZEME — MEVCUT BİNA KABULLERİ (K-05, K-06)
# ============================================================
gamma_RC = 25.0
Es_rebar = 200 * GPa

# TBEC 2018 Tablo 5A.1 (references/TBEC_2018.pdf). S220 düz (nervürsüz),
# S420 nervürlü kabul edilir. fsu/fsy: S220 = 1.20 (tablo); S420 için
# 1.15-1.35 aralığından 2R ile aynı 550/420 = 1.31.
REBAR_GRADES = {
    'S220': {'fy_MPa': 220.0, 'eps_sy': 0.0011, 'eps_sh': 0.011,
             'eps_su': 0.12, 'fsu_MPa': 264.0, 'plain': True},
    'S420': {'fy_MPa': 420.0, 'eps_sy': 0.0021, 'eps_sh': 0.008,
             'eps_su': 0.08, 'fsu_MPa': 550.0, 'plain': False},
}
GRADE_LIST = ['S220', 'S420']
for _g in REBAR_GRADES.values():
    # Steel02 pekleşme oranı b = (fsu − fsy) / (Es·(εsu − εsy))
    _g['b_steel02'] = (_g['fsu_MPa'] - _g['fy_MPa']) / (
        200000.0 * (_g['eps_su'] - _g['eps_sy']))

# TBEC 15.7.1.2: düz donatıda donatı birim şekildeğiştirme talebi × 1.5
PLAIN_BAR_STRAIN_FACTOR = 1.5

# Mevcut malzeme dayanımı: bilgi düzeyi kapsamlı → katsayı 1.00 (Tablo 15.1).
# ωwe = αse·ρsh·fyw/fc (beklenen dayanım çarpanları UYGULANMAZ, 15.2.12-b).
KNOWLEDGE_FACTOR = 1.00
USE_EXPECTED_STRENGTH_FIBERS = False

# Steel02 (Giuffré-Menegotto-Pinto) ortak parametreleri
STEEL02 = {'E0': Es_rebar, 'R0': 18.0, 'cR1': 0.925, 'cR2': 0.15}
USE_STEEL_RUPTURE = True      # MinMax ile εsu'da kopma (sınıfa göre εsu)

CONCRETE02 = {
    'epsc0_unconf':  -0.002,
    'epscu_unconf':  -0.005,
    'lambda_param':   0.10,
    # K-28: beton çekme dayanımı 0 (TBEC Ek 5A beton modelleri yalnız basınç tanımlar;
    # ft = 0.1·fc çekme yumuşaması derin dikdörtgen kesitlerde eleman içi
    # yakınsama kaybı üretiyordu — qa/logs/qa02_variants.csv). Orijinal: 0.10.
    'ft_ratio':       0.0,
    'Ets_ratio':      0.10,
}

TBDY_MANDER = {
    'eps_co':     0.002,
    'fce_ratio':  1.0,       # K-05: mevcut dayanım (beklenen değil)
    'fywe_ratio': 1.0,       # K-05
}

# TBEC 5.8.1 (15.7.1.3 ile mevcut binalara uygulanır)
TBDY_PERFORMANCE_STRAINS = {
    'eps_c_SH':       0.0025,
    'eps_c_GO_base':  0.0035,
    'eps_c_GO_coef':  0.04,
    'eps_c_GO_cap':   0.018,
    'KH_to_GO_ratio': 0.75,
    'eps_s_SH':       0.0075,
    'eps_s_GO_ratio': 0.4,
}

# TBEC 15.7.1.3: 90° kapalı etriyelerin %30'u ρsh hesabına dahil (K-07)
HOOK_EFFECTIVENESS = {135: 1.00, 90: 0.30}

# TBEC 15.7.1.4: kesme oranı Ve/(bw·d·fctm) ≤ 0.65 → sınırlar aynen;
# ≥ 1.30 → ×0.50; arada doğrusal (K-12). fctm = 0.35·√fc (TS 500, mevcut fc).
SHEAR_RATIO_LIMITS = (0.65, 1.30)
SHEAR_RATIO_MIN_FACTOR = 0.50
FCTM_COEF = 0.35

# ============================================================
# 6. ÖRNEKLEME ARALIKLARI (K-04, K-10, K-11)
# ============================================================
FC_RANGE_MPA = (15.0, 35.0)          # mevcut beton dayanımı (kullanıcı)
COL_H_RANGE = (0.25, 1.00)           # düzlem içi kolon derinliği h (m)
COL_ASPECT_RANGE = (1.0, 2.5)        # h/b (K-10)
COL_B_MIN = 0.25                     # düzlem dışı genişlik b alt sınırı (kullanıcı)
S_MIN_SAMPLING = 0.08                # etriye aralığı alt sınırı (kullanıcı: 8 cm)
S_MAX_SAMPLING = 0.25                # üst sınır (kullanıcı: 25 cm)
COL_LEG_CHOICES = [2, 3, 4]
KAPPA_MIN = 0.60                     # donatı gerçekleşme katsayısı alt sınırı
KAPPA_P_ONE = 0.50                   # çerçevelerin yarısında κ = 1.0 (K-08)
P_HOOK_90 = 0.50                     # 90° kanca olasılığı (K-07)
SPAN_RANGE = (3.0, 8.0)
ENFORCE_TBDY_CONFINEMENT = False

# ============================================================
# 7. ÇATLAMIŞ KESİT RİJİTLİĞİ (TBEC 15.4.11 → 4.5.8, Tablo 4.2) (K-09)
#    HingeRadau elastik orta bölgesine uygulanır (fiber mafsallar değişmez).
# ============================================================
EI_FACTOR_BEAM = 0.35
EI_FACTOR_COL = 0.70

# ============================================================
# 8. PUSHOVER PROTOKOLÜ (2R/2C ile aynı; K-13)
# ============================================================
PUSHOVER = {
    'control_dof':           1,
    'disp_step':             0.001,
    'max_disp_fraction':     0.08,
    'load_pattern':          'first_mode',
    'stop_ratio_post_peak':  0.80,
    'min_disp_step':         1e-5,
    'step_halving_limit':    2,
    'peak_drop_ratio':       0.98,
    'failed_steps_limit':    3,
    'min_accept_fraction':   0.04,
}

ANALYSIS = {
    'tolerance':       1.0e-6,
    'max_iter':        50,
    'gravity_steps':   10,
    'pdelta_enabled':  True,
    'algorithm_chain': [
        ('Newton',           {}),
        ('KrylovNewton',     {}),
        ('NewtonLineSearch', {'type': 'Bisection'}),
        ('ModifiedNewton',   {}),
    ],
    'num_modes':       3,
}

FIBERS = {
    'col_core_BH':         (10, 10),
    'col_cover_BH':        (10, 10),
    'beam_core_bh':        (8, 12),
    'beam_cover_bh':       (8, 12),
    'integration_points':  5,
}

INTEGRATION = {'method': 'HingeRadau', 'lp_formula': 'TBDY'}

# K-28: forceBeamColumn eleman içi iterasyon (maxIter, tol); None → varsayılan
FBC_ITER = (20, 1e-10)   # qa02: sonuçları değiştirmez (226/226 aynı etiket), başarıyı artırır

# Rijit uç: kiriş ucunda kolonun DÜZLEM İÇİ derinliğinin yarısı (K-10;
# orijinal kod col_B kullanıyordu, kare kesitte fark yoktu).
RIGID_END = {'enabled': True, 'fraction_beam': 0.5}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
