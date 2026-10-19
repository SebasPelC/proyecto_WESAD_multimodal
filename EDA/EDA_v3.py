"""
EDA_v3.py
Pipeline definitivo (congelado) de características EDA para WESAD.

Genera UN solo archivo, EDA_v3_todos.csv, con exactamente 26 columnas:
  - 4 metadatos : sujeto, t_inicio, etiqueta, pureza   (no son predictores)
  - 7 pecho     : chest_*
  - 15 muñeca   : wrist_*

PIPELINES
  Pecho   : señal cruda 700 Hz -> LP 5 Hz -> downsampling 20 Hz -> MAD sobre el gradiente
            -> validación morfológica -> interpolación -> cvxEDA -> extracción.
  Muñeca  : señal cruda 4 Hz -> máscara ACC -> detección de glitches -> interpolación
            -> cvxEDA -> extracción (incluye las 4 strz_* y spectral_power_low).

VENTANAS VÁLIDAS (inner join por t_inicio entre pecho y muñeca)
  clase dominante en {1, 2, 3}, pureza >= 0.80 y fracción de artefactos <= 0.30 en AMBOS canales.
  La fracción de artefactos solo se usa para filtrar; no se escribe en el CSV.

TRAZABILIDAD
  [PAPER]    lo basado en la literatura (Schmidt et al. 2018; Stržinar et al. 2023).
  [DECISIÓN] decisiones metodológicas propias (cvxEDA, MAD, máscara ACC, 20 Hz, umbrales).

LIMITACIÓN METODOLÓGICA
  La selección fija de estas 22 características (7 de pecho + 15 de muñeca) utilizó información
  agregada de los 15 sujetos (sin selección anidada). Esto implica un ligero sesgo optimista en la
  evaluación LOSO.

Dependencias: numpy, pandas, scipy, cvxEDA (pip install cvxEDA).

Uso:
    python EDA_v3.py --ruta ../data/WESAD --salida ./resultados_eda            # secuencial (debug)
    python EDA_v3.py --ruta /kaggle/input/wesad --salida /kaggle/working --procesos 4
"""

import argparse
import contextlib
import gc
import io
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, label
from scipy.signal import butter, decimate, filtfilt, find_peaks, welch

try:
    from cvxeda import cvxEDA
except ImportError as exc:
    raise ImportError("No se encontró cvxEDA. Instálalo con: pip install cvxEDA") from exc


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SUJETOS_VALIDOS = [f'S{i}' for i in range(2, 18) if i != 12]   # excluye S1 y S12
NOMBRE_SALIDA = 'EDA_v3_todos.csv'

FS_LABEL = 700          # frecuencia de las etiquetas del .pkl
FS_CHEST_RAW = 700      # RespiBAN
FS_CHEST = 20           # [DECISIÓN] pecho tras el downsampling
FS_WRIST = 4            # Empatica E4 (EDA)
FS_ACC_WRIST = 32       # Empatica E4 (ACC)

# Filtro pasa-bajos  [PAPER] Schmidt et al. (2018)
LP_FC_HZ = 5
LP_ORDEN = 2

# Ventaneo  [PAPER] Schmidt et al. (2018)
WINDOW_S = 60.0
STEP_S = 0.25

# Selección de ventanas  [DECISIÓN] (criterio de pureza de Birkenmaier et al., 2026)
CLASES_VALIDAS = (1, 2, 3)
PUREZA_MIN = 0.80
MAX_FRAC_ART = 0.30

# Pecho  [DECISIÓN]
PECHO_MAD_K = 10
PECHO_MAD_DILATACION_S = 0.25
SCR_RISE_MIN, SCR_RISE_MAX = 0.3, 5.0     # s, validación morfológica
SCR_REC_MIN, SCR_REC_MAX = 1.0, 20.0      # s, half-recovery
ONSET_MAX_RETROCESO_S = 10

# Muñeca  [DECISIÓN]
ACC_OFFSET_FIJO = 16              # umbral ACC = mediana del sujeto + 16
ACC_DILATACION_S = 1.5
SPIKE_K = 15
SPIKE_DILATACION_S = 0.5
ACC_P50_RANGO = (55, 75)          # reposo esperado ~64; solo genera un aviso en el QC

# cvxEDA  [DECISIÓN] (delta_knot=5 difiere del valor por defecto de la librería)
CVX_TAU0, CVX_TAU1, CVX_DELTA_KNOT, CVX_ALPHA, CVX_GAMMA = 2.0, 0.7, 5.0, 8e-4, 1e-2

# Extracción de características  [DECISIÓN]
SCR_PROM_MIN = 0.01               # µS, prominencia mínima de un pico SCR
SCR_DIST_MIN_S = 1.0              # s, distancia mínima entre picos
WELCH_SEG_S = 15                  # s, nperseg de Welch
SPECTRAL_LOW_HZ = 0.5             # potencia espectral baja: 0-0.5 Hz

# Espectral de Stržinar  [PAPER] Stržinar et al. (2023)
STRZ_SEG_S = 12                   # segmentos de 12 s
STRZ_PASO_S = 7                   # paso de 7 s
STRZ_N_SEG = 7                    # 7 segmentos por ventana (=54 s, anclados al inicio de la ventana)
STRZ_B1_MAX_HZ = 1.0              # banda 1: [0, 1) Hz
STRZ_B2_HZ = (1.0, 2.0)           # banda 2: [1, 2] Hz

_trapz = getattr(np, 'trapezoid', None) or np.trapz     # NumPy 2.x / 1.x

# Esquema del CSV final (26 columnas, en este orden)
META_COLS = ['sujeto', 't_inicio', 'etiqueta', 'pureza']
BASE_PECHO = ['scl_mean', 'scr_mean', 'scr_std', 'scr_min', 'scr_peak_count', 'scr_peak_mean_amp',
              'eda_spectral_power_low']
BASE_MUNECA = ['scl_mean', 'scl_std', 'scl_variation', 'scl_slope', 'scr_mean', 'scr_std', 'scr_min',
               'scr_peak_count', 'scr_peak_mean_amp', 'scr_peak_sum_amp', 'eda_spectral_power_low',
               'strz_b2_std_mm', 'strz_b2_mean_mv', 'strz_b1_std_mm', 'strz_b2_mean_mm']
COLS_PECHO = [f'chest_{c}' for c in BASE_PECHO]
COLS_MUNECA = [f'wrist_{c}' for c in BASE_MUNECA]
COLUMNAS_FINALES = META_COLS + COLS_PECHO + COLS_MUNECA
assert len(COLUMNAS_FINALES) == 26


# =============================================================================
# 1. CARGA
# =============================================================================

def cargar_sujeto(ruta, id_sujeto):
    ruta_archivo = os.path.join(ruta, id_sujeto, f'{id_sujeto}.pkl')
    with open(ruta_archivo, 'rb') as f:
        return pickle.load(f, encoding='latin1')


# =============================================================================
# 2. FUNCIONES COMUNES: filtro, interpolación y cvxEDA
# =============================================================================

def filtro_pasabajos(senal, fs):
    """[PAPER] Pasa-bajos Butterworth de 5 Hz, orden 2, fase cero (solo se aplica al pecho a 700 Hz)."""
    nyquist = fs / 2
    assert LP_FC_HZ < nyquist, f'corte {LP_FC_HZ} Hz >= Nyquist {nyquist} Hz'
    b, a = butter(LP_ORDEN, LP_FC_HZ / nyquist, btype='low')
    return filtfilt(b, a, senal)


def interpolar_gaps(senal, mascara):
    """Interpola linealmente los tramos enmascarados (True = artefacto). Devuelve una copia sin NaN."""
    idx_validos = np.flatnonzero(~mascara)
    if len(idx_validos) < 2:
        raise ValueError('Demasiadas muestras enmascaradas: señal no interpolable')
    limpia = np.array(senal, dtype=float)
    idx_mask = np.flatnonzero(mascara)
    if len(idx_mask):
        interp = interp1d(idx_validos, limpia[idx_validos], kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        limpia[idx_mask] = interp(idx_mask)
    return limpia


def separar_tonico_fasico_cvxeda(senal, fs):
    """[DECISIÓN] Descomposición tónico/fásico con cvxEDA sobre la señal z-normalizada de todo el registro."""
    senal = np.asarray(senal, dtype=np.float64)
    media, desv = np.mean(senal), np.std(senal)
    senal_z = (senal - media) / desv
    with contextlib.redirect_stdout(io.StringIO()):      # cvxopt imprime el log del solver
        fasico_z, _, tonico_z, *_ = cvxEDA(senal_z, 1.0 / fs, tau0=CVX_TAU0, tau1=CVX_TAU1,
                                           delta_knot=CVX_DELTA_KNOT, alpha=CVX_ALPHA, gamma=CVX_GAMMA)
    fasico = np.asarray(fasico_z).flatten() * desv
    tonico = np.asarray(tonico_z).flatten() * desv + media
    return tonico, fasico


# =============================================================================
# 3. PECHO
# =============================================================================

def downsamplear_pecho(senal):
    """[DECISIÓN] 700 -> 100 Hz -> 20 Hz (decimate en dos etapas, con anti-aliasing)."""
    assert FS_CHEST_RAW // FS_CHEST == 7 * 5 and FS_CHEST_RAW % FS_CHEST == 0
    return decimate(decimate(senal, 7), 5)


def candidatos_mad_gradiente(senal, fs, k=PECHO_MAD_K, dilatacion_s=PECHO_MAD_DILATACION_S):
    """[DECISIÓN] Candidatos a artefacto: gradiente (µS/s) fuera de mediana ± k*MAD*1.4826 (MAD global),
    dilatados para cubrir el inicio y el fin del salto."""
    senal = np.asarray(senal, dtype=float)
    x = np.diff(senal, prepend=senal[0]) * fs
    mediana = np.median(x)
    mad = np.median(np.abs(x - mediana))
    if mad == 0:                                   # señal cuantizada/plana: evitar umbral 0
        mad = np.mean(np.abs(x - mediana)) or 1e-12
    umbral = k * mad * 1.4826
    candidatos = (x < mediana - umbral) | (x > mediana + umbral)
    if dilatacion_s > 0:
        candidatos = binary_dilation(candidatos, iterations=max(1, int(round(dilatacion_s * fs))))
    return candidatos


def validacion_morfologica_scr(senal, candidatos, fs):
    """[DECISIÓN] Cada grupo candidato se conserva como artefacto salvo que su morfología sea la de un SCR
    válido (tiempo de subida y half-recovery dentro de rango). Devuelve la máscara de artefactos confirmados."""
    artefactos = np.zeros_like(candidatos)
    grupos, n_grupos = label(candidatos)
    n = len(senal)

    for i in range(1, n_grupos + 1):
        idx_grupo = np.flatnonzero(grupos == i)
        ini, fin = idx_grupo[0], idx_grupo[-1]

        # onset: retroceder mientras la señal siga bajando (máx. 10 s)
        idx_onset = ini
        lim_o = max(0, ini - int(ONSET_MAX_RETROCESO_S * fs))
        while idx_onset > lim_o and senal[idx_onset - 1] <= senal[idx_onset]:
            idx_onset -= 1

        # pico: máximo del grupo, y seguir mientras la señal siga subiendo
        idx_pico = ini + np.argmax(senal[ini:fin + 1])
        lim_p = min(n - 1, idx_pico + int(SCR_RISE_MAX * fs))
        while idx_pico < lim_p and senal[idx_pico + 1] > senal[idx_pico]:
            idx_pico += 1

        rise = (idx_pico - idx_onset) / fs
        amp = senal[idx_pico] - senal[idx_onset]

        # half-recovery: primer instante en que cae a la mitad de la amplitud
        recuperacion = np.nan
        if amp > 0:
            objetivo = senal[idx_onset] + 0.5 * amp
            fin_busq = min(n, idx_pico + int(SCR_REC_MAX * fs) + 1)
            bajo = np.flatnonzero(senal[idx_pico:fin_busq] <= objetivo)
            if len(bajo):
                recuperacion = bajo[0] / fs

        es_scr_valido = (amp > 0 and SCR_RISE_MIN <= rise <= SCR_RISE_MAX
                         and SCR_REC_MIN <= recuperacion <= SCR_REC_MAX)
        if not es_scr_valido:
            artefactos[idx_grupo] = True
    return artefactos


def preprocesar_pecho(eda_pecho):
    """Señal cruda -> LP 5 Hz -> 20 Hz -> MAD sobre gradiente -> validación morfológica -> interpolación -> cvxEDA."""
    filtrada = filtro_pasabajos(np.asarray(eda_pecho, dtype=float).flatten(), FS_CHEST_RAW)
    eda_ds = downsamplear_pecho(filtrada)
    candidatos = candidatos_mad_gradiente(eda_ds, FS_CHEST)
    artefactos = validacion_morfologica_scr(eda_ds, candidatos, FS_CHEST)
    senal = interpolar_gaps(eda_ds, artefactos)
    tonico, fasico = separar_tonico_fasico_cvxeda(senal, FS_CHEST)
    return {'senal': senal, 'tonico': tonico, 'fasico': fasico, 'fs': FS_CHEST, 'mascara': artefactos}


# =============================================================================
# 4. MUÑECA
# =============================================================================

def alinear_acc_a_eda(acc):
    """Magnitud del ACC (32 Hz) llevada a la fs de EDA (4 Hz) con max-pooling: si hubo movimiento
    en cualquier muestra del bloque, se conserva."""
    mag = np.sqrt((np.asarray(acc, dtype=float) ** 2).sum(axis=1))
    factor = FS_ACC_WRIST // FS_WRIST
    n_bloques = len(mag) // factor
    return mag[:n_bloques * factor].reshape(n_bloques, factor).max(axis=1)


def detectar_movimiento_acc(mag_acc, umbral):
    """[DECISIÓN] Muestras EDA contaminadas por movimiento (magnitud ACC > umbral), dilatadas hacia
    adelante y atrás ACC_DILATACION_S segundos."""
    mascara = mag_acc > umbral
    n_dilat = int(ACC_DILATACION_S * FS_WRIST)
    if n_dilat > 0:
        mascara = binary_dilation(mascara, iterations=n_dilat)
    return mascara


def detectar_glitches_eda(senal):
    """[DECISIÓN] Glitches electrónicos que el ACC no ve: |gradiente| > mediana + k*MAD*1.4826 (k=15)."""
    grad = np.abs(np.diff(senal, prepend=senal[0]))
    mediana = np.median(grad)
    mad = np.median(np.abs(grad - mediana))
    mascara = grad > mediana + SPIKE_K * mad * 1.4826
    n_dilat = int(SPIKE_DILATACION_S * FS_WRIST)
    if n_dilat > 0:
        mascara = binary_dilation(mascara, iterations=n_dilat)
    return mascara


def preprocesar_muneca(eda_muneca, acc_muneca):
    """Señal cruda -> máscara ACC -> glitches -> interpolación -> cvxEDA.
    (El pasa-bajos de 5 Hz no aplica a 4 Hz: 5 Hz >= Nyquist.)"""
    eda = np.asarray(eda_muneca, dtype=float).flatten()
    mag = alinear_acc_a_eda(acc_muneca)
    n = min(len(eda), len(mag))                        # pueden diferir ±1 muestra por borde
    eda, mag = eda[:n], mag[:n]

    acc_p50 = float(np.median(mag))
    mascara_mov = detectar_movimiento_acc(mag, umbral=acc_p50 + ACC_OFFSET_FIJO)   # [DECISIÓN] mediana + 16
    mascara_glitch = detectar_glitches_eda(eda)
    mascara = mascara_mov | mascara_glitch

    senal = interpolar_gaps(eda, mascara)
    tonico, fasico = separar_tonico_fasico_cvxeda(senal, FS_WRIST)
    return {'senal': senal, 'tonico': tonico, 'fasico': fasico, 'fs': FS_WRIST, 'mascara': mascara,
            'pct_mov': 100 * float(mascara_mov.mean()), 'pct_glitch': 100 * float(mascara_glitch.mean()),
            'acc_p50': acc_p50}


# =============================================================================
# 5. VENTANEO Y ETIQUETADO
# =============================================================================

def calc_vent_indices(total_muestras, fs):
    """[PAPER] Ventanas de 60 s con paso de 0.25 s. El índice k de cada ventana es su t_inicio / 0.25."""
    n = max(int(np.floor((total_muestras / fs - WINDOW_S) / STEP_S)) + 1, 0)
    ini = (np.arange(n) * STEP_S * fs).astype(int)
    return ini, ini + int(WINDOW_S * fs)


def etiquetar_ventanas(ini, fin, fs, labels):
    """Clase dominante y pureza de cada ventana (las etiquetas están a 700 Hz)."""
    factor = FS_LABEL / fs
    i0 = np.clip(np.round(ini * factor).astype(int), 0, len(labels))
    i1 = np.clip(np.round(fin * factor).astype(int), 0, len(labels))
    conteos = np.zeros((len(ini), 8), dtype=np.int64)
    for c in range(8):
        acum = np.concatenate(([0], np.cumsum(labels == c)))
        conteos[:, c] = acum[i1] - acum[i0]
    dominante = conteos.argmax(axis=1)
    pureza = conteos.max(axis=1) / np.maximum(conteos.sum(axis=1), 1)
    return dominante, pureza


def fraccion_artefactos(ini, fin, mascara):
    acum = np.concatenate(([0], np.cumsum(np.asarray(mascara, dtype=bool))))
    return (acum[fin] - acum[ini]) / (fin - ini)


def evaluar_ventanas(prep, labels):
    """Ventanas de una rama y sus criterios de validez (etiqueta/pureza y fracción de artefactos)."""
    fs = prep['fs']
    ini, fin = calc_vent_indices(len(prep['senal']), fs)
    dominante, pureza = etiquetar_ventanas(ini, fin, fs, labels)
    frac_art = fraccion_artefactos(ini, fin, prep['mascara'])
    return {'ini': ini, 'fin': fin, 'dominante': dominante, 'pureza': pureza,
            'ok_etiqueta': np.isin(dominante, CLASES_VALIDAS) & (pureza >= PUREZA_MIN),
            'ok_art': frac_art <= MAX_FRAC_ART}


# =============================================================================
# 6. EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================

def extraer_features(prep, ini):
    """Características SCL/SCR/espectral-bajo para las ventanas que empiezan en `ini` (muestras).
    SCL = componente tónico, SCR = componente fásico; `senal` es la misma que se pasó a cvxEDA."""
    fs = prep['fs']
    fasico, tonico, senal = prep['fasico'], prep['tonico'], prep['senal']
    n_win = int(WINDOW_S * fs)
    fin = ini + n_win

    # picos SCR: una sola detección sobre la señal continua (sin efecto de borde de ventana)
    picos, props = find_peaks(fasico, prominence=SCR_PROM_MIN, distance=max(1, int(round(SCR_DIST_MIN_S * fs))),
                              wlen=n_win)
    prom = props['prominences']
    a_all, b_all = np.searchsorted(picos, ini), np.searchsorted(picos, fin)

    tc = np.arange(n_win) / fs
    tc = tc - tc.mean()
    den = float(tc @ tc)
    nperseg = min(n_win, int(WELCH_SEG_S * fs))

    X = np.zeros((len(ini), 11))
    for w, (i, j) in enumerate(zip(ini, fin)):
        f, t, s = fasico[i:j], tonico[i:j], senal[i:j]
        a, b = a_all[w], b_all[w]
        n_p, amp = b - a, prom[a:b]
        d = np.diff(t)
        fr, pxx = welch(s, fs=fs, nperseg=nperseg)
        low = fr <= SPECTRAL_LOW_HZ
        X[w] = [t.mean(), t.std(), float(tc @ t) / den, d.std() if len(d) > 1 else 0.0,   # SCL: media, std, pendiente, variación
                f.mean(), f.std(), f.min(),                                                # SCR: media, std, mínimo
                n_p, amp.mean() if n_p else 0.0, amp.sum() if n_p else 0.0,                # picos: nº, amp. media, amp. suma
                _trapz(pxx[low], fr[low])]                                                 # potencia espectral baja
    cols = ['scl_mean', 'scl_std', 'scl_slope', 'scl_variation', 'scr_mean', 'scr_std', 'scr_min',
            'scr_peak_count', 'scr_peak_mean_amp', 'scr_peak_sum_amp', 'eda_spectral_power_low']
    return pd.DataFrame(X, columns=cols)


def _bandas_strz(segs, fs):
    """[PAPER] Stržinar et al. (2023): z-score por segmento, espectro de amplitud (DFT) y, en las bandas
    B1 [0,1) Hz y B2 [1,2] Hz, las tres magnitudes usadas: std de B1, media de B2 y std de B2 (ddof=1)."""
    sd = segs.std(axis=1, keepdims=True)
    z = (segs - segs.mean(axis=1, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0)
    mag = np.abs(np.fft.rfft(z, axis=1))
    fr = np.fft.rfftfreq(segs.shape[1], d=1.0 / fs)
    b1 = mag[:, fr < STRZ_B1_MAX_HZ]
    b2 = mag[:, (fr >= STRZ_B2_HZ[0]) & (fr <= STRZ_B2_HZ[1])]
    return b1.std(axis=1, ddof=1), b2.mean(axis=1), b2.std(axis=1, ddof=1)


def extraer_features_strz(senal, fs, ini):
    """[PAPER] 4 características strz_* de muñeca: 7 segmentos de 12 s (paso 7 s) anclados al inicio de
    cada ventana de 60 s, sobre la señal interpolada. _mm = media y _mv = varianza (ddof=0) entre segmentos."""
    senal = np.asarray(senal, dtype=float)
    tam_seg, paso = int(round(STRZ_SEG_S * fs)), int(round(STRZ_PASO_S * fs))
    n_win = int(WINDOW_S * fs)
    n_seg = (n_win - tam_seg) // paso + 1
    assert n_seg == STRZ_N_SEG, f'se esperaban {STRZ_N_SEG} segmentos por ventana, hay {n_seg}'

    inicios = ini[:, None] + paso * np.arange(n_seg)[None, :]            # (ventanas, segmentos)
    unicos, inverso = np.unique(inicios, return_inverse=True)             # cada segmento se calcula una vez
    segs = sliding_window_view(senal, tam_seg)[unicos]
    b1_std, b2_mean, b2_std = (v[inverso.reshape(-1)].reshape(inicios.shape) for v in _bandas_strz(segs, fs))
    return pd.DataFrame({'strz_b2_std_mm': b2_std.mean(axis=1),
                         'strz_b2_mean_mv': b2_mean.var(axis=1),
                         'strz_b1_std_mm': b1_std.mean(axis=1),
                         'strz_b2_mean_mm': b2_mean.mean(axis=1)})


# =============================================================================
# 7. PROCESAMIENTO POR SUJETO
# =============================================================================

def procesar_sujeto(datos, id_sujeto):
    """Devuelve (DataFrame con las 26 columnas, dict de QC) para un sujeto."""
    labels = np.asarray(datos['label']).astype(np.int64).ravel()

    pecho = preprocesar_pecho(datos['signal']['chest']['EDA'])
    muneca = preprocesar_muneca(datos['signal']['wrist']['EDA'], datos['signal']['wrist']['ACC'])
    vp, vm = evaluar_ventanas(pecho, labels), evaluar_ventanas(muneca, labels)

    # Inner join por t_inicio: ambas ramas usan la misma rejilla (índice k = t_inicio / 0.25 s)
    n = min(len(vp['ini']), len(vm['ini']))
    ok_etq = vp['ok_etiqueta'][:n] & vm['ok_etiqueta'][:n]
    ok_art_p, ok_art_m = vp['ok_art'][:n], vm['ok_art'][:n]
    sel = np.flatnonzero(ok_etq & ok_art_p & ok_art_m)
    assert (vp['dominante'][sel] == vm['dominante'][sel]).all(), 'Las etiquetas de pecho y muñeca no coinciden'

    qc = {'sujeto': id_sujeto, 'ventanas': n, 'sin_par': abs(len(vp['ini']) - len(vm['ini'])),
          'desc_etiqueta': int((~ok_etq).sum()),
          'desc_art_pecho': int((ok_etq & ~ok_art_p).sum()),
          'desc_art_muneca': int((ok_etq & ok_art_p & ~ok_art_m).sum()),
          'finales': int(len(sel)),
          'pct_mask_pecho': 100 * float(pecho['mascara'].mean()),
          'pct_mask_muneca': 100 * float(muneca['mascara'].mean()),
          'pct_mov': muneca['pct_mov'], 'pct_glitch': muneca['pct_glitch'], 'acc_p50': muneca['acc_p50'],
          'clases': {int(c): int(k) for c, k in zip(*np.unique(vp['dominante'][sel], return_counts=True))}}
    if len(sel) == 0:
        return pd.DataFrame(columns=COLUMNAS_FINALES), qc

    ini_p, ini_m = vp['ini'][sel], vm['ini'][sel]
    meta = pd.DataFrame({'sujeto': id_sujeto, 't_inicio': sel * STEP_S,
                         'etiqueta': vp['dominante'][sel], 'pureza': vp['pureza'][sel]})
    f_p = extraer_features(pecho, ini_p)[BASE_PECHO].add_prefix('chest_')
    f_m = pd.concat([extraer_features(muneca, ini_m),
                     extraer_features_strz(muneca['senal'], FS_WRIST, ini_m)], axis=1)[BASE_MUNECA].add_prefix('wrist_')
    df = pd.concat([meta, f_p, f_m], axis=1)

    assert list(df.columns) == COLUMNAS_FINALES, 'El esquema de columnas no coincide con las 26 esperadas'
    assert np.isfinite(df.drop(columns='sujeto').to_numpy(dtype=float)).all(), f'{id_sujeto}: NaN/inf en características'
    return df, qc


def _tarea(ruta, id_sujeto):
    """Unidad de trabajo (un sujeto): carga -> procesa -> libera memoria. Devuelve solo resultados pequeños."""
    datos = cargar_sujeto(ruta, id_sujeto)
    resultado = procesar_sujeto(datos, id_sujeto)
    del datos
    gc.collect()
    return resultado


# =============================================================================
# 8. ORQUESTACIÓN, QC Y EJECUCIÓN
# =============================================================================

def imprimir_qc(qcs):
    """Registro de control de calidad (ventanas descartadas) en la terminal. Descarte secuencial:
    etiqueta/pureza -> artefactos de pecho -> artefactos de muñeca."""
    enc = (f"{'Sujeto':<7}{'Ventanas':>10}{'D.etiq/pur':>12}{'D.art pecho':>13}{'D.art mun.':>12}"
           f"{'Sin par':>9}{'Finales':>9}{'%mask P':>9}{'%mask M':>9}{'ACC p50':>9}")
    print('\n' + '=' * len(enc) + '\n QC: ventanas descartadas por sujeto\n' + '=' * len(enc) + '\n' + enc)
    tot = {k: 0 for k in ('ventanas', 'desc_etiqueta', 'desc_art_pecho', 'desc_art_muneca', 'sin_par', 'finales')}
    clases = {}
    for q in qcs:
        print(f"{q['sujeto']:<7}{q['ventanas']:>10}{q['desc_etiqueta']:>12}{q['desc_art_pecho']:>13}"
              f"{q['desc_art_muneca']:>12}{q['sin_par']:>9}{q['finales']:>9}{q['pct_mask_pecho']:>9.2f}"
              f"{q['pct_mask_muneca']:>9.2f}{q['acc_p50']:>9.1f}")
        for k in tot:
            tot[k] += q[k]
        for c, v in q['clases'].items():
            clases[c] = clases.get(c, 0) + v
    print('-' * len(enc))
    print(f"{'TOTAL':<7}{tot['ventanas']:>10}{tot['desc_etiqueta']:>12}{tot['desc_art_pecho']:>13}"
          f"{tot['desc_art_muneca']:>12}{tot['sin_par']:>9}{tot['finales']:>9}")
    print(f"Ventanas finales por clase: {dict(sorted(clases.items()))}")
    for q in qcs:
        if not (ACC_P50_RANGO[0] <= q['acc_p50'] <= ACC_P50_RANGO[1]):
            print(f"[AVISO] {q['sujeto']}: mediana ACC = {q['acc_p50']:.1f} (reposo esperado ~64); "
                  f"revisa el umbral ACC = mediana + {ACC_OFFSET_FIJO}")
        if q['finales'] == 0:
            print(f"[AVISO] {q['sujeto']}: ninguna ventana válida tras los filtros")


def procesar_todos(ruta, salida, procesos=1):
    """Procesa los 15 sujetos y escribe {salida}/EDA_v3_todos.csv. Si algún sujeto falla se aborta
    (SystemExit 1) sin escribir el CSV, para no producir un dataset incompleto."""
    resultados, errores = {}, {}

    if procesos <= 1:                                   # secuencial: falla en el primer error (útil para debug)
        for s in SUJETOS_VALIDOS:
            print(f'Procesando {s}...', flush=True)
            resultados[s] = _tarea(ruta, s)
    else:
        with ProcessPoolExecutor(max_workers=min(procesos, len(SUJETOS_VALIDOS))) as ex:
            futuros = {ex.submit(_tarea, ruta, s): s for s in SUJETOS_VALIDOS}
            for fut in as_completed(futuros):
                s = futuros[fut]
                try:
                    resultados[s] = fut.result()
                    print(f'{s} procesado ({len(resultados)}/{len(SUJETOS_VALIDOS)})', flush=True)
                except Exception:
                    errores[s] = traceback.format_exc()
                    for f in futuros:                    # no lanzar más sujetos
                        f.cancel()
        if errores:
            for s, tb in errores.items():
                print(f'[ERROR] {s} falló:\n{tb}', file=sys.stderr)
            print(f'\nAbortado sin escribir {NOMBRE_SALIDA}. Sujetos con error: {sorted(errores)}', file=sys.stderr)
            raise SystemExit(1)

    imprimir_qc([resultados[s][1] for s in SUJETOS_VALIDOS])
    df = pd.concat([resultados[s][0] for s in SUJETOS_VALIDOS], ignore_index=True)
    os.makedirs(salida, exist_ok=True)
    ruta_csv = os.path.join(salida, NOMBRE_SALIDA)
    df.to_csv(ruta_csv, index=False)
    print(f'\n{NOMBRE_SALIDA}: {df.shape[0]} filas x {df.shape[1]} columnas -> {ruta_csv}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Pipeline EDA v3 (WESAD): genera EDA_v3_todos.csv')
    ap.add_argument('--ruta', default='../data/WESAD', help='carpeta con S2..S17')
    ap.add_argument('--salida', default='./resultados_eda', help='carpeta de salida')
    ap.add_argument('--procesos', type=int, default=1,
                    help='nº de procesos; 1 (por defecto) = secuencial, >1 = paralelo por sujeto')
    args = ap.parse_args()
    if args.procesos < 1:
        ap.error('--procesos debe ser >= 1')
    procesar_todos(args.ruta, args.salida, args.procesos)
