"""
loso_final.py  -  LOSO definitivo de EDA (WESAD)

Evalúa las características de una tabla (p. ej. EDA_v3_todos.csv) con TODOS los parámetros ya
decididos; desde la terminal solo se elige la tabla, la carpeta de salida, los núcleos y las semillas.

  Normalización : GLOBAL. En cada pliegue se podan las columnas casi idénticas (|r| > 0.98) y se
                  escala con StandardScaler, ajustando AMBAS cosas solo con el entrenamiento.
  Validación    : LOSO (15 pliegues). Se entrena con 1 de cada 80 ventanas (solo del entrenamiento)
                  y se prueba con TODAS las ventanas de la persona apartada.
  Pesos         : 'balanceado' (peso de clase = n / (k * n_clase)); LDA con previas iguales.
  Modelos       : RandomForest (100 árboles, entropy, min_samples_split=20), AdaBoost (100 árboles
                  entropy, min_samples_split=20) y LDA.
  Problemas     : binario (estrés vs. baseline + diversión) y 3clases (baseline, estrés, diversión).
  Conjuntos     : pecho (columnas chest_), muneca (wrist_) y ambos.
  Semillas      : cada semilla cambia el azar de los modelos Y cuál 1/80 de las ventanas de
                  entrenamiento se usa (semilla 0 = posiciones 0, 80, 160...). Se reporta media y
                  desviación entre semillas. NO captura la variación entre personas.

Salidas (en --salida):
  resumen_loso.csv            media y desviación entre semillas, por problema / conjunto / modelo
  resumen_por_semilla.csv     lo mismo, una fila por semilla
  metricas_por_sujeto.csv     accuracy y F1 de cada persona (media entre semillas)
  matrices_confusion.csv      conteos medios y % por fila (real vs. predicho)
  poda_correlacion.csv        en cuántos de los 15 pliegues sobrevive cada columna
  figuras/matriz_*.png        matrices de confusión (una figura por problema y conjunto)
  predicciones/*.csv.gz       probabilidad fuera de muestra de cada ventana (media entre semillas)
  config_usada.json           parámetros y versiones, para reproducir

Uso:
    python loso_final.py --tabla EDA_v3_todos.csv --salida resultados/loso_final --procesos 4
Opcionales: --semillas 5 | --paso-train 80 | --sin-figuras | --sin-predicciones | --rehacer
Se escribe al terminar cada (problema, conjunto); si repites el comando, lo ya hecho se salta.
Dependencias: numpy, pandas, scikit-learn, joblib, matplotlib (solo para las figuras).
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42
MODELOS = ("RandomForest", "AdaBoost", "LDA")
PROBLEMAS = ("binario", "3clases")
CONJUNTOS = {"pecho": ("chest_",), "muneca": ("wrist_",), "ambos": ("chest_", "wrist_")}
UMBRAL_CORR = 0.98

# Columnas que NUNCA son predictoras (identificadores, tiempo, calidad) y duplicadas por construcción.
PALABRAS_EXCLUIDAS = ("frac_art", "movement_ratio", "signal_snr", "t_inicio", "t_centro", "pureza")
SUFIJOS_DUPLICADOS = ("scr_peaks_per_min", "scr_auc")


# ---------------------------------------------------------------------------
# Datos y problema
# ---------------------------------------------------------------------------
def cargar_tabla(ruta):
    df = pd.read_csv(ruta)
    faltan = [c for c in ("sujeto", "t_inicio", "etiqueta") if c not in df.columns]
    if faltan:
        raise SystemExit(f"La tabla no tiene las columnas {faltan}")
    df = df[df["etiqueta"].isin([1, 2, 3])].copy()
    df["sujeto"] = df["sujeto"].astype(str)
    return df.reset_index(drop=True)


def columnas_predictoras(df, conjunto):
    cols = []
    for c in df.columns:
        if not c.startswith(CONJUNTOS[conjunto]):
            continue
        if any(p in c for p in PALABRAS_EXCLUIDAS) or c.endswith(SUFIJOS_DUPLICADOS):
            continue
        cols.append(c)
    return cols


def tiene_columnas(df, conjunto):
    if conjunto == "ambos":
        return bool(columnas_predictoras(df, "pecho")) and bool(columnas_predictoras(df, "muneca"))
    return bool(columnas_predictoras(df, conjunto))


def definir_problema(d, problema):
    if problema == "3clases":
        return d["etiqueta"].to_numpy(), ["Baseline", "Estres", "Diversion"], [1, 2, 3]
    return (d["etiqueta"].to_numpy() == 2).astype(int), ["No_estres", "Estres"], [0, 1]


# ---------------------------------------------------------------------------
# Preparación (solo con el entrenamiento) y modelos
# ---------------------------------------------------------------------------
def quitar_correlacionadas(X_train, umbral=UMBRAL_CORR):
    """Decide qué columnas conservar usando SOLO el entrenamiento del pliegue."""
    corr = np.nan_to_num(np.abs(np.corrcoef(X_train, rowvar=False)))
    conservar = np.ones(corr.shape[0], dtype=bool)
    for i in range(corr.shape[0]):
        if not conservar[i]:
            continue
        for j in range(i + 1, corr.shape[0]):
            if conservar[j] and corr[i, j] > umbral:
                conservar[j] = False
    return conservar


def pesos_clase(y_tr, etiquetas):
    n, k = len(y_tr), len(etiquetas)
    cuentas = np.array([(y_tr == c).sum() for c in etiquetas], dtype=float)
    return n / (k * cuentas)


def entrenar_proba(nombre, A, y_tr, B, etiquetas, semilla, n_jobs):
    w = pesos_clase(y_tr, etiquetas)
    if nombre == "RandomForest":
        modelo = RandomForestClassifier(
            n_estimators=100, criterion="entropy", min_samples_split=20,
            class_weight=dict(zip(etiquetas, w)), n_jobs=n_jobs, random_state=semilla)
        modelo.fit(A, y_tr)
    elif nombre == "AdaBoost":
        modelo = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(min_samples_split=20, criterion="entropy"),
            n_estimators=100, random_state=semilla)
        modelo.fit(A, y_tr, sample_weight=w[np.searchsorted(etiquetas, y_tr)])
    else:
        modelo = LinearDiscriminantAnalysis(priors=np.full(len(etiquetas), 1.0 / len(etiquetas)))
        modelo.fit(A, y_tr)
    return modelo.predict_proba(B).astype(np.float32)      # columnas en el orden de 'etiquetas'


def correr_pliegue(X, y, orden_t, tr, te, etiquetas, s_idx, n_sem, paso, n_jobs_rf):
    """Un pliegue de una semilla: entrena con 1/paso del entrenamiento y predice TODA la persona de prueba."""
    offset = (s_idx * max(paso // n_sem, 1)) % paso
    tr = tr[orden_t[tr] % paso == offset]                      # submuestreo SOLO del entrenamiento
    conservar = quitar_correlacionadas(X[tr])                  # decidido con el entrenamiento
    esc = StandardScaler().fit(X[tr][:, conservar])            # ajustado con el entrenamiento
    A = esc.transform(X[tr][:, conservar])
    B = esc.transform(X[te][:, conservar])
    P = {m: entrenar_proba(m, A, y[tr], B, etiquetas, SEMILLA + s_idx, n_jobs_rf) for m in MODELOS}
    return te, P, conservar


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def auc_roc(y, P, etiquetas):
    if len(etiquetas) == 2:
        return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", average="macro", labels=etiquetas)


def auc_pr(y, P, etiquetas):
    if len(etiquetas) == 2:
        return average_precision_score(y, P[:, 1])
    return float(np.mean([average_precision_score((y == c).astype(int), P[:, i])
                          for i, c in enumerate(etiquetas)]))


def metricas(y, P, sujetos, etiquetas, nombres):
    P = P.astype(np.float64)
    pred = np.asarray(etiquetas)[P.argmax(axis=1)]
    prec, rec, f1c, _ = precision_recall_fscore_support(y, pred, labels=etiquetas, zero_division=0)
    por_suj = []
    for s in np.unique(sujetos):
        m = sujetos == s
        por_suj.append({"sujeto": s, "accuracy": accuracy_score(y[m], pred[m]),
                        "f1_macro": f1_score(y[m], pred[m], average="macro", labels=etiquetas,
                                             zero_division=0)})
    fila = {"accuracy": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", labels=etiquetas, zero_division=0),
            "f1_medio_sujetos": float(np.mean([s["f1_macro"] for s in por_suj])),
            "f1_std_sujetos": float(np.std([s["f1_macro"] for s in por_suj], ddof=1)),
            "auc_roc": auc_roc(y, P, etiquetas), "auc_pr": auc_pr(y, P, etiquetas)}
    for i, n in enumerate(nombres):
        fila[f"precision_{n}"], fila[f"recall_{n}"], fila[f"f1_{n}"] = prec[i], rec[i], f1c[i]
    return fila, por_suj, pred


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
def anexar(df, ruta):
    """Agrega filas a un CSV alineando las columnas por NOMBRE (binario y 3 clases difieren)."""
    if not os.path.exists(ruta):
        df.to_csv(ruta, index=False)
        return
    previo = pd.read_csv(ruta)
    todo = pd.concat([previo, df], ignore_index=True, sort=False)
    cols = list(previo.columns) + [c for c in df.columns if c not in previo.columns]
    todo[cols].to_csv(ruta, index=False)


def ya_hecho(carpeta, problema, conjunto):
    ruta = os.path.join(carpeta, "resumen_por_semilla.csv")
    if not os.path.exists(ruta):
        return False
    r = pd.read_csv(ruta, usecols=["problema", "conjunto"])
    return bool(((r.problema == problema) & (r.conjunto == conjunto)).any())


def graficar_matrices(cm_medias, nombres, problema, conjunto, f1s, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(MODELOS), figsize=(4.6 * len(MODELOS), 4.4))
    for ax, m in zip(ejes, MODELOS):
        cm = cm_medias[m]
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]:,.0f})", ha="center", va="center",
                        color="white" if pct[i, j] > 55 else "black", fontsize=9)
        ax.set_xticks(range(len(nombres)))
        ax.set_yticks(range(len(nombres)))
        ax.set_xticklabels(nombres)
        ax.set_yticklabels(nombres)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{m}\nF1 macro = {f1s[m]:.3f}", fontsize=10)
    fig.suptitle(f"Matriz de confusión (LOSO, normalización global) | {problema} | {conjunto}\n"
                 "cada fila suma 100%; entre paréntesis, ventanas (media entre semillas)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Un experimento = (problema, conjunto)
# ---------------------------------------------------------------------------
def ejecutar(df, problema, conjunto, args, carpeta):
    cols = columnas_predictoras(df, conjunto)
    d = df.dropna(subset=cols).reset_index(drop=True)
    y, nombres, etiquetas = definir_problema(d, problema)
    X = d[cols].to_numpy(dtype=float)
    grupos = d["sujeto"].to_numpy()
    orden_t = d.groupby("sujeto").cumcount().to_numpy()
    pliegues = list(LeaveOneGroupOut().split(X, y, grupos))
    for tr, te in pliegues:
        assert grupos[te][0] not in set(grupos[tr]), "¡la persona de prueba está en el entrenamiento!"
    n_sem, n_pl = args.semillas, len(pliegues)
    print(f"\n=== {problema} | {conjunto} | {len(cols)} características | {n_pl} pliegues x "
          f"{n_sem} semillas | {len(d)} ventanas ===", flush=True)

    n_jobs_rf = 1 if args.procesos > 1 else -1
    t0 = time.time()
    res = Parallel(n_jobs=args.procesos, verbose=5)(
        delayed(correr_pliegue)(X, y, orden_t, tr, te, etiquetas, s, n_sem, args.paso_train, n_jobs_rf)
        for s in range(n_sem) for tr, te in pliegues)
    print(f"  entrenamiento y predicción: {time.time() - t0:.0f} s", flush=True)

    idx = np.concatenate([r[1 - 1] for r in res[:n_pl]])        # mismo orden de pruebas en todas las semillas
    y_all, suj_all = y[idx], grupos[idx]
    conservadas = np.sum([r[2] for r in res[:n_pl]], axis=0)     # semilla 0: pliegues donde sobrevive cada columna

    filas_sem, por_suj_todas, P_medias = [], [], {m: 0.0 for m in MODELOS}
    cm_sum = {m: np.zeros((len(etiquetas), len(etiquetas))) for m in MODELOS}
    for s in range(n_sem):
        bloque = res[s * n_pl:(s + 1) * n_pl]
        for m in MODELOS:
            P = np.concatenate([b[1][m] for b in bloque])
            fila, por_suj, pred = metricas(y_all, P, suj_all, etiquetas, nombres)
            filas_sem.append({"problema": problema, "conjunto": conjunto, "n_columnas": len(cols),
                              "modelo": m, "semilla": s, "n_ventanas_prueba": len(y_all), **fila})
            por_suj_todas += [{"modelo": m, **p} for p in por_suj]
            cm_sum[m] += confusion_matrix(y_all, pred, labels=etiquetas)
            P_medias[m] = P_medias[m] + P.astype(np.float64) / n_sem
    cm_medias = {m: cm_sum[m] / n_sem for m in MODELOS}
    f1_medios = {m: float(np.mean([f["f1_macro"] for f in filas_sem if f["modelo"] == m]))
                 for m in MODELOS}

    # --- archivos ---
    anexar(pd.DataFrame(filas_sem), os.path.join(carpeta, "resumen_por_semilla.csv"))
    ps = (pd.DataFrame(por_suj_todas).groupby(["modelo", "sujeto"], as_index=False)
          [["accuracy", "f1_macro"]].mean())
    ps.insert(0, "conjunto", conjunto)
    ps.insert(0, "problema", problema)
    anexar(ps, os.path.join(carpeta, "metricas_por_sujeto.csv"))
    filas_cm = []
    for m in MODELOS:
        cm = cm_medias[m]
        for i, real in enumerate(nombres):
            for j, pred_n in enumerate(nombres):
                filas_cm.append({"problema": problema, "conjunto": conjunto, "modelo": m,
                                 "real": real, "predicho": pred_n, "conteo_medio": cm[i, j],
                                 "pct_fila": 100 * cm[i, j] / cm[i].sum()})
    anexar(pd.DataFrame(filas_cm), os.path.join(carpeta, "matrices_confusion.csv"))
    anexar(pd.DataFrame({"problema": problema, "conjunto": conjunto, "caracteristica": cols,
                         "pliegues_conservada": conservadas}),
           os.path.join(carpeta, "poda_correlacion.csv"))
    if not args.sin_figuras:
        os.makedirs(os.path.join(carpeta, "figuras"), exist_ok=True)
        graficar_matrices(cm_medias, nombres, problema, conjunto, f1_medios,
                          os.path.join(carpeta, "figuras", f"matriz_{problema}_{conjunto}.png"))
    if not args.sin_predicciones:
        os.makedirs(os.path.join(carpeta, "predicciones"), exist_ok=True)
        pr = pd.DataFrame({"sujeto": suj_all, "t_inicio": d["t_inicio"].to_numpy()[idx],
                           "etiqueta": d["etiqueta"].to_numpy()[idx]})
        for m in MODELOS:
            clases = range(len(etiquetas)) if len(etiquetas) > 2 else [1]
            for i in clases:
                pr[f"p_{nombres[i]}_{m}"] = P_medias[m][:, i]
        pr = pr.sort_values(["sujeto", "t_inicio"])
        pr.to_csv(os.path.join(carpeta, "predicciones", f"{problema}_{conjunto}.csv.gz"),
                  index=False, compression="gzip", float_format="%.4f")

    tabla = pd.DataFrame(filas_sem).groupby("modelo")[["accuracy", "f1_macro", "auc_roc"]].agg(["mean", "std"])
    print(tabla.round(3).to_string(), flush=True)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LOSO definitivo de EDA (normalización global, 3 modelos)")
    ap.add_argument("--tabla", required=True, help="p. ej. EDA_v3_todos.csv")
    ap.add_argument("--salida", default="resultados/loso_final")
    ap.add_argument("--procesos", type=int, default=os.cpu_count() or 1,
                    help="procesos en paralelo (por defecto, todos los núcleos)")
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--paso-train", type=int, default=80)
    ap.add_argument("--sin-figuras", action="store_true")
    ap.add_argument("--sin-predicciones", action="store_true")
    ap.add_argument("--rehacer", action="store_true", help="repite lo que ya está en la carpeta de salida")
    args = ap.parse_args()
    if args.procesos < 1 or args.semillas < 1:
        ap.error("--procesos y --semillas deben ser >= 1")

    os.makedirs(args.salida, exist_ok=True)
    df = cargar_tabla(args.tabla)
    with open(os.path.join(args.salida, "config_usada.json"), "w", encoding="utf-8") as f:
        json.dump({"tabla": os.path.basename(args.tabla), "normalizacion": "global",
                   "umbral_correlacion": UMBRAL_CORR, "paso_train": args.paso_train,
                   "semillas": args.semillas, "semilla_base": SEMILLA, "modelos": list(MODELOS),
                   "pesos": "balanceado", "validacion": "LOSO (LeaveOneGroupOut)",
                   "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "scikit-learn": sklearn.__version__}},
                  f, indent=2, ensure_ascii=False)

    hechos = 0
    for problema in PROBLEMAS:
        for conjunto in CONJUNTOS:
            if not tiene_columnas(df, conjunto):
                print(f"\n[omitido] conjunto '{conjunto}': la tabla no tiene esas columnas")
                continue
            if not args.rehacer and ya_hecho(args.salida, problema, conjunto):
                print(f"\n[saltado] {problema} | {conjunto}: ya está en resumen_por_semilla.csv")
                continue
            ejecutar(df, problema, conjunto, args, args.salida)
            hechos += 1

    ruta_sem = os.path.join(args.salida, "resumen_por_semilla.csv")
    if os.path.exists(ruta_sem):
        r = pd.read_csv(ruta_sem)
        agr = r.groupby(["problema", "conjunto", "modelo"], sort=False).agg(
            **{f"{c}_{e}": (c, e) for c in r.select_dtypes("number").columns
               if c not in ("semilla", "n_columnas", "n_ventanas_prueba") for e in ("mean", "std")}
        ).reset_index()
        agr.to_csv(os.path.join(args.salida, "resumen_loso.csv"), index=False)
        print("\nF1 macro (media ± desviación entre semillas):")
        for problema in PROBLEMAS:
            sub = agr[agr.problema == problema]
            if len(sub):
                sub = sub.assign(f1=sub.f1_macro_mean.round(3).astype(str) + " ± "
                                 + sub.f1_macro_std.round(3).astype(str))
                print(f"\n{problema}")
                print(sub.pivot(index="conjunto", columns="modelo", values="f1").to_string())
    if hechos == 0:
        print("\nNo se ejecutó ningún experimento nuevo.")
    print(f"\nResultados en {args.salida}/")


if __name__ == "__main__":
    main()
