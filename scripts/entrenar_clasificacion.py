"""Entrena y compara los clasificadores de nivel de congestion (varios minutos).

Uso:  python scripts/entrenar_clasificacion.py
Salidas: data/resultados_clasificacion.csv, data/resultados_clasificacion.json,
         data/modelo_clasificacion.joblib, output/clf_*.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml import clasificacion as C
from src.ml import dataset, regresion

OUT = ROOT / "output"
DATA = ROOT / "data"
BASE = "Línea base (clase más frecuente)"


def main() -> None:
    X, y, grupos, d = C.preparar(dataset.cargar())
    dist = y.value_counts(normalize=True).reindex(C.CLASES).round(3).to_dict()
    print(f"{len(d)} filas, {grupos.nunique()} tramos, clases: {dist}")

    folds, oof = C.evaluar(X, y, grupos)
    resumen = (folds.groupby("modelo", sort=False)[["exactitud", "f1_macro"]]
               .agg(["mean", "std"]).round(3))
    resumen.columns = [f"{a}_{b}" for a, b in resumen.columns]
    resumen = resumen.reset_index()
    print(resumen.to_string(index=False))
    folds.to_csv(DATA / "resultados_clasificacion.csv", index=False, encoding="utf-8")

    mejor = max((n for n in oof if n != BASE), key=lambda n: C.metricas(y, oof[n])["f1_macro"])
    print("mejor modelo (F1 macro):", mejor)
    pred = oof[mejor]
    informe = classification_report(y, pred, labels=C.CLASES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=C.CLASES)

    final = C.ajustar_final(X, y, grupos, mejor)
    joblib.dump({"modelo": final, "nombre": mejor, "columnas": list(X.columns), "clases": C.CLASES},
                DATA / "modelo_clasificacion.joblib")
    imp = permutation_importance(final, X, y, n_repeats=3, random_state=0, scoring="f1_macro", n_jobs=-1)
    importancias = pd.Series(imp.importances_mean, index=X.columns).sort_values()

    (DATA / "resultados_clasificacion.json").write_text(json.dumps({
        "filas": len(d), "tramos": int(grupos.nunique()), "validacion": "GroupKFold(5) por tramo",
        "umbrales_m": {"media": C.UMBRAL_MEDIA, "alta": C.UMBRAL_ALTA},
        "distribucion_clases": dist,
        "media_y_std_por_pliegue": resumen.to_dict(orient="records"),
        "mejor": mejor,
        "informe_mejor": {k: {m: round(v, 3) for m, v in x.items()} if isinstance(x, dict) else round(x, 3)
                          for k, x in informe.items()},
        "matriz_confusion_filas_real_columnas_predicho": {"clases": C.CLASES, "matriz": cm.tolist()},
        "hiperparametros_por_pliegue": folds[folds.hiperparametros != "{}"]
            .groupby("modelo")["hiperparametros"].apply(list).to_dict(),
        "importancia_permutacion_f1": importancias.round(3).to_dict(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- graficas ----
    por_hora = pd.crosstab(d["hora"], d["nivel"], normalize="index")[C.CLASES]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    por_hora.plot.bar(stacked=True, ax=ax, color=["#22c55e", "#f59e0b", "#dc2626"], width=.85)
    ax.set(xlabel="hora del día", ylabel="proporción de filas", title="Nivel de congestión por hora")
    ax.legend(title="nivel", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "clf_nivel_por_hora.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(resumen["modelo"], resumen["f1_macro_mean"], yerr=resumen["f1_macro_std"], capsize=4, color="#2563eb")
    ax.set(ylabel="F1 macro", ylim=(0, 1), title="F1 macro por modelo (validación por tramo)")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout(); fig.savefig(OUT / "clf_comparacion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set(title=f"{mejor}: matriz de confusión", xlabel="predicho", ylabel="real")
    fig.tight_layout(); fig.savefig(OUT / "clf_confusion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(importancias.index, importancias.values, color="#0ea5e9")
    ax.set(xlabel="disminución del F1 macro al permutar la variable", title=f"Importancia de las variables ({mejor})")
    fig.tight_layout(); fig.savefig(OUT / "clf_importancia.png", dpi=130); plt.close(fig)

    Xd = pd.get_dummies(X, columns=regresion.CATEGORICAS, dtype=float)   # unidades originales
    arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(Xd, y)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(arbol, feature_names=list(Xd.columns), class_names=list(arbol.classes_), filled=True,
              rounded=True, impurity=False, proportion=True, precision=2, fontsize=8, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "clf_arbol.png", dpi=130); plt.close(fig)
    print("importancia:", importancias.round(3).to_dict())
    print(json.dumps(informe, indent=1, default=float)[:900])


if __name__ == "__main__":
    main()
