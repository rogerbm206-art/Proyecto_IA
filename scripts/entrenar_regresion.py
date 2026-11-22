"""Entrena y compara los modelos de regresion del tiempo por tramo (~2-5 min).

Uso:  python scripts/entrenar_regresion.py
Salidas: data/resultados_regresion.csv, data/resultados_regresion.json,
         data/modelo_regresion.joblib, output/reg_*.png
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
from sklearn.tree import DecisionTreeRegressor, plot_tree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml import dataset, regresion

OUT = ROOT / "output"
DATA = ROOT / "data"
CATEGORICAS_ARBOL = regresion.CATEGORICAS


def main() -> None:
    df = dataset.cargar()
    X, y, grupos = regresion.preparar(df)
    print(f"{len(df)} filas, {grupos.nunique()} tramos")

    folds, oof = regresion.evaluar(X, y, grupos)
    resumen = (folds.groupby("modelo", sort=False)[["mae_s", "rmse_s", "r2"]]
               .agg(["mean", "std"]).round(3))
    resumen.columns = [f"{a}_{b}" for a, b in resumen.columns]
    resumen = resumen.reset_index()
    print(resumen.to_string(index=False))
    folds.to_csv(DATA / "resultados_regresion.csv", index=False, encoding="utf-8")

    # metricas globales fuera de muestra (todas las predicciones juntas)
    globales = {n: {k: round(float(v), 3) for k, v in regresion.metricas(y.to_numpy(), p).items()}
                for n, p in oof.items()}
    mejor = min((n for n in oof if n != "Línea base (media)"), key=lambda n: globales[n]["mae_s"])
    print("mejor modelo (MAE):", mejor)

    # modelo final con todos los datos
    final = regresion.ajustar_final(X, y, grupos, mejor)
    joblib.dump({"modelo": final, "nombre": mejor, "columnas": list(X.columns)},
                DATA / "modelo_regresion.joblib")

    # importancia por permutacion sobre las predicciones del mejor modelo ajustado
    imp = permutation_importance(final, X, y, n_repeats=5, random_state=0,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
    importancias = pd.Series(imp.importances_mean, index=X.columns).sort_values()

    (DATA / "resultados_regresion.json").write_text(json.dumps({
        "filas": len(df), "tramos": int(grupos.nunique()), "validacion": "GroupKFold(5) por tramo",
        "media_y_std_por_pliegue": resumen.to_dict(orient="records"),
        "globales_fuera_de_muestra": globales, "mejor": mejor,
        "hiperparametros_por_pliegue": folds[folds.hiperparametros != "{}"]
            .groupby("modelo")["hiperparametros"].apply(list).to_dict(),
        "importancia_permutacion_mae_s": importancias.round(3).to_dict(),
        "tiempo_medio_s": round(float(y.mean()), 2),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- graficas ----
    fig, ax = plt.subplots(figsize=(7, 3.8))
    modelos_ord = list(resumen["modelo"])
    ax.bar(modelos_ord, resumen["mae_s_mean"], yerr=resumen["mae_s_std"], color="#2563eb", capsize=4)
    ax.set(ylabel="MAE (s)", title="Error absoluto medio por modelo (validación por tramo)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / "reg_comparacion.png", dpi=130); plt.close(fig)

    p = oof[mejor]
    muestra = np.random.default_rng(0).choice(len(y), size=min(len(y), 8000), replace=False)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y.iloc[muestra], p[muestra], s=3, alpha=.25, color="#2563eb")
    lim = [0, float(max(y.max(), p.max()))]
    ax.plot(lim, lim, color="#dc2626", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="tiempo real programado (s)", ylabel="tiempo predicho (s)",
           title=f"{mejor}: predicción frente a valor real")
    fig.tight_layout(); fig.savefig(OUT / "reg_prediccion.png", dpi=130); plt.close(fig)

    err = pd.DataFrame({"hora": X["hora"], "abs": np.abs(y - p)}).groupby("hora")["abs"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(err.index, err.values, marker="o", ms=3, color="#2563eb")
    ax.set(xlabel="hora del día", ylabel="MAE (s)", title=f"Error de {mejor} por hora")
    fig.tight_layout(); fig.savefig(OUT / "reg_error_hora.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(importancias.index, importancias.values, color="#0ea5e9")
    ax.set(xlabel="aumento del MAE al permutar la variable (s)",
           title=f"Importancia de las variables ({mejor})")
    fig.tight_layout(); fig.savefig(OUT / "reg_importancia.png", dpi=130); plt.close(fig)

    # arbol pequeno e interpretable (profundidad 3) sobre todos los datos
    Xd = pd.get_dummies(X, columns=CATEGORICAS_ARBOL, dtype=float)   # unidades originales
    arbol = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0).fit(Xd, y)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plot_tree(arbol, feature_names=list(Xd.columns), filled=True, rounded=True,
              impurity=False, precision=1, fontsize=8, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "reg_arbol.png", dpi=130); plt.close(fig)
    print("importancia:", importancias.round(2).to_dict())


if __name__ == "__main__":
    main()
