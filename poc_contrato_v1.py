"""PoC contrato v1: datos SINTETICOS con el esquema del diccionario v1.
Objetivo unico: fijar la forma del contrato de salida (lo que consumira la API/OpenAPI)
y comprobar dos supuestos: (1) SHAP reconstruye la probabilidad SIN calibrar;
(2) que pasa con el escenario del peor mes si sale del rango de entrenamiento."""
import json, yaml, numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

dic = yaml.safe_load(open("diccionario_v1.yaml", encoding="utf-8"))  # mismo directorio
cols = [v["nombre"] for v in sorted(dic["variables"], key=lambda v: v["orden"])]

rng = np.random.default_rng(7); n = 20000
X = pd.DataFrame({
    "ratio_cuota_ingreso": rng.gamma(2, 0.08, n),
    "ratio_deuda_ingreso": rng.gamma(2, 0.6, n),
    "ratio_monto_ingreso": rng.gamma(2, 1.0, n),
    "n_creditos_activos": rng.poisson(1.5, n),
    "atraso_30d_12m": rng.binomial(1, 0.12, n),
    "edad": rng.integers(18, 70, n),
    "nivel_educativo": rng.integers(0, 4, n),
    "n_dependientes": rng.poisson(1.0, n),
})[cols]
logit = (-3.2 + 6*X.ratio_cuota_ingreso + 0.35*X.ratio_deuda_ingreso + 0.25*X.n_creditos_activos
         + 1.3*X.atraso_30d_12m - 0.02*(X.edad-40) - 0.15*X.nivel_educativo + 0.1*X.n_dependientes)
y = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)

X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, y_tr)
iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(X_cal)[:, 1], y_cal)
background = X_tr.sample(100, random_state=42)  # SHAP usa 100 por defecto con model_output="probability"
explainer = shap.TreeExplainer(clf.booster_, data=background, model_output="probability")
max_train = X_tr.max()

def evaluar(fila, k=3, minimo=0.02):
    p_modelo = float(clf.predict_proba(fila)[:, 1][0])
    p_cal = float(iso.predict([p_modelo])[0])
    sv = np.array(explainer.shap_values(fila)).reshape(-1)
    base = float(np.array(explainer.expected_value).reshape(-1)[-1])
    contrib = sorted(({"variable": c, "valor": round(float(fila[c].iloc[0]), 4),
                       "contribucion": round(float(v), 4)} for c, v in zip(cols, sv)),
                     key=lambda d: -d["contribucion"])
    top = [d for d in contrib if d["contribucion"] >= minimo][:k]
    fuera = [c for c in cols if fila[c].iloc[0] > max_train[c]]
    return {"probabilidad_modelo": round(p_modelo, 4), "probabilidad_calibrada": round(p_cal, 4),
            "shap": {"base_value": round(base, 4), "reconstruye": round(base + sv.sum(), 4),
                     "aditividad_ok": bool(abs(base + sv.sum() - p_modelo) < 1e-3)},
            "factores_top": top, "contribuciones": contrib,
            "advertencias": [f"{c} fuera del rango de entrenamiento" for c in fuera]}

# Caso: ingreso tipico 3000 soles, peor mes 1200, cuota 450, deuda total 30000, monto 12000
ing, peor, cuota = 3000, 1200, 450
caso = pd.DataFrame([{"ratio_cuota_ingreso": cuota/ing, "ratio_deuda_ingreso": 30000/(ing*12),
                      "ratio_monto_ingreso": 12000/(ing*12), "n_creditos_activos": 3,
                      "atraso_30d_12m": 0, "edad": 34, "nivel_educativo": 1, "n_dependientes": 2}])[cols]
estres = caso.copy(); estres["ratio_cuota_ingreso"] = cuota/peor

salida = {"version_contrato": "v1", "version_modelo": "v1-sintetico", "version_diccionario": dic["version"],
          "escenario_tipico": evaluar(caso), "escenario_peor_mes": evaluar(estres),
          "recomendaciones": [], "modo": "BASE"}
json.dump(salida, open("contrato_salida_v1.json", "w"), indent=2, ensure_ascii=False)

for nombre in ("escenario_tipico", "escenario_peor_mes"):
    e = salida[nombre]
    print(f"{nombre:20s} p_modelo={e['probabilidad_modelo']:.3f}  p_calibrada={e['probabilidad_calibrada']:.3f}  "
          f"SHAP reconstruye={e['shap']['reconstruye']:.3f}  aditividad_ok={e['shap']['aditividad_ok']}")
    print("   top:", [(d['variable'], d['contribucion']) for d in e["factores_top"]], e["advertencias"])

# Saturacion: el arbol no extrapola mas alla del maximo visto en entrenamiento
m = max_train["ratio_cuota_ingreso"]
for r in (m*0.8, m, m*2, m*5):
    f = caso.copy(); f["ratio_cuota_ingreso"] = r
    print(f"ratio_cuota_ingreso={r:5.2f} -> p_modelo={clf.predict_proba(f)[:,1][0]:.3f}")
