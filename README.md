# PoC 06: contrato de salida del Motor de Inteligencia

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Fijar la forma del JSON que el Motor de Inteligencia devuelve a la API y comprobar dos supuestos:
1. SHAP reconstruye la probabilidad sin calibrar, no la calibrada.
2. Un modelo de árboles no extrapola fuera del rango de entrenamiento.

## Datos
Sintéticos: 20,000 registros con el diccionario de 8 variables de esa etapa (`diccionario_v1.yaml`, reducido a los campos que lee el script). El script también evalúa el escenario del peor mes, que luego quedó fuera del alcance.

## Cómo correrla
```
pip install -r requirements.txt
python poc_contrato_v1.py
```

## Resultado
| Escenario | Probabilidad del modelo | Probabilidad calibrada | SHAP reconstruye | Aditividad |
|---|---|---|---|---|
| Típico | 0.211 | 0.171 | 0.211 | Sí |
| Peor mes | 0.517 | 0.521 | 0.517 | Sí |

Prueba de saturación: con la razón cuota/ingreso en 0.74, 0.92, 1.85 y 4.62, la probabilidad del modelo se mantiene en 0.848.

## Hallazgos
1. SHAP explica la probabilidad sin calibrar. Por eso la interfaz muestra los factores como barras relativas y no como números.
2. El árbol no extrapola: por encima del máximo visto en entrenamiento, la predicción se queda fija.

## Decisión de diseño
El contrato de salida incluye:
- la probabilidad del modelo y la calibrada;
- la verificación de aditividad;
- los factores principales;
- las contribuciones;
- las advertencias de valores fuera de rango;
- el modo (BASE o LLM).

## Archivos
- `poc_contrato_v1.py`: script.
- `diccionario_v1.yaml`: diccionario que lee el script.
- `contrato_salida_v1.json`: salida estructurada.
- `salida_consola.txt`: salida completa de la consola.

## Corrida de referencia
6 de octubre de 2026, Linux, Python 3.12.3.
