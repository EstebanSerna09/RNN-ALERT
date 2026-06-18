# Optimización RNN — RNN-ALERT · UNIMAYOR

**Estado:** Corrección de bug aplicada · Pendiente de re-ejecución
**Última actualización:** 2026-06-14

---

## 1. Objetivo

Mejorar el desempeño del modelo RNN multimodal multitarea (RNN-ALERT) en la tarea de clasificación de riesgo de deserción estudiantil, especialmente en los programas con menor F1-score y recall, sin deteriorar la predicción de nota del siguiente semestre.

## 2. Resultados originales (línea base)

Entrenamiento masivo del 2026-06-13:

| Programa | AUC | Recall | F1 | Precision | MAE | R² | Umbral |
|---|---|---|---|---|---|---|---|
| ADMINISTRACION_DE_EMPRESAS | 0.8302 | **0.5000** | **0.6667** | 1.0000 | 0.1223 | 0.9844 | 0.61 |
| ADMINISTRACION_FINANCIERA | 0.9846 | 1.0000 | 0.6667 | 0.5000 | 0.1692 | 0.9567 | 0.05 |
| ARQUITECTURA | 0.9986 | 0.9722 | 0.9859 | 1.0000 | 0.1713 | 0.9605 | 0.61 |
| DISENO_VISUAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1401 | 0.9863 | 0.26 |
| INGENIERIA_INFORMATICA | 0.9972 | 0.9600 | 0.9600 | 0.9600 | 0.1560 | 0.9758 | 0.54 |
| TEC. DELINEANTES ARQ. ING. | 0.9405 | 0.8000 | 0.8615 | 0.9333 | 0.2032 | 0.9543 | 0.45 |
| TEC. DESARROLLO SOFTWARE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1265 | 0.9860 | 0.05 |
| **TEC. GESTION COMERCIAL Y MERCADOS** | **0.7179** | **0.5588** | **0.6441** | 0.7600 | 0.1550 | 0.9761 | 0.33 |
| TEC. GESTION EMPRESARIAL | 0.9074 | 0.8000 | 0.8615 | 0.9333 | 0.1074 | 0.9878 | 0.42 |
| TEC. GESTION FINANCIERA | 0.9830 | 0.9375 | 0.9231 | 0.9091 | 0.2021 | 0.9530 | 0.35 |
| **PROMEDIO** | **0.9359** | **0.8529** | **0.8570** | — | **0.1553** | **0.9721** | — |

## 3. Problema identificado (causas técnicas)

Tres deficiencias en el código original:

1. **Split sin estratificación** — `train_test_split` sin `stratify=`. En programas pequeños el azar puede generar test con muy pocos desertores.
2. **Sin `loss_weights`** — ambas salidas pesaban igual. El MSE de nota puede dominar implícitamente sobre la binary_crossentropy de deserción.
3. **`sample_weight` no activado** — `calcular_class_weights()` estaba implementado en `preparar_datos.py` pero nunca se usaba.

## 4. Bug detectado tras primera implementación

### Descripción del error

```
ValueError: Found input variables with inconsistent numbers of samples
```

Programas afectados:
- Administración de Empresas: [1330, 1334]
- Tecnología en Delineantes de Arquitectura e Ingeniería: [414, 415]
- Tecnología en Gestión Comercial y de Mercados: [549, 552]
- Tecnología en Gestión Empresarial: [803, 805]

### Causa raíz

El bug estaba en los pasos 3 y 4 de `preparar_datos_rnn()`. Los DataFrames `df_socio` y `df_label` se filtraban con `.isin(ids_comunes)` pero **no se deduplicaban por ID_EST**. Si un estudiante tenía múltiples filas en esos DataFrames (por programa, semestre u otro motivo), los arrays resultantes tenían más filas que `ids_comunes`:

```python
# CÓDIGO CON BUG:
df_label = df_label[df_label["ID_EST"].isin(ids_comunes)].sort_values("ID_EST")
y_desertor = df_label["DESERTOR"].values  # → puede tener N+k filas si hay duplicados

indices = np.arange(num_estudiantes)      # → exactamente N filas

# stratify=y_desertor falla porque len(y_desertor) != len(indices)
idx_train, idx_test = train_test_split(indices, stratify=y_desertor, ...)
```

La misma situación afectaba a `X_estatico` extraído de `df_socio`.

### Solución aplicada

Se refactorizó la construcción de `y_desertor` y `X_estatico` para que, igual que `X_secuencial`, se construyan indexando **directamente desde `ids_comunes`** usando diccionarios. Esto garantiza que los cuatro arrays tienen exactamente `len(ids_comunes)` filas en el mismo orden, independientemente de duplicados en los DataFrames:

```python
# SOLUCIÓN — deduplicar ANTES de extraer valores:
df_label_dedup = (
    df_label[df_label["ID_EST"].isin(ids_comunes)]
    .drop_duplicates(subset=["ID_EST"], keep="first")
    .set_index("ID_EST")
)
y_desertor_final = np.array(
    [int(df_label_dedup.loc[id_est, "DESERTOR"]) for id_est in ids_comunes],
    dtype="int32"
)
# → garantizadamente len(y_desertor_final) == len(ids_comunes)
```

Misma estrategia para `X_estatico` desde `df_socio`.

Se agregó además una **validación explícita** antes del split:

```python
assert X_secuencial.shape[0] == num_estudiantes
assert X_estatico.shape[0]   == num_estudiantes
assert len(y_desertor_final)  == num_estudiantes
assert len(y_nota_futura)     == num_estudiantes
```

Y el `stratify` se envuelve en `try/except ValueError` para fallback automático:

```python
try:
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y_desertor_final
    )
except ValueError:
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed
    )
    logger.warning("[RNN] stratify falló. Usando split aleatorio como fallback.")
```

### Programas que fallaban y estado esperado tras corrección

| Programa | Error anterior | Estado esperado |
|---|---|---|
| Administración de Empresas | [1330, 1334] muestras | Sin error — stratify o fallback |
| Tec. Delineantes Arq. e Ing. | [414, 415] muestras | Sin error — stratify o fallback |
| Tec. Gestión Comercial y Mercados | [549, 552] muestras | Sin error — stratify o fallback |
| Tec. Gestión Empresarial | [803, 805] muestras | Sin error — stratify o fallback |

## 5. Archivos modificados

| Archivo | Cambio | Respaldo |
|---|---|---|
| `modelos/modelo_rnn.py` | Corrección de alineamiento + optimizaciones | `modelo_rnn.py.bak` ✅ |
| `modelos/entrenamiento_masivo.py` | Sin cambios en producción | `entrenamiento_masivo.py.bak` ✅ |
| `experimentos_rnn.py` | Nuevo — script de experimentos comparativos | N/A |

## 6. Cambios en modelo_rnn.py

### Cambio 1 — Split estratificado con alineamiento correcto

```python
# ANTES (bug): y_desertor podía tener más filas que indices
idx_train, idx_test = train_test_split(indices, stratify=y_desertor, ...)

# DESPUÉS (correcto): y_desertor_final siempre == len(ids_comunes)
idx_train, idx_test = train_test_split(indices, stratify=y_desertor_final, ...)
# Con fallback automático via try/except si el programa es demasiado pequeño
```

### Cambio 2 — loss_weights para priorizar deserción

```python
modelo.compile(
    loss_weights={"prob_desercion": loss_weight_des, "pred_nota": 1.0}
)
# loss_weight_des=1.0 reproduce comportamiento original exactamente
```

### Cambio 3 — sample_weight balanceado (desactivado por defecto)

```python
sw = compute_class_weight("balanced", ...) / mean   # normalizado
modelo.fit(..., sample_weight=sw)
# usar_sample_weight=False reproduce comportamiento original
```

## 7. Tabla comparativa de experimentos

> **Pendiente:** Ejecuta `python experimentos_rnn.py` desde la carpeta del proyecto.

| Configuración | AUC | Recall | F1 | MAE | R² |
|---|---|---|---|---|---|
| A — RNN original | 0.9359 | 0.8529 | 0.8570 | 0.1553 | 0.9721 |
| B — +stratify | *pendiente* | *pendiente* | *pendiente* | *pendiente* | *pendiente* |
| C — +stratify +lw1.5 | *pendiente* | *pendiente* | *pendiente* | *pendiente* | *pendiente* |
| D — +stratify +lw2.0 | *pendiente* | *pendiente* | *pendiente* | *pendiente* | *pendiente* |
| E — +stratify +lw2.5 | *pendiente* | *pendiente* | *pendiente* | *pendiente* | *pendiente* |
| F — +stratify +lw2.0 +sw | *pendiente* | *pendiente* | *pendiente* | *pendiente* | *pendiente* |

## 8. Justificación académica

### Split estratificado
Kohavi (1995) demuestra que la estratificación reduce la varianza del estimador de error de generalización en datasets pequeños con clases desbalanceadas.

### loss_weights en modelos multitarea
Cipolla et al. (2018) — *Uncertainty Weighting* — establecen que en modelos multitarea la escala relativa de las pérdidas determina el balance de gradientes. Ajustar `loss_weights` es la técnica estándar.

### sample_weight / class_weight
He & Garcia (2009) — *Learning from Imbalanced Data* — sistematizan el uso de pesos por clase. Se prefiere sobre SMOTE porque no genera historias académicas sintéticas poco realistas en secuencias temporales.

## 9. Recomendación final

> **Pendiente de resultados.** Se actualizará automáticamente tras ejecutar `python experimentos_rnn.py`.

---

## Instrucciones de uso

### Ejecutar experimentos comparativos
```bash
cd C:\Users\walld\OneDrive\Escritorio\v4_rnn\rnn_alert
python experimentos_rnn.py
```

### Restaurar el modelo original
```powershell
Copy-Item modelos\modelo_rnn.py.bak modelos\modelo_rnn.py
```

---
*Preparado por el asistente de ML · RNN-ALERT · UNIMAYOR 2026*
