# D3: valor de acciones y estados en HaxEleven

Etapa D3 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con el método de [`tools/hx11_d3.py`](../../tools/hx11_d3.py). Partidos por parte: {'train': 290, 'test': 64, 'dev': 40}. Los datos están en [`d3_valor.json`](d3_valor.json).

## 1. Control del espacio

- Ajuste por verosimilitud en 6000 pases de train: reacción **15 ticks**, sigma **4 ticks** (vmax 2.86 px/tick de la física, tasa 4.3/s).
- En dev (3779 pases): log-verosimilitud media -0.392 contra -0.6908 de la tasa base, AUC 0.915.
- La P está medida en el punto donde termina el pase. Para un pase cortado, ese punto es donde lo tocó el rival, así que la separación es optimista. Sirve para ver la calibración, no como precisión de un pase planeado.

| P(control) | Pases | P media | Completados |
|---|---|---|---|
| 0.0–0.1 | 527 | 0.034 | 0.015 |
| 0.1–0.2 | 292 | 0.15 | 0.065 |
| 0.2–0.3 | 295 | 0.248 | 0.105 |
| 0.3–0.4 | 280 | 0.352 | 0.125 |
| 0.4–0.5 | 409 | 0.462 | 0.355 |
| 0.5–0.6 | 348 | 0.538 | 0.466 |
| 0.6–0.7 | 210 | 0.651 | 0.533 |
| 0.7–0.8 | 207 | 0.751 | 0.618 |
| 0.8–0.9 | 226 | 0.85 | 0.77 |
| 0.9–1.0 | 985 | 0.978 | 0.96 |

## 2. VAEP (test)

| Modelo | Acciones | Tasa | Log-loss | Base | Brier | AUC |
|---|---|---|---|---|---|---|
| marcar en 10 s | 40828 | 0.0122 | 0.0455 | 0.0661 | 0.00992 | 0.905 |
| recibir en 10 s | 40828 | 0.0078 | 0.0373 | 0.0458 | 0.00757 | 0.876 |

**Valor por tipo de acción (todas las partes):**

| Acción | n | Media | Suma |
|---|---|---|---|
| conduccion | 17400 | 0.0164 | 284.9 |
| remate | 3268 | 0.0765 | 250.1 |
| despeje | 12715 | 0.0089 | 112.9 |
| pase | 108479 | 0.001 | 112.8 |
| centro | 2334 | 0.0219 | 51.1 |
| recuperacion | 8762 | 0.0037 | 32.2 |
| intercepcion | 34707 | 0.0005 | 17.4 |
| robo | 15708 | 0.0003 | 4.1 |
| falta | 233 | -0.0016 | -0.4 |
| atajada | 1094 | -0.0215 | -23.5 |
| bloqueo | 20836 | -0.0071 | -148.5 |
| perdida | 28053 | -0.0121 | -338.8 |

**VAEP por jugador cada 10 minutos, por línea del puesto (D2; jugadores con >= 5 min):**

| Línea | Jugadores-partido | p10 | p50 | p90 |
|---|---|---|---|---|
| ARQ | 775 | -0.101 | 0.024 | 0.163 |
| DEF | 2895 | -0.119 | -0.012 | 0.11 |
| DEL | 1486 | -0.237 | -0.009 | 0.729 |
| MCD | 488 | -0.153 | -0.008 | 0.194 |
| MED | 2812 | -0.168 | -0.003 | 0.368 |
| R0 | 6 | -0.21 | -0.061 | 0.923 |
| R1 | 6 | 0.084 | 0.52 | 1.707 |
| R2 | 6 | -0.111 | 0.005 | 0.551 |
| R3 | 6 | -0.522 | -0.049 | 0.07 |
| R4 | 5 | -0.152 | 0.013 | 0.086 |
| R5 | 6 | -0.383 | -0.033 | 0.339 |
| R6 | 5 | -0.121 | -0.1 | 0.381 |
| R7 | 5 | -0.239 | 0.147 | 0.168 |
| R8 | 4 | -0.149 | 0.044 | 0.099 |
| R9 | 5 | 0.075 | 0.232 | 0.945 |

## 3. Valor de estado (test)

| Modelo | Ticks | Tasa | Log-loss | Base | Brier | AUC |
|---|---|---|---|---|---|---|
| gol a favor en 10 s | 147552 | 0.0146 | 0.0539 | 0.0762 | 0.01247 | 0.907 |
| gol en contra en 10 s | 147552 | 0.0055 | 0.028 | 0.0339 | 0.00529 | 0.865 |

**Φ = P(a favor) − P(en contra) según dónde está la pelota (x hacia el ataque del equipo con la pelota):**

| x | Ticks | Φ medio | Gol a favor | Gol en contra |
|---|---|---|---|---|
| -1800..-1200 | 19997 | -0.0255 | 0.0 | 0.018 |
| -1200..-600 | 27303 | -0.0114 | 0.0 | 0.0113 |
| -600..0 | 28977 | -0.0012 | 0.0023 | 0.0043 |
| 0..600 | 28291 | 0.0153 | 0.0101 | 0.0003 |
| 600..1200 | 24841 | 0.0364 | 0.0308 | 0.0 |
| 1200..1800 | 17557 | 0.0693 | 0.0583 | 0.0 |

## 4. Validación del valor (auditoría E3)

Dos modelos de familias y rasgos distintos. **A** (MLP, con contexto) es el candidato a shaping; **B** (logística sin acciones previas para VAEP; grilla de la pelota para el estado) es el juez independiente. Ninguno se usa a la vez para entrenar y para juzgar.

**VAEP, modelo B** (test):

| Modelo | Log-loss | Base | Brier | AUC |
|---|---|---|---|---|
| marcar | 0.0456 | 0.0661 | 0.00997 | 0.906 |
| recibir | 0.0363 | 0.0458 | 0.00749 | 0.885 |

**Confiabilidad, VAEP A, marcar en 10 s** (deciles de P; ECE 0.0015):

| P media | Observada | n |
|---|---|---|
| 0.0 | 0.0 | 4083 |
| 0.0 | 0.0002 | 4083 |
| 0.0001 | 0.0002 | 4083 |
| 0.0003 | 0.0 | 4082 |
| 0.0008 | 0.0007 | 4083 |
| 0.0022 | 0.0032 | 4083 |
| 0.0046 | 0.0047 | 4082 |
| 0.009 | 0.0132 | 4083 |
| 0.0181 | 0.0193 | 4083 |
| 0.0885 | 0.0808 | 4083 |

**Confiabilidad, VAEP A, recibir en 10 s** (deciles de P; ECE 0.00106):

| P media | Observada | n |
|---|---|---|
| 0.0 | 0.0 | 4083 |
| 0.0 | 0.0 | 4083 |
| 0.0001 | 0.0 | 4083 |
| 0.0003 | 0.0005 | 4082 |
| 0.0008 | 0.0015 | 4083 |
| 0.0018 | 0.0017 | 4083 |
| 0.0037 | 0.0059 | 4082 |
| 0.0073 | 0.0071 | 4083 |
| 0.0151 | 0.0184 | 4083 |
| 0.0472 | 0.0434 | 4083 |

**Confiabilidad, VAEP B, marcar en 10 s** (deciles de P; ECE 0.00144):

| P media | Observada | n |
|---|---|---|
| 0.0 | 0.0 | 4083 |
| 0.0002 | 0.0 | 4083 |
| 0.0004 | 0.0 | 4083 |
| 0.0008 | 0.0 | 4082 |
| 0.0015 | 0.0017 | 4083 |
| 0.0026 | 0.0042 | 4083 |
| 0.0048 | 0.0066 | 4082 |
| 0.009 | 0.0103 | 4083 |
| 0.019 | 0.0162 | 4083 |
| 0.0887 | 0.0835 | 4083 |

**Valor de estado, modelo B** (test):

| Modelo | Log-loss | Base | Brier | AUC |
|---|---|---|---|---|
| a_favor | 0.0599 | 0.0762 | 0.01345 | 0.87 |
| en_contra | 0.0293 | 0.0339 | 0.00536 | 0.833 |

**Confiabilidad, estado A, gol a favor en 10 s** (deciles de P; ECE 0.00394):

| P media | Observada | n |
|---|---|---|
| 0.0 | 0.0 | 14756 |
| 0.0 | 0.0 | 14755 |
| 0.0001 | 0.0 | 14755 |
| 0.0003 | 0.0002 | 14755 |
| 0.0013 | 0.0025 | 14755 |
| 0.0036 | 0.0025 | 14755 |
| 0.0079 | 0.0065 | 14755 |
| 0.0155 | 0.0121 | 14755 |
| 0.0312 | 0.0281 | 14755 |
| 0.1229 | 0.094 | 14756 |

**Confiabilidad, estado A, gol en contra en 10 s** (deciles de P; ECE 0.00165):

| P media | Observada | n |
|---|---|---|
| 0.0 | 0.0 | 14756 |
| 0.0 | 0.0 | 14755 |
| 0.0001 | 0.0 | 14755 |
| 0.0002 | 0.0001 | 14755 |
| 0.0007 | 0.0007 | 14755 |
| 0.0018 | 0.0027 | 14755 |
| 0.004 | 0.0047 | 14755 |
| 0.0075 | 0.007 | 14755 |
| 0.0138 | 0.0113 | 14755 |
| 0.0396 | 0.0279 | 14756 |

**Confiabilidad, estado B, gol a favor en 10 s** (deciles de P; ECE 0.00247):

| P media | Observada | n |
|---|---|---|
| 0.0001 | 0.0001 | 13028 |
| 0.0001 | 0.0 | 16471 |
| 0.0003 | 0.0002 | 12453 |
| 0.001 | 0.0001 | 16926 |
| 0.0031 | 0.0035 | 13518 |
| 0.0085 | 0.0071 | 16018 |
| 0.0127 | 0.0111 | 14059 |
| 0.019 | 0.0149 | 14466 |
| 0.0303 | 0.0257 | 15070 |
| 0.0897 | 0.0791 | 15543 |

**Acuerdo entre A y B:**

- VAEP por acción (test): Spearman 0.536; por jugador-partido cada 10 min: 0.568.
- Φ por tick (test): Spearman 0.903, Pearson 0.711.

**Fiabilidad entre partidos del valor por jugador** (clave anónima del nombre; sin arqueros):

- **VAEP A cada 10 min:** r = 0.197 entre mitades de sus partidos (IC 90% 0.124 a 0.276), Spearman-Brown 0.33, 301 jugadores con >= 6 partidos.
- **VAEP B cada 10 min:** r = 0.283 entre mitades de sus partidos (IC 90% 0.209 a 0.342), Spearman-Brown 0.442, 301 jugadores con >= 6 partidos.
- **Pase completo (referencia simple):** r = 0.501 entre mitades de sus partidos (IC 90% 0.423 a 0.555), Spearman-Brown 0.667, 287 jugadores con >= 6 partidos.
