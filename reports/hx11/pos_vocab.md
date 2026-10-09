# Vocabulario de `--pos`

P2 de la revisión de la auditoría, con [`tools/hx11_pos.py`](../../tools/hx11_pos.py). El ancla es relativa al centroide de los jugadores de campo, con el ataque hacia +x e y > 0 a la derecha. La dispersión es la mediana de la distancia de cada jugador-mitad al ancla del puesto (con y sin pelota). **Falta que la valide el usuario (E1 liviano).**

| Puesto | Línea | Ancla con pelota (x, y) | Ancla sin pelota (x, y) | Dispersión (px) | Jugadores-mitad |
|---|---|---|---|---|---|
| ARQ | ARQ | (-882.5, 2.3) | (-925.6, -1.4) | 130.5 / 134.0 | 1609 |
| CRD | MED | (-8.5, 412.1) | (-8.6, 393.9) | 167.1 / 163.7 | 457 |
| CRI | MED | (3.6, -404.3) | (8.9, -382.2) | 170.6 / 167.6 | 454 |
| DC | DEL | (335.7, -0.7) | (340.6, -0.6) | 109.6 / 106.3 | 1847 |
| DFC | DEF | (-314.6, 2.5) | (-308.5, 1.8) | 159.3 / 150.9 | 3696 |
| ED | DEL | (293.7, 296.5) | (298.3, 273.1) | 115.2 / 114.6 | 651 |
| EI | DEL | (286.8, -294.0) | (291.9, -281.4) | 118.1 / 112.6 | 653 |
| LD | DEF | (-210.1, 386.3) | (-210.9, 366.4) | 98.5 / 88.8 | 1177 |
| LI | DEF | (-217.9, -387.5) | (-212.9, -366.3) | 94.5 / 89.3 | 1176 |
| MC | MED | (62.3, -1.2) | (51.9, 0.4) | 142.8 / 140.9 | 3006 |
| MCD | MCD | (-23.8, -0.0) | (-35.7, -1.0) | 107.0 / 100.9 | 1028 |
| MCO | MED | (159.2, 10.1) | (152.4, 7.3) | 96.8 / 97.9 | 310 |
| MD | MED | (237.0, 365.6) | (238.8, 339.8) | 103.5 / 101.4 | 894 |
| MI | MED | (241.7, -371.0) | (244.1, -347.3) | 109.5 / 110.0 | 895 |

**Roles por formación** (para partidos con varios bots): 66 roles con >= 40 jugadores-mitad, en `pos_vocab.json` → `roles` (clave `formación:rol`, por ejemplo `4-3-3:DEF-CI`).
