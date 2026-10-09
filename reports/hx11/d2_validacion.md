# D2: validación interna sin etiquetas humanas

Responde a la auditoría del 2026-10-08 (§2.4, E1), con [`tools/hx11_d2_valida.py`](../../tools/hx11_d2_valida.py) sobre 394 partidos aptos. **No reemplaza la verdad humana:** mide si el puesto por tramo es una identidad estable en los datos y cuánto del rol por tick es ruido o movilidad.

## El puesto por tramo

- **Identificabilidad del ancla de 3 min:** en la ventana siguiente, el ancla de cada jugador está más cerca de la suya que de la de cualquier compañero en el **76.5%** de 31357 casos.
- **Deriva** del ancla entre ventanas, p10 / p50 / p90: 35.0 / 99.4 / 242.9 px. **Separación** con el compañero más cercano: 99.6 / 226.4 / 353.8 px. Deriva / separación, p50: 0.447.
- **Desplazamiento por tick** respecto del ancla propia del partido: 62.8 / 173.0 / 383.2 px (p10 / p50 / p90).

## El rol de plantilla de D2

- **Rol dominante, primera contra segunda mitad:** 83.4% de 7763 jugadores-partido.
- **Rol dominante contra la asignación independiente por anclas:** 95.6% de 7847.
- **Tiempo por tick:** en su rol dominante 63.1%, en permuta directa con el compañero que ocupa su rol 11.4%, otro 25.5%.

