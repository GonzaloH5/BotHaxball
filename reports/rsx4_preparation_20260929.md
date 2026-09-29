# RS4 normal — recs preparadas el 29/09/2026

- Origen: `replays_real/stadiums/rsx4`, 44 grabaciones.
- Dataset final: **41 recs, 1.854.744 muestras** en `data/bc/rsx4`.
- Se incorporaron **29 recs nuevas con 1.251.810 muestras** y se regeneraron las
  12 anteriores con RS4 normal: sin powershot, saques simplificados y sin árbitro
  Pegeche (slide/faltas). Los shards anteriores tenían powershot activado.
- Entrenamiento: 30 recs, 1.415.842 muestras. Validación: 11 recs, 438.902 muestras.
  La separación es por grabación completa, no por ticks del mismo partido.
- Formatos efectivos: 1.845.792 muestras de 4v4 y 8.952 de tramos de 3v3.
- Verificado: observaciones finitas, dimensiones, acciones 0–17 y etiquetas
  `act`, `act_lagm3`, `act_lag6`, `act_lag12`; metadata `rs_one`, powershot false,
  out_of_bounds true y peso 1.0 en todos los shards.

## Archivos no utilizados

1. `QDV 5-0 EHC Eternity.hbr2`: mapa de entrenamiento `Training RSR`.
2. `_MrHOST_-_4_-_Buenos_Aires_-1790563275736 (1).hbr2`: copia byte por byte de
   la grabación sin `(1)`. Su shard generado se apartó de la carpeta de
   entrenamiento, pero los dos replays originales se conservan.
3. `HBReplay-2026-09-27-23h30m.hbr2`: 11.294 ticks; 11.276 con equipos 4v3 y
   18 con 3v0. No cumple el filtro de equipos equilibrados y no genera muestras.
   El auditor lo muestra como `pending` porque no existe un shard, no porque
   haya material pendiente de conversión.

Respaldos recuperables:
`data/bc_backups/rsx4_before_normal_20260929_030444` (12 shards originales) y
`data/bc_backups/rsx4_excluded_20260929` (shard duplicado).

## Llevar los datos al Pod

Paquete local: `data/bc_exports/rsx4_normal_20260929.zip` (142.969.405 bytes).
El ZIP contiene únicamente `rsx4/*.npz`; se validaron sus 41 entradas y CRC.
SHA-256: `FF1FF1CEAF570CE0F160C854D5DD3BBE7BF8D3A3C4DEAF8365E3ACA32CA5FC1C`.
Los datos y el ZIP no se suben a Git. `git pull` trae las herramientas, no las recs.

Subir el ZIP con Jupyter a `/workspace/HaxballRL/data/bc_exports/` y, desde la
raíz del repositorio, extraerlo en `data/bc`:

```bash
python -m zipfile -e data/bc_exports/rsx4_normal_20260929.zip data/bc
```

La extracción reemplaza los shards RS4 antiguos del mismo nombre. No modifica
el PPO ni sus checkpoints. Si en el Pod existía el shard duplicado mencionado
arriba, apartarlo de `data/bc/rsx4` antes de entrenar.

Preparar los datos **no cambia automáticamente el imitador ni el bot en curso**.
El siguiente paso sería entrenar y evaluar un imitador nuevo en una corrida
separada, con los datasets de las demás modalidades disponibles en esa máquina.
Esta preparación sólo regeneró RS4; no reconstruyó los otros datasets ni
reemplazó `bc2`, `bc_rsx6_20260929` o `runs/multi/latest.pt`.
