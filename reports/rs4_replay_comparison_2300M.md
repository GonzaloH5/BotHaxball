# Comparación de grabaciones RS4: 2300M frente al campeón de 1400M

Fecha: 2026-10-02. Checkpoints descargados del Pod y verificados por SHA-256.
Actual: `fba155fc136afe1eefebec6efabf430ca080d30f093d324a1b029f926172b39f`.
Campeón: `1d31671bc8f3df91b19875256084ba5d9e4673e294e896126c8838ba7cb825f6`.

Cada grabación dura dos minutos, RS4 4v4, acciones greedy y reglas del checkpoint.
No se ejecutó PPO ni se modificó el campeón. Esta muestra no aprueba fases ni
reemplaza la evaluación factorial completa. El campeón no es el checkpoint
inmediatamente anterior a los cambios: no permite aislar su efecto causal.

| Rival R3, semilla 51, agente rojo | Actual 2300M | Campeón 1400M |
|---|---|---|
| Equilibrado | 3–0 | 5–0 |
| Agresivo | 2–0 | 4–0 |
| Conservador | 2–0 | 4–0 |

R3 con estilo mezclado: actual 2–0 en semillas 51/73/91; campeón 5–0, 2–0,
5–0 respectivamente. En semilla 73, el modelo inspeccionado juega de azul.
Duelo actual contra campeón: 0–0 en las tres semillas (actual azul en 73).
No hubo expiraciones de saque central registradas en estas quince grabaciones.
El replay no resume expiraciones de córner; este resultado no confirma su ausencia.

Se verificó que las trayectorias de semillas 51 y 91 con el mismo color son
idénticas tanto contra R3 mezclado como en el duelo neuronal. Greedy más el
saque inicial fijo limita la diversidad: estas repeticiones no son evidencia
independiente. Los estilos explícitos de R3 aportan variantes de rival.

Conclusión: ambos checkpoints dominan R3 en esta muestra. El actual marca menos
contra cada estilo y no supera al campeón en el duelo directo. La mejora visual
reportada por el usuario aún no demuestra una mejora competitiva; tampoco estos
pocos partidos bastan para concluir una regresión general. Revisar apoyos,
espacios, presión y saques en los replays antes de decidir otra modificación.

Abrir `replays/rs4_comparacion_2300M.html` para seleccionar las quince grabaciones.
Datos: `reports/rs4_replay_comparison_2300M.json`.
Reproducción: `.venv/Scripts/python.exe reports/rs4_replay_comparison.py`.
