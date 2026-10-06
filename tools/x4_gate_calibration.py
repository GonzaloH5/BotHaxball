"""¿Aprueba el gate a humanos reales? Calibración de `tools.x4_pass_chain.THRESHOLDS` (docs/PRELANZAMIENTO.md §1).

Muestras de grabaciones humanas de entrenamiento del tamaño de una evaluación (55 min de juego abierto, como las del
trainer; 150 min, como la certificación) se juzgan con el gate contra la referencia de prueba. Un gate bien calibrado
aprueba a casi todas: si no, rechazaría a un agente de nivel humano promedio.

  python -m tools.x4_gate_calibration      # escribe reports/x4/gate_calibracion_humanos.json
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
from tools import x4_pass_chain as PC
from learn.x4_epv import EPV
import torch; torch.set_num_threads(1)
epv = EPV(ROOT + '/runs/x4_epv/epv.pt')
cache = ROOT + '/data/pass_chain_human_units.pkl'
try:
    U = pickle.load(open(cache, 'rb'))
except FileNotFoundError:
    U = {sp: PC.human_units(sp, epv=epv) for sp in ('train', 'test')}
    pickle.dump(U, open(cache, 'wb'))
ref = json.load(open(ROOT + '/reports/x4/pass_chain_human.json'))['sanguchito_test']
rng = np.random.default_rng(0)
out = {}
for label, pool, minutes in (('train_55min', U['train'], 55), ('train_150min', U['train'], 150), ('train_todo', U['train'], 1e9)):
    res = []
    for k in range(100 if minutes < 1e8 else 1):
        idx = rng.permutation(len(pool))
        pick, m = [], 0.0
        for i in idx:
            pick.append(pool[i]); m += pool[i]['open_minutes']
            if m >= minutes: break
        b = PC.bootstrap(pick, n=200, seed=k)
        g = PC.gate(b, ref['metricas'], ref['banda'], PC.support(pick), strict=minutes > 100)
        res.append((g['aprobado'], g['indice_cadena'], g['checks'], min(g['indice_por_etapa'].values())))
    out[label] = dict(aprobados=sum(r[0] for r in res), n=len(res), indice=[round(float(np.mean([r[1] for r in res])), 3),
                      round(float(np.min([r[1] for r in res])), 3), round(float(np.max([r[1] for r in res])), 3)],
                      etapa_min=round(float(np.mean([r[3] for r in res])), 3),
                      fallas={c: sum(not r[2][c] for r in res) for c in res[0][2]})
    print(label, out[label], flush=True)
json.dump(dict(umbrales=PC.THRESHOLDS, resultados=out), open(ROOT + '/reports/x4/gate_calibracion_humanos.json', 'w'), indent=1, ensure_ascii=False)
