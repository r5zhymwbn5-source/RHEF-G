"""Recompute Q2 ensemble and paired bootstrap from exported prediction arrays."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
P=Path(__file__).resolve().parent
ref=json.loads((P/"evidence/paired_analysis.json").read_text())
weights=json.loads((P/"evidence/selected_weights.json").read_text())
z=np.load(P/"predictions/fusion_test.npz"); idx,y=z["indices"],z["y_true"]
models={}
for mode in weights["modes"]:
    parts=[]
    for seed in (42,2024,777):
        r=np.load(P/f"predictions/{mode}_{seed}_test.npz")
        assert np.array_equal(r["indices"],idx) and np.array_equal(r["y_true"],y)
        parts.append(r["probs"])
    models[mode]=np.mean(parts,axis=0)
fused=sum(w*models[m] for m,w in zip(weights["modes"],weights["weights"]))
assert np.allclose(fused,z["probs"],rtol=1e-6,atol=1e-7)
base=models["deepsets"]
def metrics(yy,pp):
    q=pp.argmax(axis=1)
    return {"macro_f1":float(f1_score(yy,q,labels=np.unique(yy),average="macro",zero_division=0)),"accuracy":float(accuracy_score(yy,q)),"mcc":float(matthews_corrcoef(yy,q)),"n":len(yy)}
for name, pp in [("rhef_g",fused),("deepsets_ensemble",base)]:
    m=metrics(y,pp)
    for k,v in m.items():assert np.isclose(v,ref[name][k],atol=1e-10,rtol=0),(name,k,v,ref[name][k])
    print(name,m)
zero=z["overlap_fraction"]==0
for name,pp in [("rhef_g",fused),("deepsets_ensemble",base)]:
    m=metrics(y[zero],pp[zero])
    for k,v in m.items():assert np.isclose(v,ref["zero_exact_training_protein_overlap"][name][k],atol=1e-10,rtol=0)
    print("zero CDS ID overlap",name,m)
groups=z["groups"]; rows=[np.flatnonzero(groups==g) for g in np.unique(groups)]
rng=np.random.default_rng(2026);differences=[]
for _ in range(2000):
    subset=np.concatenate([rows[i] for i in rng.integers(0,len(rows),size=len(rows))])
    differences.append(metrics(y[subset],fused[subset])["macro_f1"]-metrics(y[subset],base[subset])["macro_f1"])
ci=np.quantile(differences,[.025,.975]); expected=ref["paired_canonical_component_bootstrap"]["ci95"]
assert np.allclose(ci,expected,rtol=0,atol=1e-12),(ci,expected)
print("2000-draw paired component bootstrap",ci)
