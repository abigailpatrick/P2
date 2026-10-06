import glob, os, re
import numpy as np, pandas as pd

files = sorted(glob.glob("/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions/source_*_false_optimal_snr.csv"))
kept = set(pd.read_csv("/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions/false_positions_kept.csv")["ID"].astype(int))

rows = []
for f in files:
    sid = int(re.search(r"source_(\d+)_false_optimal_snr", os.path.basename(f)).group(1))
    s = pd.read_csv(f)["peak_snr"].to_numpy()
    rows.append((sid, sid in kept, np.median(s), np.percentile(s, 99), s.max()))

d = pd.DataFrame(rows, columns=["ID","in_kept","median","p99","max"]).sort_values("max", ascending=False)
print("files:", len(d), " not in kept:", (~d.in_kept).sum())
print(d.head(15).to_string(index=False))