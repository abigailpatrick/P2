import pandas as pd, numpy as np, glob, os
import lime
from astropy.io import fits
from astropy.table import Table

def band():
    w3=4958.911-15; w4=5006.843+15; w2=w3-5; w1=w2-20; w5=w4+5; w6=w5+20
    return pd.DataFrame({"wavelength":[5006.843],"w1":[w1],"w2":[w2],"w3":[w3],"w4":[w4],"w5":[w5],"w6":[w6]},index=["O3_5007A"])
cfg={"O3_5007A_b":"O3_4959A+O3_5007A","O3_4959A_amp":{"expr":"O3_5007A_amp/2.98"},"O3_4959A_kinem":"O3_5007A"}

CAT="/ceph/cephfs/apatrick/P2/jwst_catalogs"
ROOT="/ceph/cephfs/apatrick/P2/jwst_spectra"
vals=[]
for grating in ["G235M_F170LP","G235H_F170LP"]:
    cat=Table.read(os.path.join(CAT,f"JELS_F356W_DJA_{grating}_match_0p3as.fits"))
    for path in sorted(glob.glob(os.path.join(ROOT,grating,"*_lime.fits")))[:20]:
        sid=int(os.path.basename(path).split("_")[0])
        m=cat["ID"]==sid
        if m.sum()==0: continue
        z=float(cat["z"][m][0])
        try:
            with fits.open(path) as h:
                t=h["SPECTRUM"].data
                w=np.asarray(t["WAVE"],float); f=np.asarray(t["FLUX"],float); e=np.asarray(t["ERR"],float)
            g=np.isfinite(w)&np.isfinite(f)&np.isfinite(e); w,f,e=w[g],f[g],e[g]
            if len(w)==0: continue
            nf=np.nanmedian(np.abs(f[f>0]))
            spec=lime.Spectrum(w,f,e,redshift=z,units_wave="AA",units_flux="FLAM",norm_flux=nf)
            spec.fit.continuum(degree_list=[3,4],emis_threshold=[3,2])
            spec.fit.bands("O3_5007A_b",bands=band(),fit_cfg=cfg)
            r=spec.frame.loc["O3_5007A"]
            vals.append((sid,grating.split("_")[0],r["snr_line"],r["profile_flux"]/r["profile_flux_err"] if r["profile_flux_err"] else np.nan))
        except Exception as ex:
            pass
df=pd.DataFrame(vals,columns=["ID","gr","snr_line","prof_snr"])
print(df.to_string())
print("\nsnr_line pctiles:", np.nanpercentile(df.snr_line,[10,25,50,75,90]).round(1))