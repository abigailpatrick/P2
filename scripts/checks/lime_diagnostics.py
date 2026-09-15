import numpy as np
from astropy.io import fits
import lime

path = "/ceph/cephfs/apatrick/P2/jwst_spectra/G235M_F170LP/59737_G235M_F170LP_spectra_lime.fits"
with fits.open(path) as h:
    t = h["SPECTRUM"].data
    w, f, e = (np.asarray(t[c], float) for c in ("WAVE", "FLUX", "ERR"))
g = np.isfinite(w) & np.isfinite(f) & np.isfinite(e)
print("finite pixels:", g.sum(), "of", g.size)

z = 3.4173
spec = lime.Spectrum(input_wave=w[g], input_flux=f[g], input_err=e[g],
                     redshift=z, units_wave="AA", units_flux="FLAM",
                     norm_flux=np.nanmedian(np.abs(f[g][f[g] > 0])))

# What lines does LiMe think are covered?
lf = spec.retrieve.lines_frame(vacuum_waves=True)
print("O3_5007A present?", "O3_5007A" in lf.index)
print("number of lines in frame:", len(lf))
print([ix for ix in lf.index if "O3" in ix or "Hb" in ix or "O2" in ix])

# Where does OIII fall, and is there data there?
oiii_obs = 5006.843 * (1 + z)
print("OIII observed AA:", oiii_obs)
near = (w[g] > oiii_obs - 60) & (w[g] < oiii_obs + 60)
print("pixels within +/-60 AA of OIII:", near.sum())
print("flux there finite frac:", np.isfinite(f[g][near]).mean() if near.sum() else "none")