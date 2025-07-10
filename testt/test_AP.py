import numpy as np
import pandas as pd
from scipy.integrate import quad
from iminuit import Minuit

# Importation des données
df = pd.read_csv("fsig8ml.dat", sep=";")
z_data = np.array(df['z'].tolist())
fsig8 = np.array(df['fsig8'].tolist())
fsig8_err_minus = np.array(df['fsig8_err_minus'].tolist())
fsig8_err_plus = np.array(df['fsig8_err_plus'].tolist())
Omegam0 = np.array(df['omega_m0'].tolist())

# Constantes
omega_m0_fid = 0.334
w_fid = -1
H0_fid = 73.4
c = 299792.458

def gamma(w, omega_m0, z):
    return (3 * (w - 1)) / (6 * w - 5)

def omega_m(z, omega_m0, w):
    return omega_m0 / (omega_m0 + (1 - omega_m0) * (1 + z) ** (3 * w))

def omega_mGamma(z, gamma_val, omega_m0, w):
    return omega_m(z, omega_m0, w) ** gamma_val

def H(z, H0, omega_m0, w):
    return H0 * np.sqrt(omega_m(z, omega_m0, w) * (1 + z) ** 3 + (1 - omega_m(z, omega_m0, w)) * (1 + z) ** (3 * (1 + w)))

def D_a(z, H0, omega_m0, w):
    def integrand(z, H0, omega_m0, w):
        return 1.0 / H(z, H0, omega_m0, w)

    result, _ = quad(integrand, 0, z, args=(H0, omega_m0, w))
    return c * result / (1 + z)

def D_z(omega_m0, z, w):
    def integrand(z, omega_m0, w):
        return omega_mGamma(z, gamma(w, omega_m0, z), omega_m0, w) / (1 + z)

    result, _ = quad(integrand, 0, z, args=(omega_m0, w))
    return np.exp(-result)

def sigma8(sigma8_0, omega_m0, z, w):
    return sigma8_0 * np.array([D_z(omega_m0, zi, w) for zi in z])


def correction(omega_m0, w, H0):
    # Calculer H et H_fid pour tous les z_data en une seule fois
    Hubble = H(z_data, H0, omega_m0, w)
    H_fid = H(z_data, H0_fid, Omegam0, w_fid)

    # Calculer Da_fid pour tous les z_data en une seule fois
    Da_fid = np.array([D_a(z, H0_fid, omega_fid_i, w_fid) for z, omega_fid_i in zip(z_data, Omegam0)])

    # Calculer Da_vals pour tous les z_data en une seule fois
    Da_vals = np.array([D_a(z, H0, omega_m0, w) for z in z_data])

    # Retourner le résultat final
    return (Hubble * Da_vals) / (H_fid * Da_fid)


def Chi2_AP(omega_m0, sigma8_0, w, H0):
    
    sigma8_vals = sigma8(sigma8_0, omega_m0, z_data, w)
    pred = omega_mGamma(z_data, gamma(w, omega_m0, z_data), omega_m0, w) * sigma8_vals
    print(correction(omega_m0, w, H0))
    residuals = fsig8 * correction(omega_m0, w, H0) - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    chi2 = np.sum(chi2_terms)
    return chi2

# Minimisation
minimizer = Minuit(Chi2_AP, omega_m0=0.4, sigma8_0=0.7, w=-1, H0=73.4)
minimizer.limits["omega_m0"] = (0, 1)
minimizer.limits["sigma8_0"] = (0, 1)
minimizer.fixed["H0"] = True
minimizer.fixed["w"] = True
minimizer.errordef = 1.0
minimizer.strategy = 2

result = minimizer.migrad()
print(result)

# Statistiques du fit
chi2_val = minimizer.fval
ndof = len(fsig8) - len(minimizer.parameters)
chi2_reduit = chi2_val / ndof

print(f"\nRésultats du fit :")
print(f"Chi2 = {chi2_val:.2f}")
print(f"Nombre de degrés de liberté = {ndof}")
print(f"Chi2 réduit = {chi2_reduit:.2f}")
print("Paramètres estimés :")
print(f"Ωm = {minimizer.values['omega_m0']:.3f} ± {minimizer.errors['omega_m0']:.3f}")
print(f"σ8,0 = {minimizer.values['sigma8_0']:.3f} ± {minimizer.errors['sigma8_0']:.3f}")

