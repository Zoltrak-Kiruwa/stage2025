
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from iminuit import Minuit
import pandas as pd
import time 
import library
from tqdm import tqdm  # facultatif, non utilisé ici


om_min = 0.2
om_max = 0.6
sig8_min = 0.2
sig8_max = 1
w_min = -2
w_max = -1/3

minimizer = Minuit(library.Chi2,H0 = 73.4,omega_m0 = 0.3,M=-19.25,sigma8_0 = 0.7 ,w = -1)
minimizer.limits["omega_m0"] = (om_min,om_max)
minimizer.limits["sigma8_0"] = (sig8_min,sig8_max)  # Valeurs plus physiques
minimizer.limits["w"] = (w_min,w_max)
minimizer.fixed["H0"] = True
minimizer.fixed["M"] = True
minimizer.errordef = 1.0  # Pour chi2

result = minimizer.migrad()
print(result)
# Paramètres

N = 100
omega_m0_vals = np.linspace(om_min,om_max,N)
sigma8_0_vals = np.linspace(sig8_min,sig8_max,N)
w_vals = np.linspace(w_min,w_max,N)

print("Début du calcul des grilles χ²...")

start = time.time()
chi2_grid_om_sig8, chi2_grid_om_w, chi2_grid_sig8_w = library.compute_grid_Chi2(omega_m0_vals,sigma8_0_vals,w_vals,w_min,w_max,sig8_min,sig8_max, om_min, om_max)
end = time.time()

print(f"Calcul terminé en {end - start:.2f} secondes.")

# Sauvegarde
filename = "chi2_grids.npz"
np.savez_compressed(
    filename,
    omega_m0_vals=omega_m0_vals,
    sigma8_0_vals=sigma8_0_vals,
    w_vals=w_vals,
    chi2_grid_om_sig8=chi2_grid_om_sig8,
    chi2_grid_om_w=chi2_grid_om_w,
    chi2_grid_sig8_w=chi2_grid_sig8_w
)

print(f"Grilles sauvegardées dans '{filename}'.")