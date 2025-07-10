import matplotlib.pyplot as plt
import random as rdn
import numpy as np
import scipy
import math
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from iminuit import Minuit
import pandas as pd
import time 
from scipy.interpolate import interp1d
import math
import library

#importation des donnés
df = pd.read_csv("fsigma8_data.dat", sep=";")

z_data = np.array((df['z'].copy()).tolist())
fsig8 = np.array((df['fsig8'].copy()).tolist())
fsig8_err_minus = np.array((df['fsig8_err_minus'].copy()).tolist())
fsig8_err_plus = np.array((df['fsig8_err_plus'].copy()).tolist())







omega_m0_fid = 0.334
w_fid = -1
H0_fid = 73.4
c = 299792.458

def gamma(w,omega_m0,z):
    return   (3*(w-1))/(6*w-5) #- ((15/2057)*np.log(omega_m(z,omega_m0)))

def omega_m(z,omega_m0,w):
    return (omega_m0*(1/(omega_m0+(1-omega_m0)*(1+z)**(3*w))))

def omega_mGamma(z,gamma,omega_m0,w):
    return omega_m(z,omega_m0,w)**gamma

def siga8bis(omega_m0,sigma8_0):
    return sigma8_0*np.sqrt(0.3/omega_m0)
    
def func(z, omega_m0,w):
    return omega_mGamma(z,gamma(w,omega_m0,z),omega_m0,w)/(1+z)
    
def D_z(omega_m0,z,w):
    # Créer une fonction pour l'intégration
    result, error = quad(func,0,z,args =(omega_m0,w))
    return np.exp(-result)
    
def sigma8(sigma8_0,omega_m0,z,w):
    return sigma8_0*D_z(omega_m0,z,w)

def compute_D_interp(z_, omega_m0, w):
    z_unique, idx = np.unique(z_, return_index=True)
    D_vals = np.zeros_like(z_unique)
    for i, z in enumerate(z_unique):
        D_vals[i] = D_z(omega_m0, z, w)
    return interp1d(z_unique, D_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")


def Chi2(omega_m0, sigma8_0,w):
    
    # Interpolation rapide de sigma8
    D_interp = compute_D_interp(z_data, omega_m0, w)

    sigma8_vals = sigma8_0 * D_interp(z_data)

    pred = omega_mGamma(z_data,gamma(w,omega_m0,z_data),omega_m0,w) * sigma8_vals
    
    residuals = fsig8 - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    # contraintes externes
    #chi2_ext = ((omega_m0 - 0.334) / 0.018) ** 2 + ((sigma8_0 * np.sqrt(omega_m0 / 0.3) - 0.81) / 0.01) ** 2     

    return np.sum(chi2_terms)  

def compute_grid_Chi2(omega_vals, sigma_vals, w_vals, om_best, sig8_best, w_best):
    # Ωm0 vs σ8 (w fixé)
    OM, Sig8 = np.meshgrid(omega_vals, sigma_vals, indexing='ij')
    chi2_grid_om_sig8 = np.vectorize(lambda om, sig8: Chi2(om, sig8, w_best))(OM, Sig8)

    # Ωm0 vs w (σ8 fixé)
    OM, W = np.meshgrid(omega_vals, w_vals, indexing='ij')
    chi2_grid_om_w = np.vectorize(lambda om, w: Chi2(om, sig8_best,w))(OM, W)

    # σ8 vs w (Ωm0 fixé)
    Sig8, W = np.meshgrid(sigma_vals, w_vals, indexing='ij')
    chi2_grid_sig8_w = np.vectorize(lambda sig8, w: Chi2(om_best, sig8, w))(Sig8, W)

    return chi2_grid_om_sig8, chi2_grid_om_w, chi2_grid_sig8_w

def compute_grid_Chi2_ms(omega_vals, sigma_vals, w_vals, om_best, sig8_best, w_best):
    # Ωm0 vs σ8 (w fixé)
    OM, Sig8 = np.meshgrid(omega_vals, sigma_vals, indexing='ij')
    chi2_grid_om_sig8 = np.vectorize(lambda om, sig8: Chi2(om, sig8, w_best))(OM, Sig8)
    
    return chi2_grid_om_sig8

def H(H0,z,w,omega_m0):
    print(H0*np.sqrt(omega_m0*(1+z)**3+(1-omega_m0)**(3*(1+w))))
    return H0*np.sqrt(omega_m0*(1+z)**3+(1-omega_m0)**(3*(1+w)))

def f2(H0, z, w, omega_m0):
    h_val = H(H0, z, w, omega_m0)
    return 1. / h_val if h_val > 0 else np.inf 

def Da(H0,z,w,omega_m0):
    
    result, error = quad(f2,0,z,args =(H0,w,omega_m0))
    return c*result/(1+z)
    
def compute_Da_interp(z,omega_m0,w,H0):
    
    z_unique, idx = np.unique(z, return_index=True)
    Dc_vals = np.zeros_like(z_unique)
    
    for i, z in enumerate(z_unique):
        
        Dc_vals[i] = Da(H0,z,w,omega_m0)
        
    return interp1d(z_unique, Dc_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")

def HxDA(z,omega_m0,w,H0):
    Da_interp = compute_Da_interp(z,omega_m0,w,H0)
    return H(H0,z,w,omega_m0)*Da(H0,z,w,omega_m0)

def Chi2_AP(omega_m0,sigma8_0,w,H0):      #A MODOFIER !!!!!!
    
    # Interpolation rapide de sigma8
    D_interp = compute_D_interp(z_data, omega_m0, w)
    sigma8_vals = sigma8_0 * D_interp(z_data)

    pred = omega_mGamma(z_data,gamma(w,omega_m0,z_data),omega_m0,w) * sigma8_vals
    
    HxDA_fid = HxDA(z_data,omega_m0_fid,w_fid,H0_fid) 
    HxDAbis = HxDA(z_data,omega_m0,w,H0)
    correction_AP = HxDAbis/HxDA_fid
    
    residuals = fsig8*correction_AP - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    # contraintes externes
    #chi2_ext = ((omega_m0 - 0.334) / 0.018) ** 2 + ((sigma8_0 * np.sqrt(omega_m0 / 0.3) - 0.81) / 0.01) ** 2     
    print(chi2_terms)
    return np.sum(chi2_terms) 


minimizer = Minuit(Chi2_AP, omega_m0=0.3, sigma8_0=0.7,w=-1,H0 = 70)
minimizer.limits["omega_m0"] = (0,1)
minimizer.limits["sigma8_0"] = (0,1)  # Valeurs plus physiques
minimizer.limits["w"] = (-3,3)
minimizer.errordef = 1.0  # Pour chi2
minimizer.strategy = 2    # Plus précis
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
print(f"Paramètres estimés :")
print(f"Ωm = {minimizer.values['omega_m0']:.3f} ± {minimizer.errors['omega_m0']:.3f}")
print(f"σ8,0 = {minimizer.values['sigma8_0']:.3f} ± {minimizer.errors['sigma8_0']:.3f}")
print(f"w = {minimizer.values['w']:.3f} ± {minimizer.errors['w']:.3f}")
