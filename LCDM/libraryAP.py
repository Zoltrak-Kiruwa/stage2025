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

# Importation des données
df = pd.read_csv("fsig8ml.dat", sep=";")
z_data = np.array(df['z'].tolist())
fsig8 = np.array(df['fsig8'].tolist())
fsig8_err_minus = np.array(df['fsig8_err_minus'].tolist())
fsig8_err_plus = np.array(df['fsig8_err_plus'].tolist())
Omegam0 = np.array(df['omega_m0'].tolist())
C_ij = np.array([[1/0.015**2, 0, 0, 0],[0, 1040.3, -807.5, 336.8],[0, -807.5, 3720.3, -1551.9],[0, 336.8, -1551.9, 2914.9]])
# Constantes
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

def func(z, omega_m0, w):
    return omega_mGamma(z, gamma(w, omega_m0, z), omega_m0, w) / (1 + z)

def D_z(omega_m0, z, w):
    result, _ = quad(func, 0, z, args=(omega_m0, w))
    return np.exp(-result)

def sigma8(sigma8_0, omega_m0, z, w):
    return sigma8_0 * np.array([D_z(omega_m0, zi, w) for zi in z])

def Chi2(omega_m0, sigma8_0, w):
    sigma8_vals = sigma8(sigma8_0, omega_m0, z_data, w)
    pred = omega_mGamma(z_data, gamma(w, omega_m0, z_data), omega_m0, w) * sigma8_vals

    residuals = fsig8 - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    return np.sum(chi2_terms)




#########################################################################################################

def H(z, H0, omega_m0, w):
    return H0 * np.sqrt(omega_m(z, omega_m0, w) * (1 + z) ** 3 + (1 - omega_m(z, omega_m0, w)) * (1 + z) ** (3 * (1 + w)))

def D_a(z, H0, omega_m0, w):
    def integrand(z, H0, omega_m0, w):
        return 1.0 / H(z, H0, omega_m0, w)

    result, _ = quad(integrand, 0, z, args=(H0, omega_m0, w))
    return c * result / (1 + z)

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

    residuals = fsig8 * correction(omega_m0, w, H0) - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    chi2 = np.sum(chi2_terms)
    return chi2


###################################################################################################


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

