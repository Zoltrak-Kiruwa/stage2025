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

#import données pantheons

db_name = "pantheon.dat"
c = 299792.458

Zhd = library.make_ds(db_name,2)
MU_SHOES = library.make_ds(db_name,10)
IS_calib = library.make_ds(db_name,13)
ceph_dist = library.make_ds(db_name,12)
mb_corr = library.make_ds(db_name,8)

List = library.make_list("pantheoncov.cov")
n = 1701
def cov(List, n):
    # Redimensionner la liste en une matrice
    tab = np.array(List).reshape((n, n))
    return tab

# Exemple d'utilisation

Cov = cov(List, n)
Cov1 = np.linalg.inv(Cov)

#importation des donnés RSD
df = pd.read_csv("fsigma8_data.dat", sep=";")

z_data = np.array((df['z'].copy()).tolist())
fsig8 = np.array((df['fsig8'].copy()).tolist())
fsig8_err_minus = np.array((df['fsig8_err_minus'].copy()).tolist())
fsig8_err_plus = np.array((df['fsig8_err_plus'].copy()).tolist())


# Importation des données
df = pd.read_csv("fsigma8_data.dat", sep=";")
z_data = np.array(df['z'].tolist())
fsig8 = np.array(df['fsig8'].tolist())
fsig8_err_minus = np.array(df['fsig8_err_minus'].tolist())
fsig8_err_plus = np.array(df['fsig8_err_plus'].tolist())

def gamma(w,omega_m0,z):
    return   (3*(w-1))/(6*w-5) - ((15/2057)*np.log(omega_m(z,omega_m0,w)))

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

def Chi2RSD(omega_m0, sigma8_0, w):
    sigma8_vals = sigma8(sigma8_0, omega_m0, z_data, w)
    pred = omega_mGamma(z_data, gamma(w, omega_m0, z_data), omega_m0, w) * sigma8_vals

    residuals = fsig8 - pred
    sigma_errors = np.where(residuals >= 0, fsig8_err_plus, fsig8_err_minus)
    chi2_terms = (residuals / sigma_errors) ** 2

    return np.sum(chi2_terms)

def f(z,H0,omega_m,w):

    a  = (H0*np.sqrt(omega_m*math.pow((1+z),3)+(1-omega_m)*(math.pow((1+z),3*(1+w)))))
    if a == 0:
        print("a = ",0)
    return 1/a

def Mu(z,H0,omega_m,w):
    
    #print(type(X))

    # Calculer l'intégrale numérique de la fonction f(x) en utilisant la méthode des trapèzes
    I,err = quad(f,0,z,args = (H0,omega_m,w))
    return 5*np.log10(((1+z)*c*I*(10**5)))

def Chi2Panth(H0,omega_m,w,M):
    
    chi2 = 0
    diffMu = np.array([])

    for i in range(0,len(Zhd)):
        if IS_calib[i] == 0:
            diffMu = np.append(diffMu,Mu(Zhd[i],H0,omega_m,w)-(mb_corr[i]-M))
        else:
            diffMu =  np.append(diffMu,(mb_corr[i]-M)-ceph_dist[i])
    
    return np.dot(np.dot(Cov1,diffMu),diffMu)

def Chi2(H0,omega_m0,M,sigma8_0,w):
    
    #start = time.time()
    res = Chi2RSD(omega_m0,sigma8_0,w)+Chi2Panth(H0,omega_m0,w,M)
    #end = time.time()
    #print("t = ",end-start,"s")
    print("H0 = ",H0,"omega_m = ",omega_m0,"sig8_0",sigma8_0,"M =",M,"w = ",w)
    return res

minimizer = Minuit(Chi2,H0 = 70,omega_m0 = 0.3,M=-19.25,sigma8_0 = 0.7 ,w = -1)
minimizer.limits["omega_m0"] = (0,1)
minimizer.limits["sigma8_0"] = (0,1)  # Valeurs plus physiques
minimizer.limits["w"] = (-3,3)
minimizer.limits["H0"] = (0,100)
minimizer.fixed["M"] = True
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
print(f"H0 = {minimizer.values['H0']:.3f} ± {minimizer.errors['H0']:.3f}")
print(f"M = {minimizer.values['M']:.3f} ± {minimizer.errors['M']:.3f}")
print(f"w = {minimizer.values['w']:.3f} ± {minimizer.errors['w']:.3f}")