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
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def make_ds(name,position):                            #cette fonction fabrique une liste rempli par les données de la colonne de données voulue
    
    list_ = []
    
    count = 0
    
    # Ouvre le fichier en mode lecture
    with open(name, 'r') as fichier:
        # Lit chaque ligne du fichier
        for ligne in fichier:
            # Sépare les éléments de chaque ligne (par exemple, en utilisant l'espace comme séparateur)
            elements = ligne.split()
            
            # Parcours chaque élément de la ligne
            for element in elements:
                if count == position:
                    list_.append(float(element))
                count=count+1
            count = 0
    return list_


def make_list(name):                            #cette fonction fabrique un tableau cov
    
    tab = []
    i=0
    # Ouvre le fichier en mode lecture
    with open(name, 'r') as fichier:
        for element in fichier:
            # Sépare les éléments de chaque ligne (par exemple, en utilisant l'espace comme séparateur)
            i+=1
            tab.append(float(element))
        
    return tab

#import données pantheons

db_name = "pantheon.dat"
c = 299792.458

Zhd = make_ds(db_name,2)
MU_SHOES = make_ds(db_name,10)
IS_calib = make_ds(db_name,13)
ceph_dist = make_ds(db_name,12)
mb_corr = make_ds(db_name,8)

List = make_list("pantheoncov.cov")
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




def gamma(w,omega_m0,z):
    return   (3*(w-1))/(6*w-5) #- ((15/2057)*np.log(omega_m(z,omega_m0,w)))

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
    #print("H0 = ",H0,"omega_m = ",omega_m0,"sig8_0",sigma8_0,"M =",M,"w = ",w)
    return res



#########################################################################################################


def compute_chi2_om_sig8(om, sig8, h0, m, w_min, w_max, Chi2):
    minimizer = Minuit(Chi2, H0=h0, omega_m0=om, M=m, sigma8_0=sig8, w=-1)
    minimizer.fixed["omega_m0"] = True
    minimizer.fixed["sigma8_0"] = True
    minimizer.fixed["H0"] = True
    minimizer.fixed["M"] = True
    minimizer.limits["w"] = (w_min, w_max)
    minimizer.migrad()
    return minimizer.fval

def compute_chi2_om_w(om, w, h0, m, sig8_min, sig8_max, Chi2):
    minimizer = Minuit(Chi2, H0=h0, omega_m0=om, M=m, sigma8_0=0.7, w=w)
    minimizer.fixed["omega_m0"] = True
    minimizer.fixed["w"] = True
    minimizer.fixed["H0"] = True
    minimizer.fixed["M"] = True
    minimizer.limits["sigma8_0"] = (sig8_min, sig8_max)
    minimizer.migrad()
    return minimizer.fval

def compute_chi2_sig8_w(sig8, w, h0, m, om_min, om_max, Chi2):
    minimizer = Minuit(Chi2, H0=h0, omega_m0=0.3, M=m, sigma8_0=sig8, w=w)
    minimizer.fixed["sigma8_0"] = True
    minimizer.fixed["w"] = True
    minimizer.fixed["H0"] = True
    minimizer.fixed["M"] = True
    minimizer.limits["omega_m0"] = (om_min, om_max)
    minimizer.migrad()
    return minimizer.fval

def compute_grid_Chi2(om_vals, sig8_vals, w_vals, w_min, w_max, sig8_min, sig8_max, om_min, om_max):
    h0 = 73.4
    m = -19.25

    print(f"🧠 Cœurs utilisés : {os.cpu_count()}")

    # Nombre total d'itérations
    total1 = len(om_vals) * len(sig8_vals)
    total2 = len(om_vals) * len(w_vals)
    total3 = len(sig8_vals) * len(w_vals)

    # Affichage des barres de progression
    chi2_om_sig8 = Parallel(n_jobs=-1)(
        delayed(compute_chi2_om_sig8)(om, sig8, h0, m, w_min, w_max, Chi2)
        for om, sig8 in tqdm([(o, s) for o in om_vals for s in sig8_vals], desc="Calcul chi2(Ωm, σ8)", total=total1)
    )
    chi2_grid_om_sig8 = np.array(chi2_om_sig8).reshape(len(om_vals), len(sig8_vals))

    chi2_om_w = Parallel(n_jobs=-1)(
        delayed(compute_chi2_om_w)(om, w, h0, m, sig8_min, sig8_max, Chi2)
        for om, w in tqdm([(o, w) for o in om_vals for w in w_vals], desc="Calcul chi2(Ωm, w)", total=total2)
    )
    chi2_grid_om_w = np.array(chi2_om_w).reshape(len(om_vals), len(w_vals))

    chi2_sig8_w = Parallel(n_jobs=-1)(
        delayed(compute_chi2_sig8_w)(sig8, w, h0, m, om_min, om_max, Chi2)
        for sig8, w in tqdm([(s, w) for s in sig8_vals for w in w_vals], desc="Calcul chi2(σ8, w)", total=total3)
    )
    chi2_grid_sig8_w = np.array(chi2_sig8_w).reshape(len(sig8_vals), len(w_vals))

    return chi2_grid_om_sig8, chi2_grid_om_w, chi2_grid_sig8_w



########################################


def make_ds(name,position):                            #cette fonction fabrique une liste rempli par les données de la colonne de données voulue
    
    list_ = []
    
    count = 0
    
    # Ouvre le fichier en mode lecture
    with open(name, 'r') as fichier:
        # Lit chaque ligne du fichier
        for ligne in fichier:
            # Sépare les éléments de chaque ligne (par exemple, en utilisant l'espace comme séparateur)
            elements = ligne.split()
            
            # Parcours chaque élément de la ligne
            for element in elements:
                if count == position:
                    list_.append(float(element))
                count=count+1
            count = 0
    return list_


def make_list(name):                            #cette fonction fabrique un tableau cov
    
    tab = []
    i=0
    # Ouvre le fichier en mode lecture
    with open(name, 'r') as fichier:
        for element in fichier:
            # Sépare les éléments de chaque ligne (par exemple, en utilisant l'espace comme séparateur)
            i+=1
            tab.append(float(element))
        
    return tab



