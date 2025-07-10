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


# Importation des données
df = pd.read_csv("fsigma8_data.dat", sep=";")
z_data = np.array(df['z'].tolist())
fsig8 = np.array(df['fsig8'].tolist())
fsig8_err_minus = np.array(df['fsig8_err_minus'].tolist())
fsig8_err_plus = np.array(df['fsig8_err_plus'].tolist())

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

def f(z,H0,omega_m0):

    a  = (H0*np.sqrt(omega_m0*math.pow(1+z,3)+(1-omega_m0)))
    if a == 0:
        print("a = ",0)
    return 1/a

def Mu(z,H0,omega_m0):
    
    #print(type(X))

    # Calculer l'intégrale numérique de la fonction f(x) en utilisant la méthode des trapèzes
    I,err = quad(f,0,z,args = (H0,omega_m0))
    return 5*np.log10(((1+z)*c*I*(10**5)))


def Chi2Panth(H0,omega_m0,M):
    
    chi2 = 0
    diffMu = np.array([])
    #M = -19.25
    for i in range(0,len(Zhd)):
        if IS_calib[i] == 0:
            diffMu = np.append(diffMu,Mu(Zhd[i],H0,omega_m0)-(mb_corr[i]-M))
        else:
            diffMu =  np.append(diffMu,(mb_corr[i]-M)-ceph_dist[i])
        
    return np.dot(np.dot(Cov1,diffMu),diffMu)

def Chi2(H0,omega_m0,M,sigma8_0,w):
    
    #start = time.time()
    res = Chi2RSD(omega_m0,sigma8_0,w)+Chi2Panth(H0,omega_m0,M)
    #end = time.time()
    #print("t = ",end-start,"s")
    #print("H0 = ",H0,"omega_m = ",omega_m0,"sig8_0",sigma8_0,"M =",M)
    return res



#########################################################################################################


   


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



