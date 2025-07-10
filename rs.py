import camb
from camb import model

# Créer et configurer les paramètres cosmologiques
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(As=2e-9, ns=0.965)

# On demande à CAMB de calculer les distances et fonctions nécessaires
pars.set_accuracy(AccuracyBoost=2.0)  # optionnel, mais plus précis
results = camb.get_background(pars)

# Redshift de drag (découplage baryon-photon, approximativement ~1059, mais tu veux z = 1100)
z_drag = 1049

# Calcul de la distance sonore comobile jusqu’à z = 1100
rs = results.sound_horizon(z_drag)



