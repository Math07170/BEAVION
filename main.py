#l'objectif est de faire un programme pour la séance 2 du cours l'énnoncée est dans "sujet.pdf"
#on utiliseras les fichier python fournis

#Pour cette séance, vous donnerez successivement à l’altitude hG, au nombre de Mach Ma, à
"""la marge statique ms et au coefficient km de réglage de la masse de l’avion les valeurs suivantes :

hG ∈{4000,6000,8000,10000} mètres
Ma ∈{0.4,0.6,0.8}
ms ∈{0.2,1}
km ∈{0.1,1}
L’ensemble des points de trim est donc un ensemble constitué de 4 ×3 ×2 ×2 = 48 points.
1. Utilisez la méthode numérique de trim fournie dans le code Python pour déterminer les
valeurs de α, δtrim et δth correspondant à un vol en palier. Vous pourrez par exemple
présenter les différentes valeurs de trim en fonction de l’altitude, paramétrées pour deux
valeurs du nombre de Mach, 0.4 et 0.8 par exemple ; vous pourrez associer une figure pour
chaque valeur du couple {marges statique ms, coefficient de réglage de la masse km}. Vous
pourrez mettre en parallèle ces résultats avec ceux de la question précédente.
"""
# ...existing code...

import numpy as np
import matplotlib.pyplot as plt

import aero_model
import dynamic

# Paramètres à balayer
altitudes = [4000, 6000, 8000, 10000]  # en mètres
mach_numbers = [0.4, 0.6, 0.8]
static_margins = [0.2, 1]
km_values = [0.1, 1]

# Modèle d'avion (exemple : Airbus A320)
aircraft = aero_model.Airbus_A320_200()

results = []

for ms in static_margins:
    for km in km_values:
        aircraft.set_static_margin(ms)
        mass = aircraft.set_mass_from_km(km)
        for Ma in mach_numbers:
            for hG in altitudes:
                tas = aircraft.atm.tas_from_mach_altp(Ma, hG)
                try:
                    trim = dynamic.get_trim_level_flight(aircraft, hG, tas)
                    results.append({
                        "hG": hG,
                        "Ma": Ma,
                        "ms": ms,
                        "km": km,
                        "aoa_deg": np.rad2deg(trim["aoa"][0]),
                        "dtrim_deg": np.rad2deg(trim["dtrim"][0]),
                        "dthr": trim["dthr"][0]
                    })
                except Exception as e:
                    results.append({
                        "hG": hG,
                        "Ma": Ma,
                        "ms": ms,
                        "km": km,
                        "aoa_deg": np.nan,
                        "dtrim_deg": np.nan,
                        "dthr": np.nan
                    })

# Affichage des résultats (exemple pour Ma=0.4 et Ma=0.8)
import pandas as pd
df = pd.DataFrame(results)
for ms in static_margins:
    for km in km_values:
        for Ma in [0.4, 0.8]:
            subset = df[(df.ms == ms) & (df.km == km) & (df.Ma == Ma)]
            plt.plot(subset.hG, subset.aoa_deg, marker='o', label=f"ms={ms}, km={km}, Ma={Ma}")
plt.xlabel("Altitude (m)")
plt.ylabel("Alpha trim (deg)")
plt.title("Trim α en fonction de l'altitude")
plt.legend()
plt.show()

"""Utilisez la méthode numérique de trim fournie dans le code Python pour tracer les courbes
de la poussée nécessaire au vol en palier en fonction du nombre de Mach, paramétrées pour
deux valeurs de l’altitude, 4000 mètres et 10000 mètres par exemple ; vous pourrez associer
une figure pour chaque valeur du couple {marges statique ms, coefficient de réglage de la
masse km}."""
for ms in static_margins:
    for km in km_values:
        aircraft.set_static_margin(ms)
        mass = aircraft.set_mass_from_km(km)
        for hG in [4000, 10000]:
            machs = np.linspace(0.4, 0.8, 10)
            pouss = []
            for Ma in machs:
                tas = aircraft.atm.tas_from_mach_altp(Ma, hG)
                try:
                    trim = dynamic.get_trim_level_flight(aircraft, hG, tas)
                    pouss.append(trim["fu"][0])  # Poussée réelle en daN
                except Exception as e:
                    pouss.append(np.nan)
            plt.plot(machs, pouss, marker='o', label=f"ms={ms}, km={km}, hG={hG}")
plt.xlabel("Nombre de Mach")
plt.ylabel("Poussée nécessaire (daN)")
plt.title("Poussée nécessaire au vol en palier en fonction du nombre de Mach")
plt.legend()
plt.show()

#question 3
