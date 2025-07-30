import numpy as np
from scipy.constants import h, c, k, sigma, e
import matplotlib.pyplot as plt

# =======================
# PARÁMETROS DE ENTRADA
# =======================

# Emisor térmico
A_emisor = 0.01      # Área del emisor [m^2]
T_emisor = 2000     # Temperatura del emisor [K]
VF = 1             # Factor de vista (adimensional, entre 0 y 1)
emisividad = 1.0     # Suponemos cuerpo negro (emisividad unitaria)

# Celda TPV
Eg_eV = 1          # Bandgap de la celda [eV]
EQE = 0.8            # Eficiencia cuántica externa (puede ser función de lambda, aquí constante)
A_abs = 0.8          # Absorptancia media, basada en literatura
n_aire = 1.0         # Índice de refracción del aire
n_celda = 3.5        # Índice de refracción de la celda
theta_i_deg = 0      # Ángulo de incidencia [grados] (0 = normal a la superficie)

# Reciclaje de radiación
R_backplate = 0.99      # Coeficiente de reflexión del backplate (usuario define)
VF_reciclaje = 1    # Factor de vista para el reciclaje (usuario define, <1 para pérdidas realistas)

# =======================
# CONSTANTES Y VECTORES 
# =======================
theta_i = np.deg2rad(theta_i_deg)
lmin = 0.1e-6        # [m]
lmax = 5e-6          # [m]
num_points = 2000
wavelengths = np.linspace(lmin, lmax, num_points)  # [m]

# =======================
# FUNCIONES ÓPTICAS (FRESNEL)
# =======================

def fresnel_R(n1, n2, theta_i):
    try:
        sin_theta_t = n1 / n2 * np.sin(theta_i)
        theta_t = np.arcsin(sin_theta_t)
    except:
        return 1.0
    rs = ((n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) /
          (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))) ** 2
    rp = ((n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) /
          (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))) ** 2
    return (rs + rp) / 2

# =======================
# LEY DE PLANCK (CUERPO NEGRO)
# =======================

def E_BB_lambda(lamb, T):
    exponent = h * c / (lamb * k * T)
    return (2 * np.pi * h * c**2) / (lamb**5 * (np.exp(exponent) - 1))

# =======================
# CÁLCULO DE LONGITUD DE ONDA DE CORTE (BANDGAP)
# =======================

lambda_g = h * c / (Eg_eV * e)   # [m]

# =======================
# CÁLCULO DE R, A, T EN CADA LAMBDA
# =======================

R_vals = np.full_like(wavelengths, fresnel_R(n_aire, n_celda, theta_i))

A_vals = np.zeros_like(wavelengths)
T_vals = np.zeros_like(wavelengths)

# Para lambda < lambda_g, se usa A fija (literatura), T por complementariedad
mask_util = wavelengths < lambda_g
A_vals[mask_util] = A_abs
T_vals[mask_util] = 1 - R_vals[mask_util] - A_vals[mask_util]

# Para lambda > lambda_g, absorción nula, T = 1 - R
mask_sub = wavelengths >= lambda_g
A_vals[mask_sub] = 0
T_vals[mask_sub] = 1 - R_vals[mask_sub]

# Evita transmitancia negativa por redondeos numéricos
T_vals[T_vals < 0] = 0

# =======================
# ESPECTRO DEL EMISOR Y RADIACIÓN TOTAL
# =======================

E_lambda = E_BB_lambda(wavelengths, T_emisor)  # [W/m^2/m]

# Potencia total emitida (Stefan-Boltzmann + área + emisividad)
P_emisor = emisividad * sigma * T_emisor**4 * A_emisor

# =======================
# POTENCIA ABSORBIDA POR LA CELDA (INTEGRACIÓN SOBRE BANDGAP)
# =======================

P_abs = VF * A_emisor * np.trapz(E_lambda * A_vals * EQE, wavelengths)  # [W]

# =======================
# CÁLCULO DE POTENCIAS REFLEJADA, TRANSMITIDA Y RECICLADA
# =======================

# 1. Radiación reflejada superficialmente por la celda TPV
P_reflejada = VF * A_emisor * np.trapz(E_lambda * R_vals, wavelengths)  # [W]
P_reflejada_reciclada = P_reflejada * VF_reciclaje

# 2. Radiación transmitida a través de la celda TPV (llega al backplate)
P_transmitida = VF * A_emisor * np.trapz(E_lambda * T_vals, wavelengths)  # [W]
P_reincidente = P_transmitida * R_backplate * VF_reciclaje

# =======================
# CÁLCULO DE EFICIENCIA TPV (balance energético realista)
# =======================

denominador = P_emisor - P_reflejada_reciclada - P_reincidente

if denominador > 0:
    Eficiencia_TPV = P_abs / denominador
else:
    Eficiencia_TPV = 0

# =======================
# MOSTRAR RESULTADOS
# =======================

print("RESULTADOS TPV:")
print(f"Radiación total emitida por el emisor: {P_emisor:} W")
print(f"Potencia eléctrica generada por el sistema: {P_abs:} W")
print(f"Radiación reflejada superficialmente (por la celda TPV): {P_reflejada:} W")
print(f"Radiación transmitida (llega a backplate): {P_transmitida:} W")
print(f"Radiación reciclada por reflexión directa (TPV): {P_reflejada_reciclada:} W")
print(f"Radiación reciclada tras backplate: {P_reincidente:} W")
print(f"Fracción absorbida respecto a lo emitido: {P_abs/P_emisor:.2%}")
print(f"Fracción reflejada reciclada respecto a lo emitido: {P_reflejada_reciclada/P_emisor:.2%}")
print(f"Fracción reincidente (backplate) respecto a lo emitido: {P_reincidente/P_emisor:.2%}")
print(f"Eficiencia TPV: {Eficiencia_TPV:.2%}")
#print (lambda_g)

# =======================
# ANOTACIÓN IMPORTANTE PARA TU MEMORIA:
# -----------------------------------------------------------
# La eficiencia del sistema TPV se define aquí como la razón entre la potencia absorbida útil por la celda
# (la única disponible para generar electricidad) y el "gasto neto" de radiación del emisor,
# es decir, la potencia total emitida menos la radiación que regresa al emisor tanto por reflexión superficial
# como por el reciclaje del backplate.
# Esta definición es más estricta y realista que la eficiencia TPV convencional y refleja el
# aprovechamiento energético efectivo del sistema completo.
# -----------------------------------------------------------

# =======================
# OPCIONAL: PLOT (para visualización en TFM)
# =======================
plt.figure(figsize=(8,5))
plt.plot(wavelengths*1e6, E_lambda/E_lambda.max(), label="Espectro Emisor (Norm.)")
plt.axvline(lambda_g*1e6, color='red', linestyle='--', label=r'$\lambda_g$')
plt.fill_between(wavelengths[mask_util]*1e6, 0, (E_lambda* A_vals * EQE)[mask_util]/E_lambda.max(), 
                alpha=0.3, color='orange', label="Absorbido útil")
plt.xlabel("Longitud de onda [μm]")
plt.ylabel("Irradiancia relativa")
plt.legend()
plt.title("Distribución espectral y absorción útil TPV")
plt.tight_layout()
plt.show()
