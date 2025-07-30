# ray_tpv.py
"""
Utilidad para generar rayos TPV con potencia y longitud de onda asociadas
según la ley de Planck para un emisor térmico realista. Permite definir
temperatura, área del emisor, rango espectral y número de rayos a simular.

Cada rayo (RayTPV) lleva:
 - Origen y dirección (pueden ser modificados por el usuario o la escena)
 - Longitud de onda (lambda)
 - Potencia asociada, correctamente dimensionada según el área y la física

Se puede importar la función generar_rayos_planck() y la clase RayTPV
desde cualquier main de simulación TPV.
"""

import numpy as np
from scipy.constants import h, c, k

class RayTPV:
    def __init__(self, origen, direccion, lambda_m, potencia):
        """
        Inicializa un rayo TPV.
        :param origen: Vector origen del rayo (tuple/list de 3 floats)
        :param direccion: Vector dirección del rayo (normalizado)
        :param lambda_m: Longitud de onda del rayo (en metros)
        :param potencia: Potencia transportada por el rayo (en W)
        """
        self.origen = origen
        self.direccion = direccion
        self.lambda_m = lambda_m
        self.potencia = potencia  # ¡NO energía, sino potencia!

    def __str__(self):
        return f"RayTPV(l={self.lambda_m*1e6:.2f} μm, P={self.potencia:.2e} W, origen={self.origen})"

def E_BB_lambda(lambda_m, T):
    """
    Ley de Planck: Irradiancia espectral [W/m^2/m] para una longitud de onda y T dada.
    """
    exponent = h * c / (lambda_m * k * T)
    return (2 * np.pi * h * c**2) / (lambda_m**5 * (np.exp(exponent) - 1))

def generar_rayos_planck(T_emisor, A_emisor, lambda_min, lambda_max, num_rayos, 
                        origen_default=(0,0,0), direccion_default=(0,0,1)):
    """
    Genera una lista de RayTPV, cada uno con su lambda y potencia correctamente asignada
    según el espectro de Planck, el área del emisor y el ancho espectral.
    :param T_emisor: Temperatura del emisor [K]
    :param A_emisor: Área del emisor [m^2]
    :param lambda_min: Longitud de onda mínima [m]
    :param lambda_max: Longitud de onda máxima [m]
    :param num_rayos: Número de rayos (bins espectrales)
    :param origen_default: Vector origen de los rayos (puede ser sobrescrito)
    :param direccion_default: Vector dirección (puede ser sobrescrito)
    :return: Lista de RayTPV (potencia en W)
    """
    lambdas = np.linspace(lambda_min, lambda_max, num_rayos)
    delta_lambda = (lambda_max - lambda_min) / (num_rayos - 1)
    E_lambda = E_BB_lambda(lambdas, T_emisor)  # Vector de irradiancia espectral
    rayos = []
    for i in range(num_rayos):
        potencia_rayo = E_lambda[i] * A_emisor * delta_lambda  # [W/m^2/m * m^2 * m = W]
        rayos.append(RayTPV(
            origen=origen_default,
            direccion=direccion_default,
            lambda_m=lambdas[i],
            potencia=potencia_rayo
        ))
    return rayos

# =====================
# Ejemplo de uso rápido
# =====================
if __name__ == "__main__":
    T = 2000         # Temperatura del emisor en K
    A = 0.01         # Área del emisor en m^2
    lmin = 0.5e-6    # 0.5 micras
    lmax = 3.0e-6    # 3 micras
    N = 100          # Número de rayos/bins
    rayos = generar_rayos_planck(T, A, lmin, lmax, N)
    for r in rayos[:5]:  # Mostrar primeros 5 rayos
        print(r)
