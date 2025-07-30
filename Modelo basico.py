# MODELO BÁSICO TPV - CON ANÁLISIS Y GRÁFICAS DE SENSIBILIDAD
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------- CONSTANTES FÍSICAS --------------------
SIGMA = 5.67e-8  # Constante de Stefan-Boltzmann (W/m^2/K^4)
NAIRE = 1.0003   # Índice de refracción del aire

# -------------------- FUNCIONES FÍSICAS ---------------------
def calcular_Reflex_Absorcion_Transmision(incidencia, ncelda, absorcion_fraccion):
    """Devuelve R, T, A (reflectancia, transmitancia, absorción efectiva) para la celda TPV."""
    radianes = math.radians(incidencia)
    sin_ang_trans = (NAIRE / ncelda) * math.sin(radianes)
    if abs(sin_ang_trans) > 1:
        R = 1.0; A = 0.0; T = 0.0
        return R, T, A
    angulotransmision = math.asin(sin_ang_trans)
    reflex_perpendicular = ((NAIRE * math.cos(radianes) - ncelda * math.cos(angulotransmision)) /
                            (NAIRE * math.cos(radianes) + ncelda * math.cos(angulotransmision))) ** 2
    reflex_paralelo = ((ncelda * math.cos(radianes) - NAIRE * math.cos(angulotransmision)) /
                       (ncelda * math.cos(radianes) + NAIRE * math.cos(angulotransmision))) ** 2
    R = (reflex_perpendicular + reflex_paralelo) / 2
    A = absorcion_fraccion * (1 - R)
    T = (1 - absorcion_fraccion) * (1 - R)
    return R, T, A

def calcular_Qemisor(emit, Aemi, Temi):
    """Potencia total emitida por el emisor (W)."""
    return emit * SIGMA * Aemi * (Temi ** 4) 

def calcular_Qfiltrado(Qemisor, Tlambda, Fve):
    """Potencia tras filtro espectral o selectivo."""
    return Qemisor * Tlambda * Fve

def calcular_Qabs(Qfiltrado, A):
    """Potencia absorbida en la celda."""
    return Qfiltrado * A

def calcular_Pout(Qabs, ncuanti):
    """Potencia eléctrica potencialmente generada (rendimiento cuántico ideal)."""
    return Qabs * ncuanti

def calcular_Qnoentra(Qfiltrado, R):
    """Potencia reflejada en la celda, que no entra en la celda."""
    return Qfiltrado * R

def calcular_Qatravesado(Qfiltrado, T):
    """Potencia transmitida a través de la celda (hacia el backplate)."""
    return Qfiltrado * T

def calcular_Qreincidente(Qatravesado, Rbp):
    """Potencia reflejada por el backplate hacia el emisor."""
    return Qatravesado * Rbp

def calcular_eficiencia_tpv(Qemisor, Qreincidente, Qabs):
    """Eficiencia óptica: energía útil absorbida / energía neta emitida."""
    if (Qemisor - Qreincidente) > 0:
        return Qabs / (Qemisor - Qreincidente)
    else:
        return 0

# -------------------- PARÁMETROS POR DEFECTO ----------------
params = {
    'emit': 1.0,
    'Aemi': 0.01,
    'Temi': 2000,
    'Fve': 0.85,
    'Tlambda': 0.4,
    'ncuanti': 0.8,
    'Rbp': 0.9,
    'ncelda': 3.5,
    'incidencia': 0,
    'absorcion_fraccion': 0.8,
}

# -------------------- FUNCIÓN DE CÁLCULO PRINCIPAL -----------------
def calcular_resultados(params):
    R, T, A = calcular_Reflex_Absorcion_Transmision(params['incidencia'], params['ncelda'], params['absorcion_fraccion'])
    Qemisor = calcular_Qemisor(params['emit'], params['Aemi'], params['Temi'])
    Qfiltrado = calcular_Qfiltrado(Qemisor, params['Tlambda'], params['Fve'])
    Qabs = calcular_Qabs(Qfiltrado, A)
    Pout = calcular_Pout(Qabs, params['ncuanti'])
    Qnoentra = calcular_Qnoentra(Qfiltrado, R)
    Qatravesado = calcular_Qatravesado(Qfiltrado, T)
    Qreincidente = calcular_Qreincidente(Qatravesado, params['Rbp'])
    efitpv = calcular_eficiencia_tpv(Qemisor, Qreincidente, Qabs)
    return {
        "R": R,
        "T": T,
        "A": A,
        "Qemisor": Qemisor,
        "Qfiltrado": Qfiltrado,
        "Qabs": Qabs,
        "Pout": Pout,
        "Qnoentra": Qnoentra,
        "Qatravesado": Qatravesado,
        "Qreincidente": Qreincidente,
        "efitpv": efitpv
    }

# -------------------- VISUALIZACIÓN DE RESULTADOS ------------------
def print_resultados(res):
    print("\n----- RESULTADOS TPV BÁSICO -----")
    print(f"Reflectancia (R):          {res['R']:.4f}")
    print(f"Transmitancia (T):         {res['T']:.4f}")
    print(f"Absorción efectiva (A):    {res['A']:.4f}")
    print(f"Radiación emitida por el emisor: :         {res['Qemisor']:.2f} W")
    print(f"Radiación filtrada que llega a la celda:   {res['Qfiltrado']:.2f} W")
    print(f"Radiación absorbida por la celda:          {res['Qabs']:.2f} W")
    print(f"Potencia útil generada por el sistema:    {res['Pout']:.2f} W")
    print(f"Potencia reflejada por la celda:      {res['Qnoentra']:.2f} W")
    print(f"Potencia transmitida a través de la celda: {res['Qatravesado']:.2f} W")
    print(f"Potencia reincidente en el emisor (reciclaje):  {res['Qreincidente']:.2f} W")
    print(f"Eficiencia TPV:            {res['efitpv']:.2%}")

# -------------------- SENSIBILIDAD Y GRÁFICAS ---------------------
def graficar_sensibilidad(param_name, param_range, params_base, resultado='efitpv'):
    resultados = []
    for valor in param_range:
        params_local = params_base.copy()
        params_local[param_name] = valor
        res = calcular_resultados(params_local)
        resultados.append(res[resultado])

    plt.figure(figsize=(7,4))
    plt.plot(param_range, resultados, marker='o')
    plt.xlabel(param_name)
    plt.ylabel(resultado)
    plt.title(f'Sensibilidad de {resultado} respecto a {param_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_multi(param_name, param_range, params_base, result_keys=('Qemisor','Qabs','Qreincidente','efitpv')):
    result_dict = {key: [] for key in result_keys}
    for valor in param_range:
        params_local = params_base.copy()
        params_local[param_name] = valor
        res = calcular_resultados(params_local)
        for key in result_keys:
            result_dict[key].append(res[key])
    plt.figure(figsize=(10,6))
    for key in result_keys:
        plt.plot(param_range, result_dict[key], label=key)
    plt.xlabel(param_name)
    plt.title(f'Sensibilidad multi-variable respecto a {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- EJECUCIÓN PRINCIPAL ------------------------
if __name__ == "__main__":
    # Cálculo y print de los resultados básicos
    resultados = calcular_resultados(params)
    print_resultados(resultados)

    # # Ejemplo 1: sensibilidad de la eficiencia TPV frente a ángulo de incidencia
    # angulos = np.linspace(0, 80, 16)
    # graficar_sensibilidad('incidencia', angulos, params, resultado='efitpv')

    # # Ejemplo 2: análisis multi-variable (absorción)
    # absorciones = np.linspace(0.1, 1.0, 20)
    # graficar_multi('absorcion_fraccion', absorciones, params, result_keys=('Qabs','Qreincidente','efitpv'))

    # Puedes descomentar para explorar otras variables fácilmente:
    # graficar_sensibilidad('Rbp', np.linspace(0.7, 1.0, 10), params, resultado='efitpv')
    # graficar_multi('Tlambda', np.linspace(0.1, 1.0, 20), params, result_keys=('Qemisor','Qabs','Qreincidente','efitpv'))

def graficar_eficiencia_vs_temperatura(params_base):
    temperaturas = np.linspace(1000, 3000, 25)
    eficiencias = []
    for T in temperaturas:
        params_local = params_base.copy()
        params_local['Temi'] = T
        res = calcular_resultados(params_local)
        eficiencias.append(res['efitpv'])
    plt.figure(figsize=(7,4))
    plt.plot(temperaturas, np.array(eficiencias)*100, marker='o')
    plt.xlabel('Temperatura del emisor (K)')
    plt.ylabel('Eficiencia TPV (%)')
    plt.title('Eficiencia TPV vs Temperatura del emisor')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_qemisor_vs_temperatura(params_base):
    temperaturas = np.linspace(1000, 3000, 25)
    qemisores = []
    for T in temperaturas:
        params_local = params_base.copy()
        params_local['Temi'] = T
        res = calcular_resultados(params_local)
        qemisores.append(res['Qemisor'])
    plt.figure(figsize=(7,4))
    plt.plot(temperaturas, qemisores, marker='o')
    plt.xlabel('Temperatura del emisor (K)')
    plt.ylabel('Qemisor (W)')
    plt.title('Radiación total emitida por el emisor vs Temperatura')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_qabs_vs_incidencia(params_base):
    angulos = np.linspace(0, 90, 25)
    qabs = []
    for inc in angulos:
        params_local = params_base.copy()
        params_local['incidencia'] = inc
        res = calcular_resultados(params_local)
        qabs.append(res['Qabs'])
    plt.figure(figsize=(7,4))
    plt.plot(angulos, qabs, marker='o')
    plt.xlabel('Ángulo de incidencia (°)')
    plt.ylabel('Potencia absorbida Qabs (W)')
    plt.title('Potencia útil absorbida vs Ángulo de incidencia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Llama a las funciones
graficar_eficiencia_vs_temperatura(params)
graficar_qemisor_vs_temperatura(params)
graficar_qabs_vs_incidencia(params)
