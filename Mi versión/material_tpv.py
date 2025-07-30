# material_tpv.py
"""
Definición de la clase MaterialTPV para simulación en un raytracer
con propiedades espectrales y físicas inspiradas en el artículo
"Thermophotovoltaic efficiency of 40%" (Henry et al., Nature 2022).

Esta clase representa un material TPV realista con bandgap, absorptancia
útil, reflectancia sub-bandgap, eficiencia cuántica externa, etc.
Los valores por defecto corresponden a la celda tándem 1.4/1.2 eV.

A este archivo se irán añadiendo métodos y lógica para la interacción rayo-material
y el conteo de energía absorbida, reflejada y transmitida en futuros pasos.
"""

class MaterialTPV:
    def __init__(self,
                 Eg_eV=1,                # Bandgap en eV (puede ser 1.2 para otra celda)
                 absorptancia_util=0.8,    # Absorptancia para lambda < lambda_g (literatura)
                 reflectancia_subbg=0.93,  # Reflectancia para lambda > lambda_g (literatura)
                 EQE=0.8,                  # Eficiencia cuántica externa (simplificada)
                 n_celda=3.5,              # Índice de refracción del material (literatura)
                 nombre="Tandem III-V TPV cell 1.4/1.2 eV"
                 ):
        """
        Inicializa el material TPV con parámetros físicos realistas.
        :param Eg_eV: Bandgap en electronvoltios
        :param absorptancia_util: Fracción absorbida (lambda < lambda_g)
        :param reflectancia_subbg: Reflectancia sub-bandgap (lambda > lambda_g)
        :param EQE: Eficiencia cuántica externa (lambda < lambda_g)
        :param n_celda: Índice de refracción (aprox. 3.5 para III-V)
        :param nombre: Etiqueta para el material
        """
        self.Eg_eV = Eg_eV
        self.absorptancia_util = absorptancia_util
        self.reflectancia_subbg = reflectancia_subbg
        self.EQE = EQE
        self.n_celda = n_celda
        self.nombre = nombre

        # Inicializar acumuladores de energía (se usarán para sumar
        # la energía absorbida, reflejada y transmitida por este material)
        self.energia_absorbida = 0.0
        self.energia_reflejada = 0.0
        self.energia_transmitida = 0.0

    def __str__(self):
        return f"MaterialTPV: {self.nombre} (Eg={self.Eg_eV}eV)"

    # Métodos de interacción y lógica TPV se añadirán en pasos futuros
