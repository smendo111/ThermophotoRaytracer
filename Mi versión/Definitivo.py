import numpy as np
import matplotlib.pyplot as plt
from material_tpv import MaterialTPV
from ray_tpv import RayTPV

# ========== Geometría del prisma rectangular ==========
ancho = 1   # metros (X)
profundo = 0.5  # metros (Y)
alto = 0.1   # metros (Z)

A_emisor = 0.01
L_emisor = np.sqrt(A_emisor)
x0_emisor = (ancho - L_emisor) / 2
x1_emisor = (ancho + L_emisor) / 2
y0_emisor = (profundo - L_emisor) / 2
y1_emisor = (profundo + L_emisor) / 2

pos_celda = 0.05  # Altura de la celda (en Z, mitad de la caja)
pos_backplate = 0.049  # Altura del backplate

tam_punto = 1

# ========== Materiales ==========
celda = MaterialTPV()
backplate = MaterialTPV(
    Eg_eV=0, reflectancia_subbg=0.99, EQE=0, nombre="Backplate reflector"
)
pared = MaterialTPV(
    Eg_eV=0, reflectancia_subbg=0.97, EQE=0, nombre="Pared metálica reflectante"
)
emisor = MaterialTPV(
    Eg_eV=0, reflectancia_subbg=0, EQE=0, nombre="Emisor térmico"
)
emisor.potencia_reciclada = 0.0

# ========== Parámetros de simulación ==========
T_emisor = 2000
num_rayos = 1000
max_rebotes = 450
umbral_potencia = 1e-15

lambda_min = 0.5e-6
lambda_max = 5.0e-6

# ========== Acumuladores ==========
absorbed_celda = 0.0
absorbed_backplate = 0.0
absorbed_paredes = 0.0
absorbed_emisor = 0.0
potencia_no_acumulada = 0.0

# ========== Listas de impactos (para visualización 2D X-Z) ==========
impactos_celda_2d = []
impactos_backplate_2d = []
impactos_pared_2d = []
impactos_emisor_2d = []

# ========== Rayos desde la franja del emisor (cara z=alto, centrada en X-Y) ==========
def generar_rayos_planck_3d(T_emisor, A_emisor, lambda_min, lambda_max, num_rayos):
    lambdas = np.linspace(lambda_min, lambda_max, num_rayos)
    delta_lambda = (lambda_max - lambda_min) / (num_rayos - 1)
    from scipy.constants import h, c, k
    def E_BB_lambda(l, T):
        exponent = h * c / (l * k * T)
        return (2 * np.pi * h * c ** 2) / (l ** 5 * (np.exp(exponent) - 1))
    E_lambda = E_BB_lambda(lambdas, T_emisor)
    rayos = []
    for i in range(num_rayos):
        potencia = E_lambda[i] * A_emisor * delta_lambda  # [W]
        rayos.append(RayTPV(
            origen=(0, 0, 0),
            direccion=(0, 0, 0),
            lambda_m=lambdas[i],
            potencia=potencia
        ))
    return rayos

rayos = generar_rayos_planck_3d(
    T_emisor, A_emisor, lambda_min, lambda_max, num_rayos
)
potencia_total_emitida = sum(r.potencia for r in rayos)

def direccion_aleatoria_hemiesfera():
    theta = np.arccos(np.random.uniform(0, 1))
    phi = np.random.uniform(0, 2 * np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = -abs(np.cos(theta))
    return [x, y, z]

for rayo in rayos:
    x0 = np.random.uniform(x0_emisor, x1_emisor)
    y0 = np.random.uniform(y0_emisor, y1_emisor)
    z0 = alto
    pos = [x0, y0, z0]
    dir = direccion_aleatoria_hemiesfera()
    potencia = rayo.potencia
    lambda_m = rayo.lambda_m

    for rebote in range(max_rebotes):
        if abs(dir[0]) < 1e-12: dir[0] = 1e-12
        if abs(dir[1]) < 1e-12: dir[1] = 1e-12
        if abs(dir[2]) < 1e-12: dir[2] = 1e-12

        ts = []
        labels = []

        # Celda/backplate (z=pos_celda, z=pos_backplate)
        t_celda = (pos_celda - pos[2]) / dir[2]
        if t_celda > 1e-8:
            ts.append(t_celda)
            labels.append('celda')
        t_backplate = (pos_backplate - pos[2]) / dir[2]
        if t_backplate > 1e-8:
            ts.append(t_backplate)
            labels.append('backplate')

        # Suelo/techo
        t_techo = (alto - pos[2]) / dir[2]
        t_suelo = (0.0 - pos[2]) / dir[2]
        if t_techo > 1e-8:
            ts.append(t_techo)
            labels.append('techo')
        if t_suelo > 1e-8:
            ts.append(t_suelo)
            labels.append('suelo')

        # Paredes x=0/x=ancho
        t_x0 = (0.0 - pos[0]) / dir[0]
        t_x1 = (ancho - pos[0]) / dir[0]
        if t_x0 > 1e-8:
            ts.append(t_x0)
            labels.append('pared')
        if t_x1 > 1e-8:
            ts.append(t_x1)
            labels.append('pared')

        # Paredes y=0/y=profundo
        t_y0 = (0.0 - pos[1]) / dir[1]
        t_y1 = (profundo - pos[1]) / dir[1]
        if t_y0 > 1e-8:
            ts.append(t_y0)
            labels.append('pared')
        if t_y1 > 1e-8:
            ts.append(t_y1)
            labels.append('pared')

        if not ts:
            potencia_no_acumulada += potencia
            break

        t_min = min(ts)
        label_min = labels[ts.index(t_min)]
        nueva_pos = [pos[i] + t_min * dir[i] for i in range(3)]
        x, y, z = nueva_pos

        # --- Celda TPV (z=pos_celda, toda la cara X-Y) ---
        if label_min == 'celda':
            lambda_g = 1.24e-6 / celda.Eg_eV if celda.Eg_eV else 0
            if lambda_g and lambda_m < lambda_g:
                abs_frac = celda.absorptancia_util * celda.EQE
                potencia_abs = potencia * abs_frac
                potencia_refl = potencia * (1 - abs_frac)
                absorbed_celda += potencia_abs
                impactos_celda_2d.append((x, z))
                potencia = potencia_refl
            else:
                abs_frac = (1 - backplate.reflectancia_subbg)
                potencia_abs = potencia * abs_frac
                potencia_refl = potencia * backplate.reflectancia_subbg
                absorbed_backplate += potencia_abs
                impactos_backplate_2d.append((x, z))
                potencia = potencia_refl
            dir[2] *= -1
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        # --- Backplate (z=pos_backplate, toda la cara X-Y) ---
        elif label_min == 'backplate':
            abs_frac = (1 - backplate.reflectancia_subbg)
            potencia_abs = potencia * abs_frac
            potencia_refl = potencia * backplate.reflectancia_subbg
            absorbed_backplate += potencia_abs
            impactos_backplate_2d.append((x, z))
            potencia = potencia_refl
            dir[2] *= -1
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        # --- Techo (z=alto, sólo la franja central es emisor) ---
        elif label_min == 'techo':
            if (x0_emisor <= x <= x1_emisor) and (y0_emisor <= y <= y1_emisor):
                absorbed_emisor += potencia
                impactos_emisor_2d.append((x, z))
                break
            else:
                abs_frac = (1 - pared.reflectancia_subbg)
                potencia_abs = potencia * abs_frac
                potencia_refl = potencia * pared.reflectancia_subbg
                absorbed_paredes += potencia_abs
                impactos_pared_2d.append((x, z))
                potencia = potencia_refl
                dir[2] *= -1
                pos = nueva_pos
                if potencia < umbral_potencia:
                    potencia_no_acumulada += potencia
                    break
                continue

        # --- Suelo (z=0) ---
        elif label_min == 'suelo':
            abs_frac = (1 - pared.reflectancia_subbg)
            potencia_abs = potencia * abs_frac
            potencia_refl = potencia * pared.reflectancia_subbg
            absorbed_paredes += potencia_abs
            impactos_pared_2d.append((x, z))
            potencia = potencia_refl
            dir[2] *= -1
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        # --- Paredes laterales x/y ---
        elif label_min == 'pared':
            abs_frac = (1 - pared.reflectancia_subbg)
            potencia_abs = potencia * abs_frac
            potencia_refl = potencia * pared.reflectancia_subbg
            absorbed_paredes += potencia_abs
            impactos_pared_2d.append((x, z))
            # Rebote especular según qué dirección cambie más:
            if abs((x <= 1e-8) or (x >= ancho - 1e-8)):
                dir[0] *= -1
            elif abs((y <= 1e-8) or (y >= profundo - 1e-8)):
                dir[1] *= -1
            potencia = potencia_refl
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        if potencia < umbral_potencia:
            potencia_no_acumulada += potencia
            break

        pos = nueva_pos

# ========== Balance de energía ==========
total_absorbida = absorbed_celda + absorbed_backplate + absorbed_paredes
total_reciclada = absorbed_emisor
balance = total_absorbida + total_reciclada + potencia_no_acumulada

print("===== BALANCE DE ENERGÍA EN PRISMA RECTANGULAR =====")
print(f"Potencia elétrica generada por la celda TPV:           {absorbed_celda:.3f} W")
print(f"Radiación absorbida en backplate:            {absorbed_backplate:.3f} W")
print(f"Radiación absorbida en paredes:              {absorbed_paredes:.3f} W")
print(f"Radiación reciclada (emisor):                {absorbed_emisor:.3f} W")
print(f"Radiación no absorbida ni reciclada:         {potencia_no_acumulada:.3f} W")
print(f"\nRadiación total emitida:                     {potencia_total_emitida:.3f} W")
print(f"Suma de acumuladores:                       {balance:.3f} W")
print(f"Diferencia (debe ser ≈ 0):                  {(potencia_total_emitida - balance):.3f} W")

# ========== Eficiencia ==========
if potencia_total_emitida - absorbed_emisor > 0:
    eficiencia_tpv = absorbed_celda / (potencia_total_emitida - absorbed_emisor)
else:
    eficiencia_tpv = 0.0
print(f"\nEficiencia TPV del sistema: {eficiencia_tpv:.2%}")

# ========== Visualización 2D (X-Z) ==========
plt.figure(figsize=(8, 8))
""" plt.plot([0, ancho], [0, 0], color='black', linewidth=2)      # Suelo (z=0)
plt.plot([0, ancho], [alto, alto], color='grey', linewidth=2) # Techo (z=alto)
plt.plot([0, 0], [0, alto], color='black', linewidth=2)       # Pared izq
plt.plot([ancho, ancho], [0, alto], color='black', linewidth=2) # Pared dcha

plt.plot([x0_emisor, x1_emisor], [alto, alto], color='purple', linewidth=4, label="Emisor térmico")
plt.plot([0, ancho], [pos_celda, pos_celda], color='blue', linestyle='--', linewidth=2, label="Celda TPV")
plt.plot([0, ancho], [pos_backplate, pos_backplate], color='green', linestyle='-.', linewidth=2, label="Backplate")
 """
if impactos_celda_2d:
    x, z = zip(*impactos_celda_2d)
    plt.scatter(x, z, s=tam_punto, color='blue', alpha=0.7, label="Impactos Celda TPV")
if impactos_backplate_2d:
    x, z = zip(*impactos_backplate_2d)
    plt.scatter(x, z, s=tam_punto, color='green', alpha=0.7, label="Impactos Backplate")
if impactos_pared_2d:
    x, z = zip(*impactos_pared_2d)
    plt.scatter(x, z, s=tam_punto, color='red', alpha=0.6, label="Impactos Pared")
if impactos_emisor_2d:
    x, z = zip(*impactos_emisor_2d)
    plt.scatter(x, z, s=tam_punto+2, color='purple', alpha=0.8, label="Impactos Emisor")

plt.title("Simulación TPV – Distribución de impactos de rayos (X-Z)")
plt.xlabel("Coordenada X (m)")
plt.ylabel("Altura Z (m)")
plt.xlim(-0.01, ancho + 0.01)
plt.ylim(-0.01, alto + 0.01)
plt.legend(markerscale=1.6, fontsize=9, loc='lower right')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
