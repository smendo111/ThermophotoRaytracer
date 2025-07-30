import numpy as np
import matplotlib.pyplot as plt
from material_tpv import MaterialTPV
from ray_tpv import RayTPV

# ==== Geometría semiesfera (3D) ====
R = 0.4           # Radio de la semiesfera (m)
R_backplate = 0.47  # Radio del backplate, ligeramente mayor (m)
A_emisor = np.pi * R ** 2   # Área de la base circular en 3D

tam_punto = 2

# ==== Materiales ====
celda = MaterialTPV()  # Usa tus parámetros
backplate = MaterialTPV(
    Eg_eV=0, reflectancia_subbg=0.99, EQE=0, nombre="Backplate"
)
emisor = MaterialTPV(
    Eg_eV=0, reflectancia_subbg=0, EQE=0, nombre="Emisor"
)

# ==== Parámetros ====
T_emisor = 2000
num_rayos = 10000
max_rebotes = 450
umbral_potencia = 1e-15

lambda_min = 0.5e-6
lambda_max = 5.0e-6

# ==== Acumuladores ====
absorbed_celda = 0.0
absorbed_backplate = 0.0
absorbed_emisor = 0.0
potencia_no_acumulada = 0.0

impactos_celda_2d = []
impactos_backplate_2d = []
impactos_emisor_2d = []

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
        potencia = E_lambda[i] * A_emisor * delta_lambda
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

def interseccion_esfera(pos, dir, R):
    # pos, dir: arrays de 3 elementos
    a = np.dot(dir, dir)
    b = 2 * np.dot(pos, dir)
    c = np.dot(pos, pos) - R ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    t1 = (-b + np.sqrt(disc)) / (2 * a)
    t2 = (-b - np.sqrt(disc)) / (2 * a)
    ts = [t for t in [t1, t2] if t > 1e-8]
    if not ts:
        return None
    # De los posibles t, queremos el más cercano
    return min(ts)

def normal_en_esfera(x, y, z):
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.array([x, y, z]) / norm

for rayo in rayos:
    # Origen: punto aleatorio en base circular (z=0)
    r_rand = R * np.sqrt(np.random.uniform(0, 1))
    theta_rand = np.random.uniform(0, 2 * np.pi)
    x0 = r_rand * np.cos(theta_rand)
    y0 = r_rand * np.sin(theta_rand)
    z0 = 0.0
    pos = np.array([x0, y0, z0])

    # Dirección aleatoria hemisférica hacia arriba (dentro de la semiesfera)
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    phi = 2 * np.pi * u
    costheta = np.random.uniform(0, 1)
    sintheta = np.sqrt(1 - costheta ** 2)
    dir = np.array([
        sintheta * np.cos(phi),
        sintheta * np.sin(phi),
        costheta
    ])
    dir = dir / np.linalg.norm(dir)  # Por si acaso

    potencia = rayo.potencia
    lambda_m = rayo.lambda_m

    for rebote in range(max_rebotes):
        # Intersección con la semiesfera (celda, backplate)
        t_celda = interseccion_esfera(pos, dir, R)
        t_bp = interseccion_esfera(pos, dir, R_backplate)
        # Intersección con base circular (z=0)
        t_base = None
        if dir[2] != 0:
            t_candidate = (0.0 - pos[2]) / dir[2]
            x_base = pos[0] + t_candidate * dir[0]
            y_base = pos[1] + t_candidate * dir[1]
            if t_candidate > 1e-8 and (x_base ** 2 + y_base ** 2 <= R ** 2):
                t_base = t_candidate

        ts = []
        labels = []
        if t_celda is not None:
            ts.append(t_celda)
            labels.append('celda')
        if t_bp is not None:
            ts.append(t_bp)
            labels.append('backplate')
        if t_base is not None:
            ts.append(t_base)
            labels.append('emisor')

        if not ts:
            potencia_no_acumulada += potencia
            break

        t_min = min(ts)
        label_min = labels[ts.index(t_min)]
        nueva_pos = pos + t_min * dir
        x, y, z = nueva_pos

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
            n = normal_en_esfera(x, y, z)
            dir = dir - 2 * np.dot(dir, n) * n
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        elif label_min == 'backplate':
            abs_frac = (1 - backplate.reflectancia_subbg)
            potencia_abs = potencia * abs_frac
            potencia_refl = potencia * backplate.reflectancia_subbg
            absorbed_backplate += potencia_abs
            impactos_backplate_2d.append((x, z))
            potencia = potencia_refl
            n = normal_en_esfera(x, y, z)
            dir = dir - 2 * np.dot(dir, n) * n
            pos = nueva_pos
            if potencia < umbral_potencia:
                potencia_no_acumulada += potencia
                break
            continue

        elif label_min == 'emisor':
            absorbed_emisor += potencia
            impactos_emisor_2d.append((x, z))
            break

        if potencia < umbral_potencia:
            potencia_no_acumulada += potencia
            break

        pos = nueva_pos

# ==== Balance de energía ====
total_absorbida = absorbed_celda + absorbed_backplate
total_reciclada = absorbed_emisor
balance = total_absorbida + total_reciclada + potencia_no_acumulada

print("===== BALANCE DE ENERGÍA – SEMIESFERA 3D =====")
print(f"Potencia absorbida por celda TPV:           {absorbed_celda:.3f} W")
print(f"Potencia absorbida en backplate:            {absorbed_backplate:.3f} W")
print(f"Potencia reciclada (emisor):                {absorbed_emisor:.3f} W")
print(f"Potencia no absorbida ni reciclada:         {potencia_no_acumulada:.3f} W")
print(f"\nPotencia total emitida:                     {potencia_total_emitida:.3f} W")
print(f"Suma de acumuladores:                       {balance:.3f} W")
print(f"Diferencia (debe ser ≈ 0):                  {(potencia_total_emitida - balance):.3f} W")

if potencia_total_emitida - absorbed_emisor > 0:
    eficiencia_tpv = absorbed_celda / (potencia_total_emitida - absorbed_emisor)
else:
    eficiencia_tpv = 0.0
print(f"\nEficiencia TPV del sistema: {eficiencia_tpv:.2%}")

# ==== Visualización ====
theta = np.linspace(0, np.pi, 300)
x_celda = R * np.cos(theta)
z_celda = -R * np.sin(theta)
x_bp = R_backplate * np.cos(theta)
z_bp = -R_backplate * np.sin(theta)

plt.figure(figsize=(9, 7))
""" plt.plot([-R, R], [0, 0], color='orange', linewidth=8, label='Emisor térmico (base)')
plt.plot(x_celda, z_celda, color='blue', linestyle='--', linewidth=2, label='Celda TPV')
plt.plot(x_bp, z_bp, color='green', linestyle='-.', linewidth=2, label='Backplate')
 """
if impactos_celda_2d:
    x, z = zip(*impactos_celda_2d)
    plt.scatter(x, z, s=tam_punto, color='blue', alpha=0.7, label="Impactos Celda TPV")
if impactos_backplate_2d:
    x, z = zip(*impactos_backplate_2d)
    plt.scatter(x, z, s=tam_punto, color='green', alpha=0.7, label="Impactos Backplate")
if impactos_emisor_2d:
    x, z = zip(*impactos_emisor_2d)
    plt.scatter(x, z, s=tam_punto+2, color='orange', alpha=0.9, label="Impactos Emisor")

plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.gca().set_aspect('equal')
plt.title("Simulación TPV semiesférica 3D – Proyección de impactos X-Z")
plt.legend(markerscale=1.5, fontsize=10, loc='lower right')
plt.tight_layout()
plt.show()
