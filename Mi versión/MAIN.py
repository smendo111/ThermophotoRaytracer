import numpy as np
import matplotlib.pyplot as plt
from material_tpv import MaterialTPV
from ray_tpv import RayTPV, generar_rayos_planck

# ====== Parámetros de simulación ======
T_emisor = 2000
A_emisor = 0.01       # Área total del emisor (m^2)
ancho_caja = 1        # Ancho y profundidad de la caja (m)
altura_caja = 0.5      # Altura total de la caja (m)

pos_emisor = altura_caja
pos_celda = 0.25        # Mitad de la caja
pos_backplate = 0.20   # Justo debajo de la celda

num_rayos = 1000       # Número de rayos a simular
max_rebotes = 100
tam_punto = 3         # Tamaño de los puntos en la imagen

lambda_min = 0.5e-6
lambda_max = 5.0e-6

# ====== Definición de materiales ======
celda = MaterialTPV()
backplate = MaterialTPV(
    Eg_eV=0,
    absorptancia_util= 1,
    reflectancia_subbg= 0,
    EQE=0,
    nombre="Backplate reflector"
)
pared = MaterialTPV(
    Eg_eV=0,
    absorptancia_util= 1,
    reflectancia_subbg= 0,
    EQE=0,
    nombre="Pared metálica reflectante"
)
emisor = MaterialTPV(
    Eg_eV=0,
    absorptancia_util=1,
    reflectancia_subbg=0,
    EQE=0,
    nombre="Emisor térmico"
)
emisor.potencia_reciclada = 0.0  # Acumulador en Watios

# ====== Listas para impactos de rayos (para visualización) ======
impactos_celda = []
impactos_backplate = []
impactos_pared = []
impactos_emisor = []

# ====== Cálculo de la franja del emisor centrada ======
L_emisor = np.sqrt(A_emisor)
x0_emisor = (ancho_caja - L_emisor) / 2
x1_emisor = (ancho_caja + L_emisor) / 2

# ====== Generación de rayos SOLO desde la franja del emisor ======
rayos = generar_rayos_planck(
    T_emisor, A_emisor, lambda_min, lambda_max, num_rayos
)

def direccion_aleatoria_hemiesfera():
    """Vector aleatorio en hemiesfera inferior para emisión difusa."""
    theta = np.arccos(np.random.uniform(0, 1))  # [0, pi/2]
    phi = np.random.uniform(0, 2*np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = -np.abs(np.cos(theta))
    return (x, y, z)

for rayo in rayos:
    # POSICIÓN ALEATORIA EN LA FRANJA DEL EMISOR
    x0 = np.random.uniform(x0_emisor, x1_emisor)
    y0 = np.random.uniform(0, ancho_caja)  # Profundidad completa
    pos = [x0, y0, pos_emisor]
    dir = list(direccion_aleatoria_hemiesfera())
    potencia = rayo.potencia
    lambda_m = rayo.lambda_m

    for rebote in range(max_rebotes):
        if dir[2] == 0:
            break

        # Intersección con planos horizontales
        t_celda = (pos_celda - pos[2]) / dir[2]
        t_backplate = (pos_backplate - pos[2]) / dir[2]
        t_emisor = (altura_caja - pos[2]) / dir[2]
        t_suelo = (0.0 - pos[2]) / dir[2]

        # Intersección con paredes laterales (x=0, x=ancho_caja, y=0, y=ancho_caja)
        t_pared_x0 = (0.0 - pos[0]) / dir[0] if dir[0] != 0 else np.inf
        t_pared_x1 = (ancho_caja - pos[0]) / dir[0] if dir[0] != 0 else np.inf
        t_pared_y0 = (0.0 - pos[1]) / dir[1] if dir[1] != 0 else np.inf
        t_pared_y1 = (ancho_caja - pos[1]) / dir[1] if dir[1] != 0 else np.inf

        ts = []
        t_labels = []

        for t, label in [
            (t_celda, 'celda'),
            (t_backplate, 'backplate'),
            (t_emisor, 'emisor'),
            (t_suelo, 'suelo'),
            (t_pared_x0, 'pared_x0'),
            (t_pared_x1, 'pared_x1'),
            (t_pared_y0, 'pared_y0'),
            (t_pared_y1, 'pared_y1'),
        ]:
            if t > 1e-8:
                ts.append(t)
                t_labels.append(label)

        if not ts:
            break

        t_min = min(ts)
        label_min = t_labels[ts.index(t_min)]
        nueva_pos = [pos[i] + t_min * dir[i] for i in range(3)]
        x, y, z = nueva_pos

        # Paredes laterales: rebote especular
        if label_min in ['pared_x0', 'pared_x1', 'pared_y0', 'pared_y1']:
            impactos_pared.append((min(max(x, 0), ancho_caja), min(max(z, 0), altura_caja)))
            potencia_refl_pared = potencia * pared.reflectancia_subbg
            potencia_abs_pared = potencia * (1 - pared.reflectancia_subbg)
            pared.energia_reflejada += potencia_refl_pared
            pared.energia_absorbida += potencia_abs_pared
            potencia = potencia_refl_pared
            # Rebote especular: invierte la componente que ha chocado
            if label_min == 'pared_x0' or label_min == 'pared_x1':
                dir[0] *= -1
            else:
                dir[1] *= -1
            pos = nueva_pos
            continue

        # Planos horizontales:
        if label_min == 'celda':
            impactos_celda.append((x, z))
            lambda_g = 1.24e-6 #/ celda.Eg_eV if celda.Eg_eV else 0
            if lambda_g and lambda_m < lambda_g:
                potencia_abs = potencia * celda.absorptancia_util * celda.EQE
                potencia_refl = potencia * (1 - celda.absorptancia_util)
                celda.energia_absorbida += potencia_abs
                celda.energia_reflejada += potencia_refl
                potencia = potencia_refl
                dir = list(direccion_aleatoria_hemiesfera())
                dir[2] = abs(dir[2])
            else:
                potencia_refl = potencia * celda.reflectancia_subbg
                potencia_trans = potencia * (1 - celda.reflectancia_subbg)
                celda.energia_reflejada += potencia_refl
                celda.energia_transmitida += potencia_trans
                if potencia_trans > 1e-12:
                    potencia = potencia_trans
                    pos = nueva_pos
                    dir = [dir[0], dir[1], dir[2]]
                    continue
                potencia = potencia_refl
                dir = list(direccion_aleatoria_hemiesfera())
                dir[2] = abs(dir[2])

        elif label_min == 'backplate':
            impactos_backplate.append((x, z))
            potencia_refl_bp = potencia * backplate.reflectancia_subbg
            potencia_abs_bp = potencia * (1 - backplate.reflectancia_subbg)
            backplate.energia_reflejada += potencia_refl_bp
            backplate.energia_absorbida += potencia_abs_bp
            potencia = potencia_refl_bp
            dir = list(direccion_aleatoria_hemiesfera())
            dir[2] = abs(dir[2])

        elif label_min == 'emisor':
            # Solo cuenta como reciclaje si el impacto cae en la franja física del emisor
            if x0_emisor <= x <= x1_emisor:
                impactos_emisor.append((x, z))
                emisor.potencia_reciclada += potencia
            else:
                impactos_pared.append((x, z))
                pared.energia_absorbida += potencia
            break

        elif label_min == 'suelo':
            impactos_pared.append((x, z))
            potencia_refl_pared = potencia * pared.reflectancia_subbg
            potencia_abs_pared = potencia * (1 - pared.reflectancia_subbg)
            pared.energia_reflejada += potencia_refl_pared
            pared.energia_absorbida += potencia_abs_pared
            potencia = potencia_refl_pared
            dir = list(direccion_aleatoria_hemiesfera())
            dir[2] = abs(dir[2])

        else:
            break

        pos = nueva_pos
        if potencia < 1e-15:
            break

# ====== Resultados de energía (en W) ======
print("RESULTADOS TPV:\n")
print(f"Potencia generada por la celda TPV:      {celda.energia_absorbida:} W")
print(f"Radiación reflejada por celda TPV:      {celda.energia_reflejada:} W")
print(f"Radiación transmitida por celda TPV:    {celda.energia_transmitida:} W")
print(f"Radiación absorbida en backplate:       {backplate.energia_absorbida:} W")
print(f"Radiación reflejada por backplate:      {backplate.energia_reflejada:} W")
print(f"Radiación absorbida en paredes:         {pared.energia_absorbida:} W")
print(f"Radiación reflejada por paredes:        {pared.energia_reflejada:} W")
print(f"Radiación reciclada (devuelta al emisor): {emisor.potencia_reciclada:} W")
potencia_total_emitida = sum(r.potencia for r in rayos)
print(f"\nRadiación total emitida por el emisor: {potencia_total_emitida:} W")
absorcion_fraccion = celda.energia_absorbida / potencia_total_emitida if potencia_total_emitida > 0 else 0
print(f"Fracción absorbida por la celda TPV: {absorcion_fraccion:.2%}")
print(f"Fracción reciclada: {emisor.potencia_reciclada/potencia_total_emitida:.2%}")


# ====== Eficiencia TPV ======
if potencia_total_emitida - emisor.potencia_reciclada > 0:
    eficiencia_tpv = celda.energia_absorbida / (potencia_total_emitida - emisor.potencia_reciclada)
else:
    eficiencia_tpv = 0.0
print(f"\nEficiencia TPV del sistema: {eficiencia_tpv:.2%}")

# ====== Visualización mejorada ======
plt.figure(figsize=(8, 8))
# Caja: suelo, techo, paredes laterales
""" plt.plot([0, ancho_caja], [0, 0], color='black', linewidth=2)  # Suelo
plt.plot([0, ancho_caja], [altura_caja, altura_caja], color='grey', linewidth=2)  # Techo (resto)
plt.plot([0, 0], [0, altura_caja], color='black', linewidth=2)  # Pared izq
plt.plot([ancho_caja, ancho_caja], [0, altura_caja], color='black', linewidth=2)  # Pared dcha

# Emisor: solo la franja activa
plt.plot([x0_emisor, x1_emisor], [altura_caja, altura_caja], color='purple', linewidth=4, label="Emisor térmico")

# Celda TPV
plt.plot([0, ancho_caja], [pos_celda, pos_celda], color='blue', linestyle='--', linewidth=2, label="Celda TPV")
# Backplate
plt.plot([0, ancho_caja], [pos_backplate, pos_backplate], color='green', linestyle='-.', linewidth=2, label="Backplate")
 """
# Impactos como puntos bien visibles
if impactos_celda:
    x, z = zip(*impactos_celda)
    plt.scatter(x, z, s=tam_punto, color='blue', alpha=0.7, label="Impactos Celda TPV")
if impactos_backplate:
    x, z = zip(*impactos_backplate)
    plt.scatter(x, z, s=tam_punto, color='green', alpha=0.7, label="Impactos Backplate")
if impactos_pared:
    x, z = zip(*impactos_pared)
    plt.scatter(x, z, s=tam_punto, color='red', alpha=0.6, label="Impactos Pared")
if impactos_emisor:
    x, z = zip(*impactos_emisor)
    plt.scatter(x, z, s=tam_punto, color='purple', alpha=0.7, label="Impactos Emisor")

plt.title("Simulación TPV – Distribución de impactos de rayos")
plt.xlabel("Coordenada X (m)")
plt.ylabel("Altura Z (m)")
plt.xlim(-0.01, ancho_caja + 0.01)
plt.ylim(-0.01, altura_caja + 0.01)
plt.legend(markerscale=2, fontsize=9, loc='lower right')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
