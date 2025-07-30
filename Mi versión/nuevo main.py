import numpy as np
from material_tpv import MaterialTPV
from ray_tpv import RayTPV, generar_rayos_planck

# ====== Parámetros ======
T_emisor = 2000
A_emisor = 0.01
ancho_caja = 1.0
altura_caja = 1.0
pos_emisor = altura_caja
pos_celda = 0.5
pos_backplate = 0.45
num_rayos = 100   # Pocos rayos para test mínimo
max_rebotes = 10  # Suficiente, solo habrá uno

lambda_min = 0.5e-6
lambda_max = 5.0e-6

# ====== Materiales absorbentes ======
celda = MaterialTPV(
    Eg_eV=0,
    absorptancia_util=1.0,        # 100% absorbente
    reflectancia_subbg=0.0,
    EQE=0,
    nombre="Celda TPV"
)
backplate = MaterialTPV(
    Eg_eV=0,
    absorptancia_util=1.0,        # 100% absorbente
    reflectancia_subbg=0.0,
    EQE=0,
    nombre="Backplate reflector"
)
pared = MaterialTPV(
    Eg_eV=0,
    absorptancia_util=1,        # 100% absorbente
    reflectancia_subbg=0,
    EQE=0,
    nombre="Pared absorbente"
)
emisor = MaterialTPV(
    Eg_eV=0,
    absorptancia_util=0,
    reflectancia_subbg=0,
    EQE=0,
    nombre="Emisor térmico"
)
emisor.potencia_reciclada = 0.0

potencia_no_acumulada = 0.0

L_emisor = np.sqrt(A_emisor)
x0_emisor = (ancho_caja - L_emisor) / 2
x1_emisor = (ancho_caja + L_emisor) / 2

rayos = generar_rayos_planck(
    T_emisor, A_emisor, lambda_min, lambda_max, num_rayos
)
potencia_total_emitida = sum(r.potencia for r in rayos)

def direccion_aleatoria_hemiesfera():
    theta = np.arccos(np.random.uniform(0, 1))
    phi = np.random.uniform(0, 2*np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = -np.abs(np.cos(theta))
    return (x, y, z)

for rayo in rayos:
    x0 = np.random.uniform(x0_emisor, x1_emisor)
    y0 = np.random.uniform(0, ancho_caja)
    pos = [x0, y0, pos_emisor]
    dir = list(direccion_aleatoria_hemiesfera())
    potencia = rayo.potencia
    lambda_m = rayo.lambda_m

    for rebote in range(max_rebotes):
        if dir[2] == 0:
            potencia_no_acumulada += potencia
            break

        t_celda = (pos_celda - pos[2]) / dir[2]
        t_backplate = (pos_backplate - pos[2]) / dir[2]
        t_emisor = (altura_caja - pos[2]) / dir[2]
        t_suelo = (0.0 - pos[2]) / dir[2]
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
            potencia_no_acumulada += potencia
            break

        t_min = min(ts)
        label_min = t_labels[ts.index(t_min)]

        # --- Rebote en superficies perfectamente absorbentes ---
        if label_min == 'celda':
            celda.energia_absorbida += potencia
            break
        elif label_min == 'backplate':
            backplate.energia_absorbida += potencia
            break
        elif label_min in ['pared_x0', 'pared_x1', 'pared_y0', 'pared_y1', 'suelo']:
            pared.energia_absorbida += potencia
            break
        elif label_min == 'emisor':
            emisor.potencia_reciclada += potencia
            break
        else:
            potencia_no_acumulada += potencia
            break

# ====== Balance ======
print("======== BALANCE TEST MINIMO ========")
print(f"Absorbido por celda:      {celda.energia_absorbida:.6f} W")
print(f"Absorbido por backplate:  {backplate.energia_absorbida:.6f} W")
print(f"Absorbido en paredes:     {pared.energia_absorbida:.6f} W")
print(f"Reciclado (emisor):       {emisor.potencia_reciclada:.6f} W")
print(f"No absorbido (numérico):  {potencia_no_acumulada:.6f} W")
total = celda.energia_absorbida + backplate.energia_absorbida + pared.energia_absorbida + emisor.potencia_reciclada + potencia_no_acumulada
print(f"\nPotencia total emitida:   {potencia_total_emitida:.6f} W")
print(f"Suma de acumuladores:     {total:.6f} W")
print(f"Diferencia:               {(potencia_total_emitida - total):.6f} W")
