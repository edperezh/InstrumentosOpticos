# Created on: 10/12/2024
# Author: edperezh@unal.edu.co

# Esta es una simulación numérica cuyos detalles se describen a continuación estructurados por pasos y elementos clave del código:
#
# Parámetros iniciales:
# - Longitud de onda (λ): 633 nm.
# - Distancia de propagación (z): 0.1 m.
# - Resolución de la cuadrícula: 1024×1024 puntos.
# - Período de la transmittancia (L): 1 mm.
# - Índice de modulación (m): 0.5.
#
# Transmitancia periódica:
# - Se genera una función periódica de transmitancia t(x, y), utilizando los valores de L y m definidos.
#
# Cálculo del campo difractado:
# - Transformada de Fourier 2D: Se aplica a la transmitancia periódica inicial.
# - Multiplicación con la función de transferencia de Fresnel: Esto modela la propagación del campo en el espacio libre bajo la aproximación de Fresnel.
# - Transformada Inversa de Fourier: Permite recuperar la distribución del campo en el plano z.
#
# Visualización:
# - Representación de la transmittancia inicial.
# - Visualización de la intensidad del campo difractado en el plano z.
# - Gráfica del perfil de intensidad a lo largo del eje x en el punto y=0.

import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros iniciales
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
distance = 0.1       # Distancia de propagación (metros)
grid_size = 1024     # Resolución de la cuadrícula
L = 1e-3             # Período espacial de la transmitancia periódica (metros)
m = 0.5              # Índice de modulación
dx = 10e-6           # Tamaño del píxel (metros)

# Coordenadas del plano
x = np.linspace(-grid_size/2, grid_size/2, grid_size) * dx
y = np.linspace(-grid_size/2, grid_size/2, grid_size) * dx
X, Y = np.meshgrid(x, y)

# Transmitancia periódica
def periodic_transmittance(x, m, L):
    return 0.5 * (1 + m * np.cos(2 * np.pi * x / L))

# Generar transmitancia
transmittance = periodic_transmittance(X, m, L)

# Coordenadas de frecuencia
fx = np.fft.fftfreq(grid_size, dx)
fy = np.fft.fftfreq(grid_size, dx)
FX, FY = np.meshgrid(fx, fy)

# Función de transferencia de Fresnel
def fresnel_transfer_function(fx, fy, wavelength, distance):
    return np.exp(-1j * np.pi * wavelength * distance * (fx**2 + fy**2))

# Calcular el campo difractado
start = time.time()
H = fresnel_transfer_function(FX, FY, wavelength, distance)
field_spectrum = np.fft.fft2(transmittance)
output_field = np.fft.ifft2(field_spectrum * H)
intensity = np.abs(output_field)**2
calc_time = time.time() - start

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Transmitancia inicial
axes[0].imshow(transmittance, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
axes[0].set_title("Transmitancia inicial")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

# Intensidad del campo difractado
im = axes[1].imshow(intensity, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1].set_title(f"Intensidad difractada (z={distance} m)")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
plt.colorbar(im, ax=axes[1], label="Intensidad")

# Perfil de intensidad
profile = intensity[int(grid_size/2), :]
axes[2].plot(x, profile)
axes[2].set_title("Perfil de intensidad (y=0)")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("Intensidad")

plt.tight_layout()
plt.show()

print(f"Tiempo de cálculo: {calc_time:.4f} segundos")
