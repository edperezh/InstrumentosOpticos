# Created on: 10/12/2024
# Author: edperezh@unal.edu.co

# Simulación y visualización el comportamiento del campo 
# difractado en planos específicos definidos por la expresión:

# z = N * L^2 / λ
 
# donde N puede tomar valores pares o impares. Este enfoque
# permite analizar las diferencias en el patrón de difracción
# dependiendo de la paridad de N.

# Planos simulados:

# - Caso N = 2 (par): En este plano, el patrón difractado es
# idéntico al patrón inicial de la transmittancia periódica,
# conservando la misma distribución de intensidad.

# - Caso N = 3 (impar): En este plano, el patrón presenta una
# inversión de fase, lo que se traduce en una alteración 
# evidente en la distribución del campo difractado comparado
# con el caso N = 2.

# Resultados obtenidos:

# - Visualización de la intensidad difractada: Se generaron
# gráficos que muestran claramente las distribuciones de
# intensidad para ambos casos (N par e impar).

# - Comparación de patrones: Los resultados permiten observar
# de manera precisa las diferencias clave entre los patrones 
# difractados para valores pares e impares de N, destacando 
# la inversión de fase en el caso impar.

import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
L = 1e-3             # Período espacial de la transmittancia periódica (metros)
m = 0.5              # Índice de modulación
grid_size = 1024     # Resolución de la cuadrícula
dx = 10e-6           # Tamaño del píxel (metros)

# Coordenadas del plano
x = np.linspace(-grid_size/2, grid_size/2, grid_size) * dx
y = np.linspace(-grid_size/2, grid_size/2, grid_size) * dx
X, Y = np.meshgrid(x, y)

# Transmittancia periódica
def periodic_transmittance(x, m, L):
    return 0.5 * (1 + m * np.cos(2 * np.pi * x / L))

# Generar transmittancia
transmittance = periodic_transmittance(X, m, L)

# Coordenadas de frecuencia
fx = np.fft.fftfreq(grid_size, dx)
fy = np.fft.fftfreq(grid_size, dx)
FX, FY = np.meshgrid(fx, fy)

# Función de transferencia de Fresnel
def fresnel_transfer_function(fx, fy, wavelength, z):
    return np.exp(-1j * np.pi * wavelength * z * (fx**2 + fy**2))

# Parámetros para z
z_period = L**2 / wavelength
N_values = [2, 3]  # N par e impar

fig, axes = plt.subplots(1, len(N_values), figsize=(15, 6))

for i, N in enumerate(N_values):
    z = N * z_period

    # Función de transferencia
    H = fresnel_transfer_function(FX, FY, wavelength, z)

    # Transformada de Fourier de la transmittancia
    field_spectrum = np.fft.fft2(transmittance)

    # Campo difractado
    output_field = np.fft.ifft2(field_spectrum * H)
    intensity = np.abs(output_field)**2

    # Visualización
    ax = axes[i]
    im = ax.imshow(intensity, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
    ax.set_title(f"Intensidad difractada (N={N}, z={z:.2e} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, label="Intensidad")

plt.tight_layout()
plt.show()
