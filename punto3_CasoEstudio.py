# Created on: 10/12/2024
# Author: edperezh@unal.edu.co

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# Parámetros iniciales
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
distance = 0.1       # Distancia de propagación (metros)
grid_size = 1024     # Resolución de la cuadrícula
aperture_radius = 0.001  # Radio de la abertura circular (metros)
L = 0.01             # Tamaño físico del plano (metros)
dx = L / grid_size   # Tamaño del píxel en el plano
m = 0.5              # Modulación de la transmittancia
period = 2.5e-3      # Período de la transmittancia periódica

# Coordenadas del plano
x = np.linspace(-L/2, L/2, grid_size)
y = np.linspace(-L/2, L/2, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Transmittancia periódica
def periodic_transmittance(x, m, period):
    return 0.5 * (1 + m * np.cos((2 * np.pi / period) * x))

# Generar la transmittancia periódica
t_periodic = periodic_transmittance(X, m, period)

# Función de la Transformada de Fresnel con DFT
def fresnel_dft(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Cálculo del campo difractado
start = time.time()
fresnel_output_periodic = fresnel_dft(t_periodic, wavelength, distance, dx)
time_fresnel_periodic = time.time() - start

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Transmittancia periódica
ax_periodic = axes[0, 0]
ax_periodic.set_title("Transmittancia periódica")
im_periodic = ax_periodic.imshow(t_periodic, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_periodic, ax=ax_periodic)

# Campo difractado - Magnitud
ax_magnitude = axes[0, 1]
ax_magnitude.set_title(f"Campo difractado (Magnitud) - Tiempo: {time_fresnel_periodic:.4f} s")
im_magnitude = ax_magnitude.imshow(np.abs(fresnel_output_periodic), cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_magnitude, ax=ax_magnitude)

# Campo difractado - Real
ax_real = axes[1, 0]
ax_real.set_title("Campo difractado (Parte Real)")
im_real = ax_real.imshow(np.real(fresnel_output_periodic), cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_real, ax=ax_real)

# Campo difractado - Imaginario
ax_imag = axes[1, 1]
ax_imag.set_title("Campo difractado (Parte Imaginaria)")
im_imag = ax_imag.imshow(np.imag(fresnel_output_periodic), cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_imag, ax=ax_imag)

plt.tight_layout()
plt.show()
