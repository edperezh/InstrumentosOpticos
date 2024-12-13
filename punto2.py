# Created on: 09/12/2024
# Author: edperezh@unal.edu.co

"""Implementación en Python de los métodos de
    difracción escalar de Transformada de Fresnel y Espectro Angular
    en sus formas de sumatorias discretas (DFT), y en la forma basada
    en transformadas rápidas de Fourier (FFT). Compare el desempeño
    de ambos métodos en términos del tiempo necesario para realizar
    el cómputo de un mismo experimento difractivo de su elección.
"""

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

# Coordenadas del plano
x = np.linspace(-L/2, L/2, grid_size)
y = np.linspace(-L/2, L/2, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Abertura circular
aperture = np.where(R <= aperture_radius, 1, 0)

# Función de la Transformada de Fresnel con DFT
def fresnel_dft(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Función de la Transformada de Fresnel con FFT
def fresnel_fft(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Función del Espectro Angular
def angular_spectrum(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(1j * distance * np.sqrt((k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2).clip(min=0)))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Cálculo con los tres métodos
start = time.time()
fresnel_output_dft = fresnel_dft(aperture, wavelength, distance, dx)
time_fresnel_dft = time.time() - start

start = time.time()
fresnel_output_fft = fresnel_fft(aperture, wavelength, distance, dx)
time_fresnel_fft = time.time() - start

start = time.time()
angular_output = angular_spectrum(aperture, wavelength, distance, dx)
time_angular = time.time() - start

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Abertura circular
ax_aperture = axes[0, 0]
ax_aperture.set_title("Abertura circular")
im_aperture = ax_aperture.imshow(aperture, cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_aperture, ax=ax_aperture)

# Fresnel DFT
ax_dft = axes[0, 1]
ax_dft.set_title(f"Fresnel (DFT) - Tiempo: {time_fresnel_dft:.4f} s")
im_dft = ax_dft.imshow(np.abs(fresnel_output_dft)**2, cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_dft, ax=ax_dft)

# Fresnel FFT
ax_fft = axes[1, 0]
ax_fft.set_title(f"Fresnel (FFT) - Tiempo: {time_fresnel_fft:.4f} s")
im_fft = ax_fft.imshow(np.abs(fresnel_output_fft)**2, cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_fft, ax=ax_fft)

# Espectro Angular
ax_angular = axes[1, 1]
ax_angular.set_title(f"Espectro Angular - Tiempo: {time_angular:.4f} s")
im_angular = ax_angular.imshow(np.abs(angular_output)**2, cmap='Purples', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im_angular, ax=ax_angular)

plt.tight_layout()

# Añadir deslizadores para ajustar la intensidad
slider_height = 0.15
ax_slider_dft = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider_dft = Slider(ax_slider_dft, 'Intensidad', 0.1, 1.0, valinit=1.0)

# Etiquetas numéricas para los métodos
def update(val):
    factor = slider_dft.val
    im_dft.set_data(np.abs(fresnel_output_dft)**2 * factor)
    im_fft.set_data(np.abs(fresnel_output_fft)**2 * factor)
    im_angular.set_data(np.abs(angular_output)**2 * factor)
    im_aperture.set_data(aperture * factor)
    fig.canvas.draw_idle()

slider_dft.on_changed(update)

# Indicadores numéricos
for ax, label in zip([ax_aperture, ax_dft, ax_fft, ax_angular], ["Abertura", "DFT", "FFT", "Espectro"]):
    ax.text(0.5, -0.15, label, transform=ax.transAxes, ha='center', fontsize=10, color='black')

plt.show()
