# Created on: 10/12/2024
# Author: edperezh@unal.edu.co

""" El objetivo es determinar la distancia a la que se encontraba el objeto
del sensor durante el registro """

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.optimize import minimize

# Parámetros iniciales
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
N = 2048             # Tamaño de la cuadrícula (pixeles)
dx = 3.45e-6         # Tamaño de píxel en metros
dz_initial = 0.1     # Distancia inicial aproximada en metros

# Coordenadas del plano
x = np.linspace(-N/2, N/2 - 1, N) * dx
y = np.linspace(-N/2, N/2 - 1, N) * dx
X, Y = np.meshgrid(x, y)

# Cargar datos reales de intensidad registrada
# Aquí se simula con datos aleatorios; reemplaza esto con la carga de un archivo real
intensity = np.random.rand(N, N)  # Esto debería reemplazarse con datos reales

# Método del espectro angular para invertir la propagación
def angular_spectrum_method(intensity, wavelength, dx, dz):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)

    # Espectro angular
    H = np.exp(1j * dz * np.sqrt(k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2).clip(min=0))

    # Transformada de Fourier
    spectrum = fft2(intensity)

    # Aplicar la inversión del espectro angular
    spectrum_inverted = spectrum * np.conj(H)

    # Transformada inversa para regresar al plano inicial
    field_reconstructed = ifft2(spectrum_inverted)

    return field_reconstructed

# Función de pérdida para optimización de dz
def loss_function(dz, intensity, wavelength, dx):
    reconstructed_field = angular_spectrum_method(intensity, wavelength, dx, dz)
    reconstructed_intensity = np.abs(reconstructed_field)**2
    # Supongamos que queremos maximizar la similitud con algún patrón esperado (simulado aquí)
    reference_pattern = np.ones_like(intensity)  # Patrón de referencia
    return np.sum((reconstructed_intensity - reference_pattern)**2)

# Optimización de dz para encontrar la mejor distancia
result = minimize(loss_function, dz_initial, args=(intensity, wavelength, dx), method='Nelder-Mead')
best_dz = result.x[0]
print(f"Mejor distancia z encontrada: {best_dz:.6f} m")

# Calcular el campo reconstruido con el mejor dz
reconstructed_field = angular_spectrum_method(intensity, wavelength, dx, best_dz)
reconstructed_intensity = np.abs(reconstructed_field)**2

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Intensidad registrada
axes[0].imshow(intensity, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
axes[0].set_title("Intensidad registrada (sensor)")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

# Intensidad reconstruida
axes[1].imshow(reconstructed_intensity, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1].set_title(f"Intensidad reconstruida (z={best_dz:.6f} m)")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")

# Diferencias
difference = np.abs(reconstructed_intensity - intensity)
axes[2].imshow(difference, cmap='hot', extent=[x.min(), x.max(), y.min(), y.max()])
axes[2].set_title("Diferencia entre intensidades")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("y (m)")

plt.tight_layout()
plt.show()
