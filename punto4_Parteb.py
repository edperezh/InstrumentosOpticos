# Created on: 10/12/2024
# Author: edperezh@unal.edu.co

# Comparación entre la Reconstrucción desde Intensidad y desde Campo Complejo
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# Parámetros iniciales
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
N = 2048             # Tamaño de la cuadrícula (pixeles)
dx = 3.45e-6         # Tamaño de píxel en metros
dz = 0.1             # Distancia del objeto al sensor (m)

# Coordenadas del plano
x = np.linspace(-N/2, N/2 - 1, N) * dx
y = np.linspace(-N/2, N/2 - 1, N) * dx
X, Y = np.meshgrid(x, y)

# Simulación de datos
# Parte real e imaginaria del campo complejo
real_component = np.random.rand(N, N)  # Simulación de la parte real
imag_component = np.random.rand(N, N)  # Simulación de la parte imaginaria
field_complex = real_component + 1j * imag_component  # Campo complejo inicial
intensity_measured = np.abs(field_complex)**2  # Intensidad medida en el sensor

# Función del espectro angular
# Propaga el campo hacia adelante o hacia atrás dependiendo de dz (positivo o negativo)
def angular_spectrum_method(field, wavelength, dx, dz):
    k = 2 * np.pi / wavelength  # Número de onda
    fx = np.fft.fftfreq(N, dx)  # Frecuencias espaciales en x
    fy = np.fft.fftfreq(N, dx)  # Frecuencias espaciales en y
    FX, FY = np.meshgrid(fx, fy)

    # Construcción de la función de transferencia H(fx, fy)
    H = np.exp(1j * dz * np.sqrt(k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2).clip(min=0))

    # Transformada de Fourier del campo
    spectrum = fft2(field)

    # Aplicar H para propagar el campo
    propagated_spectrum = spectrum * H

    # Transformada inversa para regresar al dominio espacial
    propagated_field = ifft2(propagated_spectrum)

    return propagated_field

# Simulación 1: Reconstrucción desde intensidad medida
field_from_intensity = np.sqrt(intensity_measured)  # Suponer fase uniforme
reconstructed_field_from_intensity = angular_spectrum_method(field_from_intensity, wavelength, dx, -dz)
reconstructed_intensity_from_intensity = np.abs(reconstructed_field_from_intensity)**2

# Simulación 2: Reconstrucción desde campo complejo
reconstructed_field_from_complex = angular_spectrum_method(field_complex, wavelength, dx, -dz)
reconstructed_intensity_from_complex = np.abs(reconstructed_field_from_complex)**2

# Visualización
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Intensidad medida
axes[0, 0].imshow(intensity_measured, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
axes[0, 0].set_title("Intensidad medida (sensor)")
axes[0, 0].set_xlabel("x (m)")
axes[0, 0].set_ylabel("y (m)")

# Reconstrucción desde intensidad medida
axes[0, 1].imshow(reconstructed_intensity_from_intensity, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
axes[0, 1].set_title("Reconstrucción desde intensidad")
axes[0, 1].set_xlabel("x (m)")
axes[0, 1].set_ylabel("y (m)")

# Diferencias desde intensidad
difference_intensity = np.abs(reconstructed_intensity_from_intensity - intensity_measured)
axes[0, 2].imshow(difference_intensity, cmap='hot', extent=[x.min(), x.max(), y.min(), y.max()])
axes[0, 2].set_title("Diferencias (intensidad)")
axes[0, 2].set_xlabel("x (m)")
axes[0, 2].set_ylabel("y (m)")

# Campo complejo inicial
axes[1, 0].imshow(np.abs(field_complex)**2, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1, 0].set_title("Campo complejo inicial")
axes[1, 0].set_xlabel("x (m)")
axes[1, 0].set_ylabel("y (m)")

# Reconstrucción desde campo complejo
axes[1, 1].imshow(reconstructed_intensity_from_complex, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1, 1].set_title("Reconstrucción desde campo complejo")
axes[1, 1].set_xlabel("x (m)")
axes[1, 1].set_ylabel("y (m)")

# Diferencias desde campo complejo
difference_complex = np.abs(reconstructed_intensity_from_complex - np.abs(field_complex)**2)
axes[1, 2].imshow(difference_complex, cmap='hot', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1, 2].set_title("Diferencias (campo complejo)")
axes[1, 2].set_xlabel("x (m)")
axes[1, 2].set_ylabel("y (m)")

plt.tight_layout()
plt.show()

# Descripción de las diferencias
print("Diferencias observadas:")
print("1. La reconstrucción desde intensidad muestra pérdidas debido a la ausencia de información de fase.")
print("2. La reconstrucción desde el campo complejo es precisa, ya que conserva amplitud y fase.")
