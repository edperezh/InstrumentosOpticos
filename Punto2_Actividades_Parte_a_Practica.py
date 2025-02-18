"""
Este script en Python calcula el ángulo de interferencia (θ) en un montaje de holografía digital off-axis.
El código consta de:

1) Carga del holograma (en formato TIFF).
2) Cálculo de la Transformada de Fourier 2D y centrado (fftshift).
3) Búsqueda del lóbulo off-axis en el espectro (excluyendo el orden 0).
4) Conversión del desplazamiento en píxeles a frecuencia espacial (ciclos/m),
   usando la indexación discreta adecuada para evitar aliasing.
5) Cálculo de sin(θ) = λ·f y, por consiguiente, el ángulo θ en grados.
6) Visualización del holograma y del espectro, marcando el pico hallado.

Comentarios detallados se incluyen dentro del bloque de código.
"""

import numpy as np               # Manejo de arreglos y funciones matemáticas
import matplotlib.pyplot as plt  # Gráficas y visualización
from skimage import io           # Para cargar la imagen (holograma)
from scipy.ndimage import gaussian_filter  # Filtro Gaussiano opcional

# --------------------------------------------------------------------
# PARÁMETROS FÍSICOS
# --------------------------------------------------------------------
lambda_laser = 632.8e-9  # (m) Longitud de onda. Ej: Láser He-Ne ~ 632.8 nm
pixel_size   = 3.45e-6   # (m) Tamaño de píxel del sensor (p.ej. DMM 37UX250-ML)

# --------------------------------------------------------------------
# 1) CARGA DEL HOLOGRAMA
# --------------------------------------------------------------------
# Se lee la imagen 'Hologram.tiff' en escala de grises y se pasa a float
holograma = io.imread('Hologram.tiff', as_gray=True).astype(float)

# Se obtienen las dimensiones: Ny (filas), Nx (columnas)
Ny, Nx = holograma.shape
print(f"Holograma cargado con tamaño: {Ny} filas x {Nx} columnas")

# --------------------------------------------------------------------
# 2) FFT 2D + CENTRADO (fftshift)
# --------------------------------------------------------------------
# Se calcula la transformada de Fourier 2D para pasar del dominio espacial al de frecuencias
H = np.fft.fft2(holograma)

# fftshift centra el orden 0 de la FFT en el medio de la imagen
H_shifted = np.fft.fftshift(H)

# Se toma la magnitud (en log) para visualizar mejor
magnitude_spectrum = np.log1p(np.abs(H_shifted))

# Filtrado Gaussiano (opcional), para suavizar el espectro y facilitar la detección del pico
magnitude_spectrum_filt = gaussian_filter(magnitude_spectrum, sigma=2)

# --------------------------------------------------------------------
# 3) LOCALIZAR EL LÓBULO (PICO) OFF-AXIS
# --------------------------------------------------------------------
# Se define el centro de la imagen FFT (frecuencia 0) en pixeles
cy, cx = Ny // 2, Nx // 2

# Se crea una copia para enmascarar la región central (orden 0)
masked = magnitude_spectrum_filt.copy()

# Se anulan 20 píxeles alrededor del centro para no confundir con el pico DC
R_central = 20
masked[cy - R_central:cy + R_central, cx - R_central:cx + R_central] = 0.0

# Se busca el índice del valor máximo en 'masked', que debe ser el lóbulo off-axis
peak_y, peak_x = np.unravel_index(np.argmax(masked), masked.shape)

# Delta en píxeles respecto al centro
delta_x_pix = peak_x - cx
delta_y_pix = peak_y - cy

print(f"Pico en la FFT: (peak_x, peak_y)=({peak_x}, {peak_y})")
print(f"Desplazamiento X (píxeles): {delta_x_pix}")
print(f"Desplazamiento Y (píxeles): {delta_y_pix}")

# --------------------------------------------------------------------
# 4) FRECUENCIA ESPACIAL (CICLOS/M) con indexación DFT
# --------------------------------------------------------------------
# Cada paso en el eje de frecuencias en la DFT vale df = 1 / (N_x * pixel_size)
# Por lo tanto, la frecuencia f_x = delta_x_pix * df, y f_y = delta_y_pix * df
df = 1.0 / (Nx * pixel_size)   # (ciclos/m)

freq_x_m = delta_x_pix * df
freq_y_m = delta_y_pix * df

# Se obtiene la magnitud de esa frecuencia espacial para el caso de inclinación en x,y
freq_mod_m = np.sqrt(freq_x_m**2 + freq_y_m**2)

print(f"Frecuencia en X (ciclos/m): {freq_x_m:.3e}")
print(f"Frecuencia en Y (ciclos/m): {freq_y_m:.3e}")
print(f"Frecuencia total (ciclos/m): {freq_mod_m:.3e}")

# --------------------------------------------------------------------
# 5) ÁNGULO DE INTERFERENCIA
# --------------------------------------------------------------------
# De la teoría off-axis: sin(theta) = lambda_laser * freq_mod_m
sin_theta = lambda_laser * freq_mod_m

# Si sin_theta excede 1, hay aliasing o se eligió un pico equivocado
if abs(sin_theta) > 1:
    print(f"¡Advertencia!: sin(theta)={sin_theta:.2f} => >1 => aliasing/pico malo.")
    sin_theta = np.clip(sin_theta, -1, 1)

# Se calcula theta en radianes y luego se convierte a grados
theta_rad = np.arcsin(sin_theta)
theta_deg = np.degrees(theta_rad)

print(f"Ángulo de interferencia (grados) ≈ {theta_deg:.3f}°")

# --------------------------------------------------------------------
# 6) VISUALIZAR RESULTADOS
# --------------------------------------------------------------------
# Se muestra el holograma original en escala de grises
plt.figure(figsize=(5,4))
plt.title("Holograma original")
plt.imshow(holograma, cmap='gray')
plt.colorbar()
plt.show()

# Se muestra el espectro (log) con el pico marcado
plt.figure(figsize=(5,4))
plt.title("Espectro (log) con fftshift")
plt.imshow(magnitude_spectrum, cmap='jet')
plt.colorbar()

# Dibujamos un punto rojo en la posición del pico hallado
plt.scatter([peak_x], [peak_y], color='red', marker='x')
plt.show()
