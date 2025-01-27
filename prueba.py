import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Definición de parámetros
lamda = 600E-9
f1 = 500E-3
f2 = 500E-3
d = 500E-3

L0 = 2 * f1
L02 = f2 + d
lens1_diameter = 100E-3

sensor_width = 2448  # Resolución del sensor en píxeles (ancho)
sensor_height = 2048  # Resolución del sensor en píxeles (alto)
pixel_size = 3.45e-6  # Tamaño de píxel del sensor en metros (3.45 micrómetros)

pixel_size_pupila = 3.45e-6 * 2  # Para facilidades de cómputo y uso de la RAM
tamañopupila_x = 2 * (sensor_width * pixel_size)
tamañopupila_y = 2 * (sensor_height * pixel_size)

# Leer la imagen con ruido
U0_raw = cv2.imread("Ruido_E06.png", cv2.IMREAD_GRAYSCALE)
height, width = U0_raw.shape

# Crear el plano de la pupila
num_pixels_x = int(tamañopupila_x / pixel_size_pupila)
num_pixels_y = int(tamañopupila_y / pixel_size_pupila)

if num_pixels_y % 2 != 0:
    num_pixels_y += 1  # Asegurar número par para simetría

if num_pixels_x % 2 != 0:
    num_pixels_x += 1  # Asegurar número par para simetría

x = np.linspace(-tamañopupila_x / 2, tamañopupila_x / 2, num_pixels_x)
y = np.linspace(-tamañopupila_y / 2, tamañopupila_y / 2, num_pixels_y)
X, Y = np.meshgrid(x, y)

# Transformada de Fourier del campo de entrada
U_pupila = fftshift(fft2(ifftshift(U0_raw)))

# Implementar filtro pasa altas
def high_pass_filter(shape, cutoff_radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter_ = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance < cutoff_radius:
                filter_[i, j] = 0  # Bloquear frecuencias bajas
    return filter_

cutoff_radius = 50  # Ajustar el radio de corte según la necesidad
high_pass = high_pass_filter(U_pupila.shape, cutoff_radius)

# Aplicar el filtro pasa altas
U_filtrado = U_pupila * high_pass

# Transformada inversa para obtener el campo de observación
U_observacion = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_filtrado)))

# Visualización
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title("Espectro original")
plt.imshow(np.log(np.abs(U_pupila) + 1), cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Filtro pasa altas")
plt.imshow(high_pass, cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Campo filtrado (magnitud)")
plt.imshow(np.abs(U_observacion), cmap='gray')
plt.colorbar()

plt.tight_layout()
plt.show()
