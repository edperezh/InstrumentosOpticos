import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image

# Definición de parámetros físicos y escalas
lamda = 600E-9  # Longitud de onda en metros
f1 = 500E-3  # Distancia focal de la primera lente en metros
f2 = 500E-3  # Distancia focal de la segunda lente en metros
d = 500E-3  # Distancia entre lentes en metros

L0 = 2 * f1  # Distancia entre entrada y lente L1
L02 = f2 + d  # Distancia entre lente L2 y plano de observación
lens1_diameter = 100E-3  # Diámetro de la lente en metros

sensor_width = 2448  # Resolución del sensor en píxeles (ancho)
sensor_height = 2048  # Resolución del sensor en píxeles (alto)
pixel_size = 3.45e-6  # Tamaño de píxel del sensor en metros

# Dimensiones físicas de la pupila
tamañopupila_x = 2 * (sensor_width * pixel_size)
tamañopupila_y = 2 * (sensor_height * pixel_size)

# Cargar la imagen de entrada
image_path = "C:\\Eder Perez\\Unal\\13\\IO\\Ruido_E06.png"
image = Image.open(image_path).convert("L")  # Convertir a escala de grises
U0_raw = np.array(image)

# Ajustar U0 al tamaño del sistema
U0 = np.pad(U0_raw, ((0, sensor_height - U0_raw.shape[0]), 
                     (0, sensor_width - U0_raw.shape[1])), 
            mode='constant', constant_values=0)

# Asociar dimensiones físicas a la imagen
pixel_size_pupila = tamañopupila_x / U0.shape[1]
x = np.linspace(-tamañopupila_x / 2, tamañopupila_x / 2, U0.shape[1])
y = np.linspace(-tamañopupila_y / 2, tamañopupila_y / 2, U0.shape[0])
X, Y = np.meshgrid(x, y)

# Transformada de Fourier de U0
U0_fft = fftshift(fft2(U0))
spectrum = np.log(1 + np.abs(U0_fft))

# Crear un filtro pasa altas
cutoff_radius = 150  # Radio de corte para el filtro pasa altas

def high_pass_filter(shape, cutoff_radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    filter_ = np.zeros(shape)
    filter_[distance > cutoff_radius] = 1
    return filter_

filter_ = high_pass_filter(U0.shape, cutoff_radius)

# Aplicar el filtro en el dominio de la frecuencia
filtered_fft = U0_fft * filter_
filtered_image_padded = np.abs(ifft2(ifftshift(filtered_fft)))

# Remover el padding de la imagen filtrada
filtered_image = filtered_image_padded[:U0_raw.shape[0], :U0_raw.shape[1]]

# Visualización
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Imagen original (U0)")
plt.imshow(U0, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Espectro de Fourier")
plt.imshow(spectrum, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Filtro Pasa Altas")
plt.imshow(filter_, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("FFT filtrada")
plt.imshow(np.log(1 + np.abs(filtered_fft)), cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Imagen filtrada")
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
