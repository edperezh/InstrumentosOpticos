import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import pandas as pd

# Paso 1: Cargar la imagen de transmittancia con ruido
image_path = r"C:\\Eder Perez\\Unal\\13\\Instrumentos ópticos\\Ruido_E06.png"

# image_path = "Ruido_E06.png"
image = Image.open(image_path).convert("L")  # Convertir a escala de grises
image_array = np.array(image)

# Paso 2: Definir dimensiones físicas
D_L1 = 100e-3  # Diámetro de la lente en metros
f_1 = 500e-3   # Distancia focal de la lente en metros

# Asociar dimensiones físicas a la imagen
pixel_size = D_L1 / image_array.shape[0]  # Tamaño del píxel en metros
x = np.linspace(-D_L1 / 2, D_L1 / 2, image_array.shape[1])
y = np.linspace(-D_L1 / 2, D_L1 / 2, image_array.shape[0])
X, Y = np.meshgrid(x, y)

# Paso 3: Transformada de Fourier de la imagen
image_fft = fftshift(fft2(image_array))
spectrum = np.log(1 + np.abs(image_fft))

# Crear un filtro pasa bajas (gaussiano)
def gaussian_filter(shape, cutoff):
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    filter_ = np.exp(-(distance**2) / (2 * (cutoff**2)))
    return filter_

cutoff = 0.05  # Frecuencia de corte
filter_ = gaussian_filter(image_array.shape, cutoff)

# Aplicar el filtro en el dominio de la frecuencia
filtered_fft = image_fft * filter_
filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))

# Paso 4: Visualización de la primera parte
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Imagen original")
plt.imshow(image_array, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Espectro de Fourier")
plt.imshow(spectrum, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Filtro Gaussiano")
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

# Paso 5: Cargar datos de la muestra biológica
sample_path = "C:\\Eder Perez\\Unal\\13\\Instrumentos ópticos\\MuestraBio_E06.csv"
# sample_data = pd.read_csv(sample_path, header=None).values  
# sample_data = pd.read_csv(sample_path, header=None, dtype=complex).values # Leer como matriz de valores
# Leer el archivo CSV como texto
sample_data_raw = pd.read_csv(sample_path, header=None, dtype=str)

# Convertir cada valor a número complejo
sample_data = sample_data_raw.applymap(lambda x: complex(x)).values



# Dimensiones físicas de la muestra
sample_size = 125e-6  # 125 micrones
sample_pixel_size = sample_size / sample_data.shape[0]  # Tamaño del píxel

# Coordenadas para la muestra
x_sample = np.linspace(-sample_size / 2, sample_size / 2, sample_data.shape[1])
y_sample = np.linspace(-sample_size / 2, sample_size / 2, sample_data.shape[0])
X_sample, Y_sample = np.meshgrid(x_sample, y_sample)

# Paso 6: Transformada de Fourier de la muestra
sample_fft = fftshift(fft2(sample_data))
sample_spectrum = np.log(1 + np.abs(sample_fft))

# Aplicar un filtro gaussiano similar al anterior
sample_filtered_fft = sample_fft * filter_
sample_filtered_image = np.abs(ifft2(ifftshift(sample_filtered_fft)))

# Paso 7: Visualización de la segunda parte
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Muestra original")
plt.imshow(sample_data, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Espectro de Fourier de la muestra")
plt.imshow(sample_spectrum, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Filtro Gaussiano")
plt.imshow(filter_, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("FFT filtrada de la muestra")
plt.imshow(np.log(1 + np.abs(sample_filtered_fft)), cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Muestra filtrada")
plt.imshow(sample_filtered_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
