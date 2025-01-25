import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image

# Paso 1: Cargar la imagen de transmittancia con ruido
image_path = "C:\\Eder Perez\\Unal\\13\\Instrumentos ópticos\\Ruido_E06.png"

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
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt((X - X[crow, ccol])**2 + (Y - Y[crow, ccol])**2)
    filter_ = np.exp(-(distance**2) / (2 * (cutoff**2)))
    return filter_

cutoff = 0.05  # Frecuencia de corte
filter_ = gaussian_filter(image_array.shape, cutoff)

# Aplicar el filtro en el dominio de la frecuencia
filtered_fft = image_fft * filter_
filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))

# Paso 4: Visualización
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
