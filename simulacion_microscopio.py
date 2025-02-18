import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifftshift, fftshift

# Definición de parámetros
lamda = 533E-7  # Longitud de onda en metros
f1 = 0.02  # Focal del primer lente
f2 = 0.2  # Focal del segundo lente
d = 0.2  # Distancia
M = 10  # Magnificación del objetivo

L0 = 2 * f1  # Camino óptico 1
L1 = f2 + d  # Camino óptico 2
diam_pupila = 0.0105  # Diámetro de la pupila en metros
radio_pupila = diam_pupila / 2

sensor_pixels = 2848  # Pixeles del sensor (sensor cuadrado)
pixel_size = 2.74E-6  # Tamaño de píxel en metros

# Cálculo del tamaño físico del sensor
sensor_size = sensor_pixels * pixel_size
print(f"Dimensiones físicas del sensor: {sensor_size} m x {sensor_size} m")

# Tamaño de pixel en la muestra
dx_muestra = pixel_size / M
# Tamaño del campo de visión en la muestra
Lx = sensor_pixels * dx_muestra
print(f"Dimensiones físicas de la muestra: {Lx} m x {Lx} m")

# Coordenadas espaciales en el plano de entrada (muestra)
px = np.linspace(-radio_pupila, radio_pupila, sensor_pixels)
py = np.linspace(-radio_pupila, radio_pupila, sensor_pixels)
Px, Py = np.meshgrid(px, py)

# Crear la pupila circular basada en la apertura numérica NA = 0.25
distance = np.sqrt(Px**2 + Py**2)
pupila = np.zeros((sensor_pixels, sensor_pixels))
pupila[distance <= radio_pupila] = 1  # Se usa 1 en vez de 255 para normalización

plt.figure(figsize=(6, 6))
plt.imshow(pupila, cmap='gray', vmin=0, vmax=1)
plt.title("Pupila Circular")

# Cargar la imagen de entrada y verificar que se carga correctamente
image_path = "USAF_3000px_cl.jpg"
U0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if U0 is None:
    print(f"Error: No se pudo cargar la imagen '{image_path}'. Verifica la ruta.")
    exit()

# Redimensionar la imagen al tamaño del sensor si es necesario
U0 = cv2.resize(U0, (sensor_pixels, sensor_pixels), interpolation=cv2.INTER_AREA)
plt.figure(figsize=(6, 6))
plt.imshow(U0, cmap='gray')
plt.title("Imagen de Entrada")

# Transformada de Fourier de la imagen de entrada
U_pupila = fftshift(fft2(ifftshift(U0)))
U_pupila_observado = np.abs(U_pupila) ** 2

plt.figure(figsize=(6, 6))
plt.imshow(np.log(U_pupila_observado + 1), cmap='gray')
plt.title("Transformada de Fourier de la Imagen")

# Multiplicación en frecuencia con la pupila
U = U_pupila * pupila
U_observado = np.abs(U) ** 2

plt.figure(figsize=(6, 6))
plt.imshow(np.log(U_observado + 1), cmap='gray')
plt.title("Imagen Filtrada por la Pupila")

# Propagación óptica de la imagen
k = 2 * np.pi / lamda  # Número de onda
terminos_faseprop = (-1 * np.exp(1j * (k * (L0 + L1))) / ((lamda**2) * f1 * f2)) \
    * np.exp((1j * k * (f2 - d) * (Px**2 + Py**2)) / (2 * f2**2))

U_sensor = fftshift(fft2(ifftshift(U)))
U_final_propagado = terminos_faseprop * U_sensor
U_observado2 = np.abs(U_final_propagado) ** 2

plt.figure(figsize=(6, 6))
plt.imshow(np.log(U_observado2 + 1), cmap='gray')
plt.title("Imagen Final en el Sensor")

plt.show()
