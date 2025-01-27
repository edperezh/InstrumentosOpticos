import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# Cargar datos de la muestra biológica
muestra_path = "C:\\Eder Perez\\Unal\\13\\IO\\MuestraBio_E06.csv"
muestra_biologica = np.genfromtxt(muestra_path, delimiter=',')

# Definición de parámetros físicos
lamda = 600E-9  # Longitud de onda en metros
f1 = 10E-3  # Distancia focal de la lente L1 en metros
D_L1 = 7E-3  # Diámetro de la lente L1 en metros

sensor_width = muestra_biologica.shape[1]  # Número de columnas de la muestra
sensor_height = muestra_biologica.shape[0]  # Número de filas de la muestra
pixel_size = 3.45e-6  # Tamaño de píxel del sensor en metros

# Dimensiones físicas del plano de entrada
ancho_muestra = sensor_width * pixel_size
alto_muestra = sensor_height * pixel_size
x = np.linspace(-ancho_muestra / 2, ancho_muestra / 2, sensor_width)
y = np.linspace(-alto_muestra / 2, alto_muestra / 2, sensor_height)
X, Y = np.meshgrid(x, y)

# Crear una transmitancia compleja basada en la muestra
muestra_transmitancia = muestra_biologica * np.exp(1j * np.pi * (X**2 + Y**2) / lamda)

# Transformada de Fourier de la muestra
muestra_fft = fftshift(fft2(ifftshift(muestra_transmitancia)))

# Máscara de la función pupila
def funcion_pupila(shape, D_L1):
    rows, cols = shape
    x = np.linspace(-D_L1 / 2, D_L1 / 2, cols)
    y = np.linspace(-D_L1 / 2, D_L1 / 2, rows)
    X, Y = np.meshgrid(x, y)
    pupila = np.sqrt(X**2 + Y**2) <= (D_L1 / 2)
    return pupila.astype(np.float32)

pupila = funcion_pupila(muestra_biologica.shape, D_L1)

# Aplicar la pupila en el dominio de Fourier
muestra_filtrada_fft = muestra_fft * pupila

# Regresar al dominio espacial
campo_observado = ifft2(ifftshift(muestra_filtrada_fft))
intensidad_observada = np.abs(campo_observado)**2

# Visualización de los resultados
plt.figure(figsize=(10, 8))
plt.title("Mapa de Intensidad")
plt.imshow(intensidad_observada, extent=[-ancho_muestra/2, ancho_muestra/2, -alto_muestra/2, alto_muestra/2], cmap='gray')
plt.colorbar(label="Intensidad")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()
