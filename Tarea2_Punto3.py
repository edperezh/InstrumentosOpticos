import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifftshift, fftshift

lamda= 600E-9
f1=500E-3
f2=500E-3
d=500E-3

L0=2*f1
L02=f2+d
lens1_diameter= 100E-3

sensor_width = 2448  # resolución del sensor en píxeles (ancho)
sensor_height = 2048  # resolución del sensor en píxeles (alto)
pixel_size = 3.45e-6  # tamaño de píxel del sensor en metros (3.45 micrómetros)

pixel_size_pupila = 3.45e-6 * 2 #Para facilidades de computo y uso de la RAM
tamañopupila_x=2* (sensor_width*pixel_size) #hacemos la pupila dos veces el tamaño fisico del sensor, para evitar problemas de difracción y exceso de información
tamañopupila_y=2* (sensor_height*pixel_size)

image_path = "C:\\Eder Perez\\Unal\\13\\IO\\Ruido_E06.png"
U0_raw= cv2.imread (image_path,cv2.IMREAD_GRAYSCALE)
height, width = U0_raw.shape
print(f"Dimensiones de la imagen de entrada: {width}x{height} píxeles")

tamaño_fisico_entrada = (width * pixel_size,  # Ancho físico
                         height * pixel_size)  # Alto físico
print(f"Dimensiones fisicas de la imagen de entrada: {tamaño_fisico_entrada[0]}x{tamaño_fisico_entrada[1]} m")

U0 = np.pad(U0_raw, ((0, (2048 - 768)), (0, 2448 - 768)), mode='constant', constant_values=0)
plt.imshow(U0, cmap='gray')


# Intervalo de muestreo en el plano de entrada
di = tamaño_fisico_entrada[0] / width  # Paso en i
dj = tamaño_fisico_entrada[1] / height  # Paso en j

# Coordenadas físicas del plano de entrada
i = np.linspace(-tamaño_fisico_entrada[0] / 2, tamaño_fisico_entrada[0] / 2, width)
h = np.linspace(-tamaño_fisico_entrada[1] / 2, tamaño_fisico_entrada[1] / 2, height)
I, H = np.meshgrid(i, h)





#Defino la malla del campo que llega a la pupila, deberá tener al menos un tamaño de 100 mm para garantizar que toda la luz que pase por la lente L1, llegue a la Pupila.
num_pixels_x = int(tamañopupila_x / pixel_size_pupila)
num_pixels_y = int(tamañopupila_y / pixel_size_pupila)

if num_pixels_y % 2 != 0:
    num_pixels_y += 1  # Asegurar número par para simetría
    
if num_pixels_x % 2 != 0:
    num_pixels_x += 1  # Asegurar número par para simetría

# Crear el plano de la pupila
x = np.linspace(-tamañopupila_x / 2, tamañopupila_x / 2, num_pixels_x)  # Coordenadas x
y = np.linspace(-tamañopupila_y / 2, tamañopupila_y / 2, num_pixels_y)  # Coordenadas y
X, Y = np.meshgrid(x, y)

#Defino la malla del campo de observacion

# Calcular el tamaño físico del plano de observación (en metros)
plane_width = sensor_width * pixel_size  # en metros
plane_height = sensor_height * pixel_size  # en metros

# Crear la malla de muestreo para el plano de observación
u = np.linspace(-plane_width / 2, plane_width / 2, sensor_width)  # intervalo en x
v = np.linspace(-plane_height / 2, plane_height / 2, sensor_height)  # intervalo en y

# Crear malla 2D de coordenadas
U, V = np.meshgrid(u, v)

U_pupila= fftshift(fft2(ifftshift(U0)))

magnitude = np.abs(U_pupila)

# Visualizar el espectro
plt.imshow(np.log(magnitude + 1), cmap='gray')  # Usamos log para mejorar la visualización
plt.title('Espectro de Fourier campo de entrada')
plt.colorbar()
plt.show()

k = 2 * np.pi / lamda
terminos_fase = -1 * np.exp(1j * (k / (2 * f2**2)) * (f2-d) * (U**2 + V**2)) * np.exp(1j * (k * (L0-L02))) * (lamda**2) * f1 * f2


# Filtro para pasa altas
cutoff_radius = 50  # Radio de corte para el filtro pasa altas
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

P2 = high_pass_filter(U_pupila.shape, cutoff_radius)

U_filtrado2 = U_pupila * P2
#Para facilidad de computo se tomará d=f2 para hacer 1 el termino parabolico
#Transfor_U_filtrado2= fft2(ifftshift(U_filtrado2))
U_filtrado_final2= U_filtrado2*terminos_fase

U_observacion2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_filtrado_final2)))
plt.imshow(np.abs(U_observacion2),cmap='gray')