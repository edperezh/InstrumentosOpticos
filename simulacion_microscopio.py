import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.ndimage import gaussian_filter
from skimage import io

# -------------------------
# Parámetros del sistema óptico
# -------------------------

lambda_nm = 533  # Longitud de onda en nm
NA = 0.25  # Apertura numérica del objetivo
M = 10  # Aumento del objetivo
f_tube = 200  # Longitud focal del lente de tubo en mm
pixel_size = 2.74  # Tamaño de píxel del sensor en µm

# Convertimos unidades a micrómetros (µm)
lambda_um = lambda_nm / 1000  # nm a µm
f_tube_um = f_tube * 1000  # mm a µm
pixel_size_um = pixel_size  # µm

# Cálculo del límite de resolución según el criterio de Abbe
resolution_abbe = lambda_um / (2 * NA)

# -------------------------
# Carga de la imagen de prueba
# -------------------------

image_path = "Star_2048.tif" 
image = io.imread(image_path, as_gray=True)  # Cargar imagen en escala de grises
image = image / np.max(image)  # Normalizar la imagen para que sus valores estén entre 0 y 1

# -------------------------
# Reducción del tamaño de la imagen para optimizar memoria
# -------------------------

resize_factor = 0.1  # Reducir al 10% del tamaño original
new_size = (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA).astype(np.float32)

# -------------------------
# Simulación de la respuesta del sistema óptico
# -------------------------

def optical_transfer_function(image, resolution):
    """
    Simula el efecto del sistema óptico aplicando una función de transferencia óptica (OTF) basada en el criterio de Abbe.
    """
    H, W = image.shape  # Obtiene dimensiones de la imagen
    fx = np.fft.fftfreq(W, d=pixel_size_um)  # Calcula frecuencias espaciales en X
    fy = np.fft.fftfreq(H, d=pixel_size_um)  # Calcula frecuencias espaciales en Y
    FX, FY = np.meshgrid(fx, fy)  # Crea malla de coordenadas en el espacio de Fourier
    
    # Cálculo del filtro en Fourier basado en la función de transferencia óptica (OTF)
    radius = np.sqrt(FX**2 + FY**2)  # Radio en el dominio de Fourier
    cutoff = 1 / resolution  # Frecuencia de corte basada en la resolución de Abbe
    OTF = np.exp(- (radius / cutoff)**2)  # Aproximación gaussiana de la OTF
    
    # Transformada de Fourier de la imagen y aplicación del filtro
    image_ft = fft2(image)
    image_ft_filtered = image_ft * fftshift(OTF)
    
    # Transformada inversa para recuperar la imagen filtrada
    image_filtered = np.abs(ifft2(image_ft_filtered))
    return image_filtered

# Aplicar la función de transferencia óptica
image_simulated = optical_transfer_function(image_resized, resolution_abbe)

# -------------------------
# Guardado de resultados
# -------------------------

plt.imsave("imagen_original_reducida.png", image_resized, cmap='gray')
plt.imsave("imagen_simulada_microscopio.png", image_simulated, cmap='gray')

# -------------------------
# Visualización de los resultados
# -------------------------

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image_resized, cmap='gray')
ax[0].set_title("Imagen Original Reducida")
ax[0].axis("off")

ax[1].imshow(image_simulated, cmap='gray')
ax[1].set_title("Imagen con Simulación de Microscopio")
ax[1].axis("off")

plt.show()
