import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros iniciales 
wavelength = 633e-9  # Longitud de onda en metros (luz roja: 633 nm)
distance = 0.1       # Distancia de propagación (metros)
grid_size = 1024     # Resolución de la cuadrícula
aperture_radius = 0.001  # Radio de la abertura circular (metros)
L = 0.01             # Tamaño físico del plano (metros)
dx = L / grid_size   # Tamaño del píxel en el plano

# Coordenadas del plano
x = np.linspace(-L/2, L/2, grid_size)
y = np.linspace(-L/2, L/2, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Abertura circular 
aperture = np.where(R <= aperture_radius, 1, 0)

# Función de la Transformada de Fresnel con DFT
def fresnel_dft(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Función de la Transformada de Fresnel con FFT
def fresnel_fft(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Función del Espectro Angular
def angular_spectrum(input_field, wavelength, distance, dx):
    k = 2 * np.pi / wavelength
    N = input_field.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(1j * distance * np.sqrt((k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2).clip(min=0)))
    field_out = np.fft.ifft2(np.fft.fft2(input_field) * H)
    return field_out

# Cálculo con los tipos métodos
start = time.time()
fresnel_output_dft = fresnel_dft(aperture, wavelength, distance, dx)
time_fresnel_dft = time.time() - start

start = time.time()
fresnel_output_fft = fresnel_fft(aperture, wavelength, distance, dx)
time_fresnel_fft = time.time() - start

start = time.time()
angular_output = angular_spectrum(aperture, wavelength, distance, dx)
time_angular = time.time() - start

# Visualización
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.title("Abertura circular")
plt.imshow(aperture, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar()

plt.subplot(232)
plt.title(f"Fresnel (DFT) - Tiempo: {time_fresnel_dft:.4f} s")
plt.imshow(np.abs(fresnel_output_dft)**2, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar()

plt.subplot(233)
plt.title(f"Fresnel (FFT) - Tiempo: {time_fresnel_fft:.4f} s")
plt.imshow(np.abs(fresnel_output_fft)**2, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar()

plt.subplot(234)
plt.title(f"Espectro Angular - Tiempo: {time_angular:.4f} s")
plt.imshow(np.abs(angular_output)**2, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar()

plt.tight_layout()
plt.show()
