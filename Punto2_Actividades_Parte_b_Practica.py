import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------
# PARÁMETROS FÍSICOS
# --------------------------------------------------------
lambda_laser = 632.8e-9
pixel_size   = 3.45e-6

# --------------------------------------------------------
# 1) CARGAR HOLOGRAMA
# --------------------------------------------------------
holograma = io.imread('Hologram.tiff', as_gray=True).astype(float)
Ny, Nx = holograma.shape
print(f"Holograma shape: {Ny} x {Nx}")

# --------------------------------------------------------
# 2) FFT 2D + CENTRADO
# --------------------------------------------------------
H = np.fft.fft2(holograma)
H_shifted = np.fft.fftshift(H)

magnitude_spectrum = np.log1p(np.abs(H_shifted))
magnitude_spectrum_filt = gaussian_filter(magnitude_spectrum, sigma=2)

# --------------------------------------------------------
# 3) LOCALIZAR LÓBULO OFF-AXIS
# --------------------------------------------------------
cy, cx = Ny // 2, Nx // 2
masked = magnitude_spectrum_filt.copy()
R_central = 20
masked[cy-R_central:cy+R_central, cx-R_central:cx+R_central] = 0.0

peak_y, peak_x = np.unravel_index(np.argmax(masked), masked.shape)
delta_x_pix = peak_x - cx
delta_y_pix = peak_y - cy
print(f"Pico FFT en x={peak_x}, y={peak_y}")
print(f"Desplaz. X={delta_x_pix} pix, Y={delta_y_pix} pix")

# --------------------------------------------------------
# 4) FRECUENCIA ESPACIAL (opcional, ya lo tenías)
# --------------------------------------------------------
df = 1.0/(Nx*pixel_size)
fx = delta_x_pix*df
fy = delta_y_pix*df
fmod = np.sqrt(fx**2 + fy**2)
print(f"f_mod={fmod:.3e} [1/m]")
# ...

# --------------------------------------------------------
# 5) FILTRAR LÓBULO
# --------------------------------------------------------
mask_lobulo = np.zeros_like(H_shifted, dtype=bool)
R_lob = 50  # radio en pix
yy, xx = np.indices(H_shifted.shape)
rr = np.sqrt((xx-peak_x)**2 + (yy-peak_y)**2)
mask_lobulo[rr<R_lob] = True

H_lob = H_shifted*mask_lobulo  # se deja solo la región del lóbulo

# (OPCIONAL) Visualizar la máscara
# plt.figure(); plt.imshow(mask_lobulo, cmap='gray'); plt.show()

# --------------------------------------------------------
# B) RECENTRAR EL LÓBULO AL CENTRO
# --------------------------------------------------------
# Queremos que el pico pase de (peak_x, peak_y) a (cx, cy):
shift_x = cx - peak_x
shift_y = cy - peak_y

# np.roll permite desplazar (fila, col) => (shift_y, shift_x)
H_lob_centered = np.roll(H_lob, shift=(shift_y, shift_x), axis=(0,1))



# --------------------------------------------------------
# 6) IFFT -> CAMPO COMPLEJO
# --------------------------------------------------------
# Ojo con ifftshift: si ya está en "versión shift", lo usual es ifftshift()
# y luego ifft2.  
H_lob_centered_ifft = np.fft.ifftshift(H_lob_centered)
campo_complex = np.fft.ifft2(H_lob_centered_ifft)

amplitud = np.abs(campo_complex)
fase     = np.angle(campo_complex)

plt.figure(figsize=(5,4))
plt.title("H_lob_centered (magnitud en log)")
plt.imshow( np.log1p(np.abs(H_lob_centered)), cmap='gray')
plt.colorbar()
plt.show()


# --------------------------------------------------------
# GRAFICAR
# --------------------------------------------------------
plt.figure()
plt.title("Holograma original")
plt.imshow(holograma, cmap='gray')
plt.colorbar()

plt.figure()
plt.title("Espectro (log) con fftshift")
plt.imshow(magnitude_spectrum, cmap='jet')
plt.colorbar()
plt.scatter([peak_x],[peak_y], color='red', marker='x')

plt.figure()
plt.title("Amplitud (reconstruida) ya con el pico centrado")
plt.imshow(amplitud, cmap='gray')
plt.colorbar()
plt.show()



