import numpy as np
import rtlsdr
import matplotlib.pyplot as plt
from scipy.signal import welch, windows, freqz


# -------------------------- RTL-SDR Configuration -------------------------------------------

sdr = rtlsdr.RtlSdr()                       # Inicializar el dispositivo RTL-SDR

sample_rate = 2.4e6                         # Tasa de muestreo en muestras por segundo (3MHz)
                                            # The maximum sample rate is 3.2 MS/s (mega samples per second). 
                                            # However, the RTL-SDR is unstable at this rate and may drop samples.
                                            # The maximum sample rate that does not drop samples is 2.56 MS/s.

# Configurar parámetros
# Rafael Micro R820T/2/R860 	24 – 1766 MHz (Can be improved to ~13 - 1864 MHz with experimental drivers)
# Experimentalmente capta desde 12MHz 

sdr.sample_rate = sample_rate               # Tasa de muestreo en muestras por segundo
sdr.center_freq = 99.9e6                    # Frecuencia central en Hz (99.9 MHz)
sdr.gain = 42                               # Ganancia del receptor

output_file = "fm999.bin"                   # Nombre del archivo de salida con las muestras capturadas

block_size = 1024 * 1024                    # Tamaño de cada bloque de muestras (tasa de muestreo * duracion de la muestra)
samples_buffer = []                         # Buffer para almacenar las muestras


# ------------------------ Captura y procesamiento de muestras en bloques ---------------------------------

num_blocks = 1                                              # Capturar 1 bloques de muestras

for _ in range(num_blocks):
    
    samples = sdr.read_samples(block_size)                  # Capturar bloque de muestras
    np.save(output_file, samples, allow_pickle=False)       # Guardar las muestras en el archivo binario
    samples_buffer.extend(samples)                          # Agregar las muestras al buffer

    # Mantener el buffer dentro del tamaño deseado
    if len(samples_buffer) > block_size * num_blocks:
        samples_buffer = samples_buffer[-block_size * num_blocks:]


sdr.close()                                                 # Cerrar el dispositivo RTL-SDR

signal_energy = np.sum(np.abs(samples_buffer)**2) / len(samples_buffer)    # Calcular la energía de la señal

threshold = 1e-8                                # Umbral de energía de la señal

information_present = signal_energy > threshold # Verificar si hay información en las muestras

if information_present:
    print("Se detectó información en las muestras.")
else:
    print("No se detectó información en las muestras.")

# -------------------------------------- Procesamiento para generar el espectro de potencia con Welch ---------------------------------------------------------

# Parámetros de Welch
m = 256                                                     # Tamaño de cada segmento de datos
k = 128                                                     # Superposición entre segmentos
ventana = np.bartlett(m)                                    # Ventana de Bartlett

# Cálculo del espectro de potencia mediante Welch
N = len(samples_buffer)
S = int(np.floor((N-m) / (m-k))) + 1                        # Número total de segmentos

Pxx_segments = np.zeros((S, m))                             # Inicializar arreglo para almacenar los espectros de potencia de cada segmento

# Calcular el espectro de potencia para cada segmento
for i in range(S-1):
    start = i* (m-k) +1
    end = start + m
    segment = samples_buffer[start:end] * ventana

    Pxx_segments[i, :] = np.abs(np.fft.fft(segment))**2 / (np.sum(ventana**2) * sample_rate)

PSD = np.mean(Pxx_segments, axis=0)                         # Promediar los espectros de potencia de todos los segmentos
frecuencias_welch = np.fft.fftfreq(m, d=1/sample_rate)      # Frecuencias correspondientes a los segmentos de datos

# -------------------------------------- Ploteos ---------------------------------------------------------

# Muestras capturadas
plt.plot(samples_buffer)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Muestras capturadas')
plt.show()

# Muestras capturadas
plt.plot(np.abs(samples_buffer))
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Muestras capturadas absolutas')
plt.show()

# Transformada de Fourier y frecuencias correspondiente de las muestras en el dominio de la frecuencia
fft_result = np.fft.fft(samples_buffer)
magnitud_fft = np.abs(fft_result)
frecuencias = np.fft.fftfreq(len(samples_buffer), d=1/sample_rate)

# Grafica de la transformada de Fourier
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, magnitud_fft)
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.show()

# Espectro de frecuencia
frequencies, spectrum = plt.psd(samples_buffer, NFFT=1024, Fs=sample_rate, scale_by_freq=True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad espectral de potencia')
plt.title('Espectro de frecuencia')
plt.show()

# Calcular la respuesta en frecuencia
w, h = freqz(np.sqrt(PSD), worN=8000)       # Utilizar la raíz cuadrada para obtener la magnitud
w_normalized = w / np.pi                    # Convertir la frecuencia a radianes por segundo normalizada por pi
magnitude_db = 20 * np.log10(np.abs(h))     # Calcular la magnitud en decibelios

# Graficar la respuesta en magnitud 
plt.figure(figsize=(10, 6))
plt.plot(w_normalized, magnitude_db)
plt.title('Estimación de la Respuesta en Magnitud')
plt.xlabel('Frecuencia Normalizada (xπ rad/muestras)')
plt.ylabel('Magnitud (dB)')
plt.grid(True)
plt.show()

# Calcular las frecuencias normalizadas en unidades de omega/pi
frecuencias_normalizadas = frecuencias_welch / np.pi

# Graficar el espectro de potencia promedio en función de omega/pi
plt.figure(figsize=(10, 6))
plt.semilogy(frecuencias_normalizadas, PSD)
plt.text(0.5, 0.9, "Ventaneo con Bartlett", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.title('Espectro de Potencia usando Welch')
plt.xlabel('Frecuencia Normalizada (ω/π)')
plt.ylabel('Densidad espectral de potencia')
plt.grid(True)
plt.show()
