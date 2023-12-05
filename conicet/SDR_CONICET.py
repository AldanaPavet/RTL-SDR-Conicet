import numpy as np
import rtlsdr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detectar_modulacion_fm(muestras, tasa_muestreo):
    # Calcular la magnitud de la transformada de Fourier de las muestras
    fft_result = np.fft.fft(muestras)
    magnitud_fft = np.abs(fft_result)

    # Calcular las frecuencias correspondientes a las muestras en el dominio de la frecuencia
    frecuencias = np.fft.fftfreq(len(muestras), d=1/tasa_muestreo)

    # Encontrar picos en el espectro de frecuencia
    picos, _ = find_peaks(magnitud_fft, height=0.1)

    # Definir un umbral para determinar si la señal está modulada en frecuencia
    umbral = 0.01

    # Analizar los picos y determinar si la señal está modulada en frecuencia
    for pico_idx in picos:
        frecuencia_pico = frecuencias[pico_idx]
        amplitud_pico = magnitud_fft[pico_idx]

        if amplitud_pico > umbral:
            # La señal está modulada en frecuencia si hay un pico significativo en el espectro
            return True, frecuencia_pico

    # Si no se encontraron picos significativos, la señal no está modulada en frecuencia
    return False, None

#--------------------------- RTL-SDR Configuration -------------------------------------------

sdr = rtlsdr.RtlSdr() # Inicializar el dispositivo RTL-SDR

sample_rate = 1e6

# Configurar parámetros
sdr.sample_rate = sample_rate               # Tasa de muestreo en muestras por segundo
sdr.center_freq = 98300000                  # Frecuencia central en Hz (99.9 MHz)
sdr.gain = 42                               # Ganancia del receptor

output_file = "fm999.bin"                   # Nombre del archivo de salida con las muestras capturadas

samples = sdr.read_samples(1024 * 1024)     # Cantidad de muestras a capturar
np.save(output_file, samples)               # Guardar las muestras en un archivo binario

sdr.close()                                 # Cerrar el dispositivo RTL-SDR

# Muestras capturadas
plt.plot(np.abs(samples))
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Muestras capturadas')

# Detectar FM
modulada, frecuencia_pico = detectar_modulacion_fm(samples, sample_rate)

# Magnitud de la transformada de Fourier y las frecuencias
fft_result = np.fft.fft(samples)
magnitud_fft = np.abs(fft_result)
frecuencias = np.fft.fftfreq(len(samples), d=1/sample_rate)

# Grafica de la transformada de Fourier
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, magnitud_fft)
plt.title('Magnitud de la Transformada de Fourier')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.show()

# Resultado de la detección de FM
if modulada:
    print(f"La señal está modulada en frecuencia. Frecuencia pico: {frecuencia_pico} Hz")
else:
    print("La señal no está modulada en frecuencia.")

plt.show()

# Espectro de frecuencia
frequencies, spectrum = plt.psd(samples, NFFT=1024, Fs=sample_rate, scale_by_freq=True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad espectral de potencia')
plt.title('Espectro de frecuencia')
plt.show()

