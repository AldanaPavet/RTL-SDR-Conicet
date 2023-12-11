import numpy as np
import rtlsdr
import matplotlib.pyplot as plt

# -------------------------- RTL-SDR Configuration -------------------------------------------

sdr = rtlsdr.RtlSdr()                       # Inicializar el dispositivo RTL-SDR

sample_rate = 2.4e6                         # Tasa de muestreo en muestras por segundo (3MHz)
                                            # The maximum sample rate is 3.2 MS/s (mega samples per second). 
                                            # However, the RTL-SDR is unstable at this rate and may drop samples.
                                            # The maximum sample rate that does not drop samples is 2.56 MS/s.

# Configurar parámetros
# Rafael Micro R820T/2/R860 	24 – 1766 MHz (Can be improved to ~13 - 1864 MHz with experimental drivers)

sdr.sample_rate = sample_rate               # Tasa de muestreo en muestras por segundo
sdr.center_freq = 99900000                  # Frecuencia central en Hz (99.9 MHz)
sdr.gain = 42                               # Ganancia del receptor

output_file = "fm999.bin"                   # Nombre del archivo de salida con las muestras capturadas

block_size = 1024 * 1024                    # Tamaño de cada bloque de muestras
samples_buffer = []                         # Buffer para almacenar las muestras

# ---------------------------------------------------------------------------------------------------------

# ------------------------ Captura y procesamiento de muestras en bloques ---------------------------------

num_blocks = 1                                              # Capturar 1 bloques de muestras

for _ in range(num_blocks):
    
    samples = sdr.read_samples(block_size)                  # Capturar bloque de muestras
    np.save(output_file, samples, allow_pickle=False)       # Guardar las muestras en el archivo binario
    samples_buffer.extend(samples)                          # Agregar las muestras al buffer

    # Mantener el buffer dentro del tamaño deseado
    if len(samples_buffer) > block_size * num_blocks:
        samples_buffer = samples_buffer[-block_size * num_blocks:]


sdr.close()                                     # Cerrar el dispositivo RTL-SDR

signal_energy = np.sum(np.abs(samples_buffer)**2) / len(samples_buffer)    # Calcular la energía de la señal

threshold = 1e-8                                # Umbral de energía de la señal

information_present = signal_energy > threshold # Verificar si hay información en las muestras

if information_present:
    print("Se detectó información en las muestras.")
else:
    print("No se detectó información en las muestras.")

# -------------------------------------- Ploteos ---------------------------------------------------------

# Muestras capturadas
plt.plot(samples_buffer)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Muestras capturadas (ABS)')
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
