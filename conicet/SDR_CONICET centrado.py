import numpy as np
import rtlsdr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#------------------- espectro de potencia de la señal FM recibida -----------------------------------
#                      centrarlo alrededor de la frecuencia cero 

def detectar_modulacion_fm(muestras, tasa_muestreo):
    # Calcular la magnitud de la transformada de Fourier de las muestras
    fft_result = np.fft.fft(muestras)
    magnitud_fft = np.abs(fft_result)

    # Aplicar fftshift para centrar el espectro de frecuencia
    magnitud_fft_shifted = np.fft.fftshift(magnitud_fft)

    # Calcular las frecuencias correspondientes a las muestras en el dominio de la frecuencia
    frecuencias = np.fft.fftshift(np.fft.fftfreq(len(muestras), d=1/tasa_muestreo))

    # Encontrar picos en el espectro de frecuencia
    picos, _ = find_peaks(magnitud_fft_shifted, height=0.1)

    # Definir un umbral para determinar si la señal está modulada en frecuencia
    umbral = 0.01

    # Analizar los picos y determinar si la señal está modulada en frecuencia
    for pico_idx in picos:
        frecuencia_pico = frecuencias[pico_idx]
        amplitud_pico = magnitud_fft_shifted[pico_idx]

        if amplitud_pico > umbral:
            # La señal está modulada en frecuencia si hay un pico significativo en el espectro
            return True, frecuencia_pico

    # Si no se encontraron picos significativos, la señal no está modulada en frecuencia
    return False, None

# Configuración de parámetros
sample_rate = 2000000  # Tasa de muestreo en muestras por segundo
center_frequency = 99900000  # Frecuencia central en Hz (99.9 MHz)
gain = 42  # Ganancia del receptor

# Inicializar el dispositivo RTL-SDR
sdr = rtlsdr.RtlSdr()

# Configurar parámetros
sdr.sample_rate = sample_rate
sdr.center_freq = center_frequency
sdr.gain = gain

# Capturar muestras
samples = sdr.read_samples(1024 * 1024)  # Puedes ajustar la cantidad de muestras según tus necesidades

# Visualizar la magnitud de la transformada de Fourier y las frecuencias
fft_result = np.fft.fft(samples)
magnitud_fft = np.abs(fft_result)

# Aplicar fftshift para centrar el espectro de frecuencia
magnitud_fft_shifted = np.fft.fftshift(magnitud_fft)
frecuencias = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1/sample_rate))

# Graficar la magnitud de la transformada de Fourier centrada
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, magnitud_fft_shifted)
plt.title('Espectro de Potencia Centrado')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.show()

# Detectar modulación en frecuencia
modulada, frecuencia_pico = detectar_modulacion_fm(samples, sample_rate)

# Demodulación en frecuencia y extracción de la señal de información
if modulada:
    # Obtener la fase de las muestras
    fase_muestras = np.angle(samples)

    # Diferenciar la fase para obtener las variaciones de fase
    variaciones_fase = np.diff(fase_muestras)

    # Obtener la señal de información demodulada
    signal_demodulada = np.unwrap(variaciones_fase)

    # Graficar las variaciones de fase
    plt.figure(figsize=(10, 6))
    plt.plot(variaciones_fase)
    plt.title('Variaciones de Fase (Demodulación en Frecuencia)')
    plt.xlabel('Muestras')
    plt.ylabel('Fase')
    plt.show()

    # Graficar la señal de información transmitida y la señal demodulada
    tiempo = np.arange(len(signal_demodulada)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo, np.sin(2 * np.pi * 1000 * tiempo), label='Señal de Información Transmitida')
    plt.plot(tiempo, signal_demodulada, label='Señal Demodulada')
    plt.title('Señal de Información Transmitida y Señal Demodulada')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.show()

    # Puedes continuar con el análisis de la señal demodulada según sea necesario
else:
    print("La señal no está modulada en frecuencia.")

# Cerrar el dispositivo RTL-SDR
sdr.close()



