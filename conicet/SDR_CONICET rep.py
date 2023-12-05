import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

def capture_fm_signal(frequency, sample_rate, gain, duration, output_file):
    sdr = RtlSdr()

    # Configurar el dispositivo RTL-SDR
    sdr.sample_rate = sample_rate
    sdr.center_freq = frequency
    sdr.gain = gain

    # Capturar la señal
    samples = sdr.read_samples(duration * sample_rate)
    samples = np.complex64(samples)

    # Guardar la señal en un archivo binario
    with open(output_file, 'wb') as file:
        samples.tofile(file)

    # Cerrar el dispositivo RTL-SDR
    sdr.close()

    return samples

def plot_fm_signal(samples, sample_rate):
    # Visualizar la señal modulada en frecuencia recibida
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(samples))
    plt.title('Señal Modulada en Frecuencia Recibida')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.show()

def plot_spectrum(samples, sample_rate):
    # Calcular el espectro de la señal
    spectrum = np.fft.fftshift(np.fft.fft(samples))
    freq = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))

    # Visualizar el espectro de la señal
    plt.figure(figsize=(10, 4))
    plt.plot(freq, 20 * np.log10(np.abs(spectrum)))
    plt.title('Espectro de la Señal Modulada en Frecuencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.show()

if __name__ == "__main__":
    # Configuración de la captura
    frequency = 99.9e6  # Frecuencia de la emisora 99.9 FM en Hz
    sample_rate = 2.048e6  # Frecuencia de muestreo en Hz
    gain = 42  # Ganancia RF en dB
    duration = 0.05  # Duración de la captura en segundos
    output_file = "captured_signal.bin"

    # Capturar la señal
    samples = capture_fm_signal(frequency, sample_rate, gain, duration, output_file)

    # Visualizar la señal modulada en frecuencia recibida
    plot_fm_signal(samples, sample_rate)

    # Visualizar el espectro de la señal modulada en frecuencia recibida
    plot_spectrum(samples, sample_rate)

