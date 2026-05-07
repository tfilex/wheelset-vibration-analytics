import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import pywt

def main():
    # Путь к файлу с дефектом (предполагается запуск из корня проекта)
    file_path = 'data/raw/1_IR_007/105.mat'
    fs = 12000
    
    # Загрузка сигнала
    if not os.path.exists(file_path):
        print(f"Внимание: Файл {file_path} не найден!")
        print("Для демонстрации будет сгенерирован искусственный сигнал с импульсами.")
        t = np.linspace(0, 0.1, int(0.1 * fs), endpoint=False)
        signal_slice = 0.5 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.2, len(t))
        # Добавляем импульсы
        signal_slice[int(0.04 * fs):int(0.045 * fs)] += 2 * np.sin(2 * np.pi * 3000 * t[int(0.04 * fs):int(0.045 * fs)])
        signal_slice[int(0.08 * fs):int(0.085 * fs)] += 2 * np.sin(2 * np.pi * 3000 * t[int(0.08 * fs):int(0.085 * fs)])
    else:
        mat_data = scipy.io.loadmat(file_path)
        signal = None
        # Динамический поиск ключа, оканчивающегося на _DE_time
        for key in mat_data.keys():
            if not key.startswith('__') and key.endswith('_DE_time'):
                signal = mat_data[key].flatten()
                break
                
        if signal is None:
            print(f"Ошибка: Не найден ключ _DE_time в файле {file_path}")
            return
            
        # Короткий срез сигнала (первые 0.1 секунды)
        num_points = int(fs * 0.1)
        signal_slice = signal[:num_points]
        t = np.linspace(0, 0.1, num_points, endpoint=False)

    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    print("Начинаю генерацию графиков...")

    # ==========================================
    # 1. Сырой вибросигнал
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal_slice, color='darkblue', linewidth=1)
    plt.title('Сырой вибросигнал во временной области', fontsize=14)
    plt.xlabel('Время, с', fontsize=12)
    plt.ylabel('Амплитуда', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(t[0], t[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_raw_signal.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. STFT-спектрограмма
    # ==========================================
    f_stft, t_stft, Zxx = scipy.signal.stft(signal_slice, fs=fs, nperseg=256, noverlap=128)
    Zxx_mag = np.abs(Zxx)
    Zxx_db = 20 * np.log10(np.maximum(Zxx_mag, 1e-10))
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_stft, f_stft, Zxx_db, cmap='magma', shading='gouraud')
    plt.title('STFT-спектрограмма', fontsize=14)
    plt.xlabel('Время, с', fontsize=12)
    plt.ylabel('Частота, Гц', fontsize=12)
    plt.colorbar(label='Логарифм модуля, дБ')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_stft_spectrogram.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. CWT-скалограмма
    # ==========================================
    scales = np.arange(1, 129)
    wavelet = 'cmor1.5-1.0'
    coefficients, frequencies = pywt.cwt(signal_slice, scales, wavelet, sampling_period=1/fs)
    cwt_mag = np.abs(coefficients)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, frequencies, cwt_mag, cmap='jet', shading='auto')
    plt.title('CWT-скалограмма (Вейвлет Морле)', fontsize=14)
    plt.xlabel('Время, с', fontsize=12)
    plt.ylabel('Частота, Гц', fontsize=12)
    plt.colorbar(label='Модуль коэффициентов')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_cwt_scalogram.png'), dpi=300)
    plt.close()

    # ==========================================
    # 4. Спектр огибающей (Envelope Spectrum) - классика для диагностики
    # ==========================================
    # Преобразование Гильберта для нахождения огибающей
    analytic_signal = scipy.signal.hilbert(signal_slice)
    envelope = np.abs(analytic_signal)
    envelope = envelope - np.mean(envelope) # Убираем постоянную составляющую
    
    f_env, Pxx_env = scipy.signal.welch(envelope, fs=fs, nperseg=min(1024, len(envelope)))
    
    plt.figure(figsize=(10, 4))
    plt.plot(f_env, Pxx_env, color='purple', linewidth=1.5)
    plt.title('Спектр огибающей (Envelope Spectrum)', fontsize=14)
    plt.xlabel('Частота, Гц', fontsize=12)
    plt.ylabel('Мощность', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 2000) # Ограничим до 2 кГц, дефектные частоты обычно низкие
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_envelope_spectrum.png'), dpi=300)
    plt.close()

    # ==========================================
    # 5. Спектральная плотность мощности (Метод Уэлча)
    # ==========================================
    f_psd, Pxx_den = scipy.signal.welch(signal_slice, fs=fs, nperseg=min(1024, len(signal_slice)))
    
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_psd, Pxx_den, color='forestgreen', linewidth=1.5)
    plt.title('Спектральная плотность мощности (PSD) Уэлча', fontsize=14)
    plt.xlabel('Частота, Гц', fontsize=12)
    plt.ylabel('Спектральная плотность [V²/Гц]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, fs/2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_psd_welch.png'), dpi=300)
    plt.close()

    print("Готово! Все графики сохранены в reports/figures/")

if __name__ == '__main__':
    main()
