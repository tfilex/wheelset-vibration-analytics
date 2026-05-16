import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import pywt
import glob

# Настройки
FS = 12000
DURATION = 0.1  # Длина среза в секундах


def load_signal(file_path):
    """Загрузка 1D сигнала из .mat файла."""
    mat_data = scipy.io.loadmat(file_path)
    for key in mat_data.keys():
        if not key.startswith('__') and key.endswith('_DE_time'):
            signal = mat_data[key].flatten()
            num_points = int(FS * DURATION)
            t = np.linspace(0, DURATION, num_points, endpoint=False)
            return signal[:num_points], t
    raise ValueError(f"Ключ _DE_time не найден в {file_path}")


def generate_individual_plots(signal_slice, t, output_dir, defect_name):
    """Генерация 5 графиков для конкретного сигнала."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Сырой сигнал
    plt.figure(figsize=(8, 3))
    plt.plot(t, signal_slice, color='darkblue', linewidth=1)
    plt.title(f'Сырой вибросигнал: {defect_name}')
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(t[0], t[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_raw_signal.png'), dpi=300)
    plt.close()

    # 2. STFT
    f_stft, t_stft, Zxx = scipy.signal.stft(
        signal_slice, fs=FS, nperseg=256, noverlap=128)
    Zxx_db = 20 * np.log10(np.maximum(np.abs(Zxx), 1e-10))
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(t_stft, f_stft, Zxx_db, cmap='magma', shading='gouraud')
    plt.title(f'STFT: {defect_name}')
    plt.colorbar(label='дБ')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_stft.png'), dpi=300)
    plt.close()

    # 3. CWT
    scales = np.arange(1, 129)
    coefs, freqs = pywt.cwt(signal_slice, scales,
                            'cmor1.5-1.0', sampling_period=1/FS)
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(t, freqs, np.abs(coefs), cmap='jet', shading='auto')
    plt.title(f'CWT: {defect_name}')
    plt.colorbar(label='Модуль')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_cwt.png'), dpi=300)
    plt.close()

    # 4. Envelope Spectrum
    analytic = scipy.signal.hilbert(signal_slice)
    env = np.abs(analytic) - np.mean(np.abs(analytic))
    f_env, Pxx_env = scipy.signal.welch(
        env, fs=FS, nperseg=min(1024, len(env)))
    plt.figure(figsize=(8, 3))
    plt.plot(f_env, Pxx_env, color='purple')
    plt.title(f'Спектр огибающей: {defect_name}')
    plt.xlim(0, 2000)  # Самые важные частоты дефектов до 2кГц
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_envelope.png'), dpi=300)
    plt.close()

    # 5. PSD
    f_psd, Pxx_den = scipy.signal.welch(
        signal_slice, fs=FS, nperseg=min(1024, len(signal_slice)))
    plt.figure(figsize=(8, 3))
    plt.semilogy(f_psd, Pxx_den, color='forestgreen')
    plt.title(f'PSD: {defect_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, FS/2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_psd.png'), dpi=300)
    plt.close()


def generate_summary(signals_dict, t, output_dir):
    """Генерация сводных сравнительных графиков для ключевых классов."""
    os.makedirs(output_dir, exist_ok=True)
    # Берем представителей: Норма и начальные стадии (0.007 дюйма) трёх дефектов
    representatives = ['0_Normal', '1_IR_007', '4_Ball_007', '7_OR_007']
    available = [c for c in representatives if c in signals_dict]

    if not available:
        print("  [ВНИМАНИЕ] Не удалось найти нужные классы для сводных графиков.")
        return

    # Подготавливаем 5 холстов (Фигуры)
    fig_raw, axs_raw = plt.subplots(
        len(available), 1, figsize=(10, 2*len(available)), sharex=True)
    fig_stft, axs_stft = plt.subplots(
        len(available), 1, figsize=(10, 2.5*len(available)), sharex=True)
    fig_cwt, axs_cwt = plt.subplots(
        len(available), 1, figsize=(10, 2.5*len(available)), sharex=True)
    fig_env, axs_env = plt.subplots(
        len(available), 1, figsize=(10, 2*len(available)), sharex=True)
    fig_psd, axs_psd = plt.subplots(
        len(available), 1, figsize=(10, 2*len(available)), sharex=True)

    # Если вдруг нашелся только 1 класс (предотвращение ошибки индексации axs)
    if len(available) == 1:
        axs_raw, axs_stft, axs_cwt, axs_env, axs_psd = [
            axs_raw], [axs_stft], [axs_cwt], [axs_env], [axs_psd]

    for i, defect in enumerate(available):
        sig = signals_dict[defect]

        # -- Raw --
        axs_raw[i].plot(t, sig, color='darkblue', linewidth=1)
        axs_raw[i].set_title(f'Сырой сигнал: {defect}', fontsize=12)
        axs_raw[i].grid(True, linestyle='--', alpha=0.7)

        # -- STFT --
        f_stft, t_stft, Zxx = scipy.signal.stft(
            sig, fs=FS, nperseg=256, noverlap=128)
        Zxx_db = 20 * np.log10(np.maximum(np.abs(Zxx), 1e-10))
        pcm_stft = axs_stft[i].pcolormesh(
            t_stft, f_stft, Zxx_db, cmap='magma', shading='gouraud')
        axs_stft[i].set_title(f'STFT: {defect}', fontsize=12)
        fig_stft.colorbar(pcm_stft, ax=axs_stft[i])

        # -- CWT --
        scales = np.arange(1, 129)
        coefs, freqs = pywt.cwt(
            sig, scales, 'cmor1.5-1.0', sampling_period=1/FS)
        pcm_cwt = axs_cwt[i].pcolormesh(
            t, freqs, np.abs(coefs), cmap='jet', shading='auto')
        axs_cwt[i].set_title(f'CWT: {defect}', fontsize=12)
        fig_cwt.colorbar(pcm_cwt, ax=axs_cwt[i])

        # -- Envelope --
        analytic = scipy.signal.hilbert(sig)
        env = np.abs(analytic) - np.mean(np.abs(analytic))
        f_env, Pxx_env = scipy.signal.welch(env, fs=FS, nperseg=1024)
        axs_env[i].plot(f_env, Pxx_env, color='purple')
        axs_env[i].set_title(f'Спектр огибающей: {defect}', fontsize=12)
        # Ограничим до 1000 Гц для наглядности сравнения
        axs_env[i].set_xlim(0, 1000)
        axs_env[i].grid(True, linestyle='--', alpha=0.7)

        # -- PSD --
        f_psd, Pxx_den = scipy.signal.welch(
            sig, fs=FS, nperseg=min(1024, len(sig)))
        axs_psd[i].semilogy(f_psd, Pxx_den, color='forestgreen')
        axs_psd[i].set_title(f'PSD: {defect}', fontsize=12)
        axs_psd[i].grid(True, linestyle='--', alpha=0.7)
        axs_psd[i].set_xlim(0, FS/2)

    # Оформление и сохранение сводных
    axs_raw[-1].set_xlabel('Время, с')
    fig_raw.tight_layout()
    fig_raw.savefig(os.path.join(output_dir, 'summary_raw.png'), dpi=300)
    plt.close(fig_raw)

    axs_stft[-1].set_xlabel('Время, с')
    fig_stft.tight_layout()
    fig_stft.savefig(os.path.join(output_dir, 'summary_stft.png'), dpi=300)
    plt.close(fig_stft)

    axs_cwt[-1].set_xlabel('Время, с')
    fig_cwt.tight_layout()
    fig_cwt.savefig(os.path.join(output_dir, 'summary_cwt.png'), dpi=300)
    plt.close(fig_cwt)

    axs_env[-1].set_xlabel('Частота, Гц')
    fig_env.tight_layout()
    fig_env.savefig(os.path.join(output_dir, 'summary_envelope.png'), dpi=300)
    plt.close(fig_env)

    axs_psd[-1].set_xlabel('Частота, Гц')
    fig_psd.tight_layout()
    fig_psd.savefig(os.path.join(output_dir, 'summary_psd.png'), dpi=300)
    plt.close(fig_psd)


def main():
    print("=== Массовая генерация графиков для диссертации ===")
    data_dir = 'data/raw/CWRU'

    if not os.path.exists(data_dir):
        print(f"Директория {data_dir} не найдена! Проверьте пути.")
        sys.exit(1)

    defect_folders = sorted([f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Найдено {len(defect_folders)} классов дефектов. Начинаю обработку...\n")

    signals_dict = {}
    time_arr = None

    # Проходим по каждой папке дефекта
    for defect in defect_folders:
        defect_path = os.path.join(data_dir, defect)
        mat_files = glob.glob(os.path.join(defect_path, '*.mat'))

        if not mat_files:
            print(f"  [ПРОПУСК] В папке {defect} нет .mat файлов.")
            continue

        # Берем первый попавшийся файл из папки
        file_to_process = mat_files[0]
        try:
            sig, t = load_signal(file_to_process)
            signals_dict[defect] = sig
            time_arr = t

            # Сохраняем в индивидуальную подпапку
            output_dir = os.path.join('reports/figures/individual', defect)
            generate_individual_plots(sig, t, output_dir, defect)
            print(f"  [OK] Графики для {defect} сохранены в {output_dir}")
        except Exception as e:
            print(f"  [ОШИБКА] Не удалось обработать {defect}: {e}")

    # После обработки всех генерируем сводные
    if signals_dict:
        print("\nГенерация сводных графиков (Сравнение: Норма vs Внутреннее vs Шарик vs Внешнее)...")
        summary_dir = 'reports/figures/summary'
        generate_summary(signals_dict, time_arr, summary_dir)
        print(f"  [OK] Сводные графики сохранены в {summary_dir}")

    print("\n=== Все задачи успешно завершены! ===")


if __name__ == '__main__':
    main()
