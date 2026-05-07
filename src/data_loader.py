import os
import torch
from torch.utils.data import Dataset
import scipy.io
import scipy.signal
import numpy as np

class CWRUDataset(Dataset):
    def __init__(self, data_dir='data/raw', window_size=1024):
        self.data_dir = data_dir
        self.window_size = window_size
        self.spectrograms = []
        self.labels = []
        
        self._load_data()
        
    def _load_data(self):
        for root, dirs, files in os.walk(self.data_dir):
            folder_name = os.path.basename(root)
            if not folder_name:
                continue
                
            # Извлекаем метку из первого символа имени папки, если это цифра
            if not folder_name[0].isdigit():
                continue
                
            label = int(folder_name[0])
            
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = scipy.io.loadmat(file_path)
                    
                    # Ищем ключ, оканчивающийся на '_DE_time' и игнорируем системные ключи
                    signal = None
                    for key in mat_data.keys():
                        if not key.startswith('__') and key.endswith('_DE_time'):
                            signal = mat_data[key].flatten()
                            break
                            
                    if signal is None:
                        continue
                        
                    # Нарезаем сигнал на неперекрывающиеся окна
                    num_windows = len(signal) // self.window_size
                    for i in range(num_windows):
                        start_idx = i * self.window_size
                        end_idx = start_idx + self.window_size
                        window = signal[start_idx:end_idx]
                        
                        # Применяем STFT
                        f, t, Zxx = scipy.signal.stft(window, nperseg=256, noverlap=128)
                        
                        # Вычисляем квадрат модуля спектрограммы
                        spectrogram = np.abs(Zxx) ** 2
                        
                        self.spectrograms.append(spectrogram)
                        self.labels.append(label)

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        
        # Конвертируем в тензоры и добавляем канал для спектрограммы: [1, Freq, Time]
        tensor_spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        tensor_label = torch.tensor(label, dtype=torch.long)
        
        return tensor_spectrogram, tensor_label

if __name__ == '__main__':
    print("Инициализация датасета CWRU...")
    # Путь по умолчанию предполагает, что скрипт запускается из корня проекта
    dataset = CWRUDataset(data_dir='data/raw', window_size=1024)
    
    print(f"Общее количество примеров в датасете: {len(dataset)}")
    
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Размерность одного тензора спектрограммы: {spec.shape}")
        print(f"Метка класса первого примера: {label}")
    else:
        print("Датасет пуст. Убедитесь, что в data/raw/ есть подпапки с .mat файлами.")
