import torch
import torch.nn as nn

class LightweightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        
        # Полностью связные слои (Fully Connected Layers)
        # Расчет размерности:
        # Freq: 129 -> pool1 -> 64 -> pool2 -> 32 -> pool3 -> 16
        # Time: 9 -> pool1 -> 4 -> pool2 -> 2 -> pool3 -> 1
        # Итог после Flatten: 64 канала * 16 * 1 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Создаем объект сети
    model = LightweightCNN(num_classes=10)
    print("Архитектура LightweightCNN:")
    print(model)
    
    # Создаем dummy input (случайный тензор)
    # Размерность: [batch_size, channels, freq_bins, time_frames]
    # На основе STFT (nperseg=256, noverlap=128) из окна 1024 получаем спектрограмму 129x9
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 129, 9)
    
    print(f"\nВходная размерность: {dummy_input.shape}")
    
    try:
        # Прогоняем тензор через сеть
        output = model(dummy_input)
        print(f"Выходная размерность: {output.shape}")
        
        if output.shape == (batch_size, 10):
            print("Успех! Forward pass прошел без ошибок размерностей (Shape errors).")
        else:
            print("Предупреждение: Неожиданная размерность выхода.")
    except Exception as e:
        print(f"Ошибка при выполнении forward pass: {e}")
