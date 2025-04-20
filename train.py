import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from torchvision import transforms
from dataloader import SignLanguageDataset, load_aggregated_csv, KeypointAugmentation

# CNN-LSTM 모델 정의
class CNN_LSTMModel(nn.Module):
    def __init__(self, cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5):
        """
        cnn_out_dim: CNN 출력 후 차원 축소된 특징 (예: 256)
        lstm_hidden_size: LSTM 은닉 상태 차원 (예: 128)
        num_classes: 분류할 수어 단어 개수
        dropout: 드롭아웃 비율
        """
        super(CNN_LSTMModel, self).__init__()
        # CNN 부분: 각 프레임 입력은 (1, 42, 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        # CNN 출력: (32, 42, 2) → flatten: 32*42*2 = 2688
        self.cnn_fc = nn.Sequential(
            nn.Linear(32 * 42 * 2, cnn_out_dim),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU()
        )
        # LSTM 부분
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        # 최종 분류를 위한 FC 네트워크
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, 1, 42, 2)
        B, T, C, H, W = x.size()
        # CNN은 개별 프레임마다 적용: (B×T, C, H, W)
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)  # (B×T, 32, 42, 2)
        cnn_out = cnn_out.view(B * T, -1)  # flatten
        cnn_features = self.cnn_fc(cnn_out)  # (B×T, cnn_out_dim)
        # 다시 시퀀스 형태로 복원: (B, T, cnn_out_dim)
        cnn_features = cnn_features.view(B, T, -1)
        # LSTM 처리: (B, T, lstm_hidden_size)
        lstm_out, _ = self.lstm(cnn_features)
        # 시퀀스의 마지막 타임스텝 출력 사용
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

def train():
    aggregated_csv = 'csv_output/aggregated_dataset.csv'
    if not os.path.exists(aggregated_csv):
        raise FileNotFoundError("Aggregated dataset not found. 먼저 키포인트 추출 스크립트를 실행하세요.")
    
    aggregated = load_aggregated_csv(aggregated_csv, encoding='utf-8')
    print("Label Map:", aggregated['label_map'])
    
    sequence_length = 64

    # 학습용 transform: 높은 확률로 노이즈 증강을 수행하여 데이터 다양성 확보
    train_transform = transforms.Compose([
        KeypointAugmentation(noise_std=0.005, prob=0.8)
    ])
    # 검증용 transform은 None (증강 없이 원본 그대로 사용)
    val_transform = None

    # 학습 데이터셋
    dataset = SignLanguageDataset(aggregated, sequence_length=sequence_length, transform=train_transform)
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.8 * total_samples)
    train_dataset = Subset(dataset, indices[:split])
    # 검증 데이터셋은 증강 없이 로드
    val_dataset = Subset(SignLanguageDataset(aggregated, sequence_length=sequence_length, transform=val_transform), indices[split:])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 설정
    cnn_out_dim = 256
    lstm_hidden_size = 128
    num_classes = len(aggregated['label_map'])
    model = CNN_LSTMModel(cnn_out_dim=cnn_out_dim, lstm_hidden_size=lstm_hidden_size,
                          num_classes=num_classes, dropout=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    early_stop_patience = 15
    num_epochs = 150

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_correct += (outputs.argmax(dim=1) == batch_y).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model_cnn_lstm.pth')
            print("✅ Best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                break

    torch.save(model.state_dict(), 'final_sign_model_cnn_lstm.pt')
    print("Final model saved.")

if __name__ == '__main__':
    train()
