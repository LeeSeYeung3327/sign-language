import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class KeypointAugmentation(object):
    """
    키포인트에 가우시안 노이즈를 추가하는 간단한 증강 클래스.
    noise_std: 노이즈의 표준편차, prob: 적용 확률.
    """
    def __init__(self, noise_std=0.01, prob=0.5):
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, x):
        # x의 shape: (T, 1, 42, 2)
        if np.random.rand() < self.prob:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
        return x

class SignLanguageDataset(Dataset):
    def __init__(self, aggregated, sequence_length=64, transform=None):
        """
        aggregated: {"data": [...], "label_map": {...}} 형식의 딕셔너리.
        sequence_length: 각 샘플의 프레임 수.
        transform: 데이터 증강이나 전처리를 위한 추가 transform.
        """
        self.data = aggregated['data']
        self.label_map = aggregated['label_map']
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_str = item['label']
        keypoints = item['keypoints']  # 전체 길이는 sequence_length * 42 * 2 이어야 함.
        # (시퀀스 길이, 42, 2)로 복원
        kp = np.array(keypoints, dtype=np.float32).reshape(self.sequence_length, 42, 2)
        # 채널 차원 추가: (T, 1, 42, 2)
        kp = np.expand_dims(kp, axis=1)
        if self.transform is not None:
            kp = self.transform(kp)
        kp = torch.tensor(kp, dtype=torch.float32)
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        return kp, label

def load_aggregated_csv(csv_path, encoding='utf-8'):
    """
    CSV 파일에서 키포인트와 라벨 정보를 읽어 aggregated 형태의 딕셔너리로 반환합니다.
    """
    data = []
    with open(csv_path, 'r', newline='', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            kp_keys = [k for k in row.keys() if k.startswith('kp_')]
            kp_keys.sort(key=lambda x: int(x.split('_')[1]))
            kp_data = [float(row[k]) for k in kp_keys]
            data.append({'label': label, 'keypoints': kp_data})
    unique_labels = sorted(list(set(item['label'] for item in data)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    aggregated = {'data': data, 'label_map': label_map}
    return aggregated
