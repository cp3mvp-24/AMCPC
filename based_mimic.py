import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 数据读取 & 预处理（动态特征 + 静态特征）
# -------------------------

def load_and_preprocess_mimic(
    csv_path: str,
    patient_col: str,
    label_col: str,
    dynamic_feature_cols: List[str],  # 只包含动态特征
    static_feature_cols: List[str],   # 只包含静态特征
    min_seq_len: int = 16,
    max_seq_len: int = 10000,
) -> Tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    """
    读取 MIMIC-III EHR 数据，按病人过滤长度，并对动态特征做 z-score 标准化。
    还对静态特征（性别、年龄、体重、身高）进行编码和标准化处理。
    """
    df = pd.read_csv(csv_path)

    grouped = df.groupby(patient_col)
    filtered_df = grouped.filter(
        lambda g: (len(g) >= min_seq_len) and (len(g) <= max_seq_len)
    )

    # 动态特征标准化
    scaler_dynamic = StandardScaler()
    filtered_df[dynamic_feature_cols] = scaler_dynamic.fit_transform(filtered_df[dynamic_feature_cols])

    # 静态特征处理
    # Sex: One-Hot 编码（0 男性，1 女性）
    filtered_df['Sex'] = filtered_df['Sex'].map({0: 0, 1: 1})  # 假设 Sex 只包含 0, 1
    scaler_static = StandardScaler()
    filtered_df[static_feature_cols] = scaler_static.fit_transform(filtered_df[static_feature_cols])

    return filtered_df, scaler_dynamic, scaler_static


def build_patient_sequences_for_segment_prediction(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    dynamic_feature_cols: List[str],
    static_feature_cols: List[str],
    outer_len: int,
    inner_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    切分病人序列为 4 段，每段内长度为 inner_len（例如 4），
    用前 3 段预测第 4 段的目标。
    另外，将静态特征添加到每个子窗口。
    """
    assert outer_len % inner_len == 0, "outer_len 必须能被 inner_len 整除"

    windows = []
    window_to_patient = []
    patient_labels = []
    patient_ids = []
    static_features_list = []

    grouped = df.groupby(patient_col)

    for pid, g in grouped:
        seq = g[dynamic_feature_cols].values  # (T_total, C)
        static = g[static_feature_cols].iloc[0].values  # (static_dim,)
        T_total = len(seq)
        if T_total < outer_len:
            continue

        # 记录病人（一次）
        patient_ids.append(pid)
        patient_labels.append(g[label_col].iloc[0])
        pid_idx = len(patient_ids) - 1

        # 外层分块
        num_outer = T_total // outer_len
        for o in range(num_outer):
            outer_start = o * outer_len
            outer_end = outer_start + outer_len
            outer_chunk = seq[outer_start:outer_end]  # (outer_len, C)

            # 内层分块
            num_inner = outer_len // inner_len
            for j in range(num_inner - 1):  # 只取前三段
                inner_start = j * inner_len
                inner_end = inner_start + inner_len
                inner_chunk = outer_chunk[inner_start:inner_end]  # (inner_len, C)

                windows.append(inner_chunk)
                window_to_patient.append(pid_idx)
                static_features_list.append(static)  # 添加静态特征

            # 最后一段（第 4 段）作为目标，不会参与预测
            target_chunk = outer_chunk[-inner_len:]  # (inner_len, C)
            windows.append(target_chunk)
            window_to_patient.append(pid_idx)
            static_features_list.append(static)  # 添加静态特征

    X_windows = np.stack(windows, axis=0)  # (num_windows, inner_len, C)
    window_to_patient = np.array(window_to_patient, dtype=np.int64)
    y_patient = np.array(patient_labels)
    static_features = np.array(static_features_list)  # (num_windows, static_dim)

    return X_windows, window_to_patient, y_patient, static_features


# -------------------------
# Dataset 定义
# -------------------------

class WindowDataset(Dataset):
    """子窗口数据，用于 AMCPC 自监督预训练。"""

    def __init__(self, windows: np.ndarray, static_features: np.ndarray):
        self.x = torch.from_numpy(windows).float()
        self.static_features = torch.from_numpy(static_features).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.static_features[idx]  # 返回动态特征和静态特征


class PatientSequenceDataset(Dataset):
    """
    病人级别的窗口 embedding 序列，用于 LSTM 下游分类。
    每个样本是一个变长序列 (num_windows_i, D)。
    """

    def __init__(self, seqs: List[np.ndarray], labels: np.ndarray, static_features: np.ndarray, patient_indices: np.ndarray):
        self.seqs = [seqs[i] for i in patient_indices]
        self.labels = labels[patient_indices]
        self.static_features = [static_features[i] for i in patient_indices]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.seqs[idx]).float()  # (T_i, D)
        label = torch.tensor(self.labels[idx]).long()
        static = torch.from_numpy(self.static_features[idx]).float()
        return seq, label, static


def collate_fn(batch):
    """
    collate_fn 用于把变长序列打包成一个 batch：
      - 对序列用 pad_sequence 对齐（只是为了张量对齐，不是缺失值填充）
      - 记录每个序列的真实长度 lengths
      - 按长度从大到小排序，方便 pack_padded_sequence
    """
    seqs, labels, static = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)  # (B,)

    # 按长度排序
    lengths, perm_idx = lengths.sort(descending=True)
    seqs = [seqs[i] for i in perm_idx]
    labels = torch.tensor([labels[i] for i in perm_idx], dtype=torch.long)
    static = [static[i] for i in perm_idx]

    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (B, T_max, D)
    return padded, lengths, labels, static


# -------------------------
# AMCPC 模型组件
# -------------------------

class MultiIndicatorEncoder(nn.Module):
    """
    2D encoder g_enc。

    输入:  x, 形状 (B, T, C)
      - B: batch size
      - T: 子窗口内部时间长度（这里就是 inner_len，例如 4）
      - C: 指标数量（医疗特征数）

    输出: latent 庈序列 z, 形状 (B, T, d_model)
    """

    def __init__(self, num_features: int, d_model: int = 128, kernel_size: int = 3):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.kernel_size = kernel_size

        pad_t = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(kernel_size, num_features),
            padding=(pad_t, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        assert C == self.num_features, "Input feature dim does not match num_features."

        x = x.unsqueeze(1)        # (B, 1, T, C)
        x = self.conv(x)          # (B, d_model, T, 1)
        x = x.squeeze(-1)         # (B, d_model, T)
        x = x.transpose(1, 2)     # (B, T, d_model)
        return x


class AutoregressiveModel(nn.Module):
    """
    自回归模块 g_ar（LSTM），对 z_t 序列做编码，得到上下文 c_t。
    """

    def __init__(self, d_model: int = 128, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(z)   # (B, T, hidden_size)
        return output


class AMCPC(nn.Module):
    """
    Multi-indicator CPC 模块（不含静态特征 fusion）。

    给定子窗口 x, 形状 (B, T, C)，输出:
      - z: g_enc(x) -> (B, T, d_model)
      - c: g_ar(z)  -> (B, T, hidden_size)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        prediction_steps: int = 3,
    ):
        super().__init__()
        self.encoder = MultiIndicatorEncoder(num_features, d_model)
        self.ar = AutoregressiveModel(d_model, hidden_size, num_layers)
        self.prediction_steps = prediction_steps
        self.hidden_size = hidden_size
        self.d_model = d_model

        self.wk = nn.ModuleList(
            [nn.Linear(hidden_size, d_model) for _ in range(prediction_steps)]
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)   # (B, T, d_model)
        c = self.ar(z)        # (B, T, hidden_size)
        return z, c


# -------------------------
# 下游 LSTM 分类模型
# -------------------------

class LSTMClassifier(nn.Module):
    """
    下游用 LSTM 对“窗口序列”做建模，最后一层最后一个 hidden state 用来分类。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # 融合静态特征
        x = torch.cat([x, static.unsqueeze(1).repeat(1, x.size(1), 1)], dim=-1)  # (B, T_max, D + static_dim)

        # 打包变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        if self.num_directions == 1:
            last = h_n[-1]  # (B, hidden_dim)
        else:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*hidden_dim)

        out = self.dropout(last)
        logits = self.fc(out)
        return logits


def train_lstm_classifier(
    patient_seqs: List[np.ndarray],
    labels: np.ndarray,
    static_features: np.ndarray,
    device: torch.device,
    hidden_dim: int = 128,
    num_layers: int = 1,
    bidirectional: bool = False,
    test_size: float = 0.2,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    patient_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        patient_indices,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    train_ds = PatientSequenceDataset(patient_seqs, y_encoded, static_features, train_idx)
    test_ds = PatientSequenceDataset(patient_seqs, y_encoded, static_features, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    input_dim = patient_seqs[0].shape[1] + static_features.shape[1]  # 动态 + 静态特征
    num_classes = len(np.unique(y_encoded))

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, lengths, y_batch, static_batch in train_loader:
            x_batch = x_batch.to(device)
            lengths = lengths.to(device)
            y_batch = y_batch.to(device)
            static_batch = static_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch, lengths, static_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, lengths, y_batch, static_batch in test_loader:
                x_batch = x_batch.to(device)
                lengths = lengths.to(device)
                y_batch = y_batch.to(device)
                static_batch = static_batch.to(device)

                logits = model(x_batch, lengths, static_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total if total > 0 else 0.0
        print(
            f"[Epoch {epoch:03d}] "
            f"LSTM Classifier Train Loss: {avg_train_loss:.4f} | Test Acc: {acc:.4f}"
        )


# -------------------------
# 主入口
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMCPC on MIMIC-III (动态特征 + 静态特征，LSTM downstream)"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to preprocessed MIMIC CSV.")
    parser.add_argument("--outer_len", type=int, default=16, help="外层长度（例如 16）")
    parser.add_argument("--inner_len", type=int, default=4, help="内层窗口长度（例如 4）")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_cpc", type=int, default=100)
    parser.add_argument("--epochs_cls", type=int, default=50)
    parser.add_argument("--lr_cpc", type=float, default=1e-3)
    parser.add_argument("--lr_cls", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prediction_steps", type=int, default=3)
    parser.add_argument("--patient_col", type=str, default="PatientID")
    parser.add_argument("--label_col", type=str, default="Numeric_Label")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--lstm_bidirectional", action="store_true")
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "Diastolic blood pressure",
            "Fraction inspired oxygen",
            "Glucose",
            "Heart Rate",
            "Mean blood pressure",
            "Oxygen saturation",
            "Respiratory rate",
            "Systolic blood pressure",
            "Temperature",
            "pH",
        ],
        help="List of dynamic feature column names.",
    )
    parser.add_argument(
        "--static_features",
        nargs="+",
        default=["Sex", "Age", "Weight", "Height"],
        help="List of static feature column names.",
    )
    args = parser.parse_args()

    assert args.outer_len % args.inner_len == 0, "outer_len 必须能被 inner_len 整除"
    assert args.prediction_steps < args.inner_len, "prediction_steps 应该小于 inner_len"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1）读取 & 预处理数据（包括静态特征）
    df, _, _ = load_and_preprocess_mimic(
        csv_path=args.csv_path,
        patient_col=args.patient_col,
        label_col=args.label_col,
        dynamic_feature_cols=args.features,
        static_feature_cols=args.static_features,
        min_seq_len=args.outer_len,
    )

    # 2）切分病人数据（16 步 -> 4 段）
    X_windows, window_to_patient, y_patient, static_features = build_patient_sequences_for_segment_prediction(
        df,
        patient_col=args.patient_col,
        label_col=args.label_col,
        dynamic_feature_cols=args.features,
        static_feature_cols=args.static_features,
        outer_len=args.outer_len,
        inner_len=args.inner_len,
    )
    num_patients = y_patient.shape[0]
    print(f"Total patients: {num_patients}, total windows: {X_windows.shape[0]}")

    # 3）训练集与验证集
    patient_indices = np.arange(num_patients)
    train_patients, val_patients = train_test_split(
        patient_indices, test_size=0.2, random_state=42, stratify=y_patient
    )

    train_mask = np.isin(window_to_patient, train_patients)
    val_mask = ~train_mask

    X_train_windows = X_windows[train_mask]
    X_val_windows = X_windows[val_mask]

    train_dataset = WindowDataset(X_train_windows, static_features[train_mask])
    val_dataset = WindowDataset(X_val_windows, static_features[val_mask])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 4）初始化 AMCPC
    num_features = len(args.features)
    amcpc = AMCPC(
        num_features=num_features,
        d_model=128,
        hidden_size=128,
        num_layers=1,
        prediction_steps=args.prediction_steps,
    ).to(device)

    # 5）自监督预训练
    train_amcpc(
        model=amcpc,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs_cpc,
        lr=args.lr_cpc,
        temperature=args.temperature,
        patience=10,
        save_path="amcpc_mimic_pretrained.pt",
    )

    amcpc.load_state_dict(torch.load("amcpc_mimic_pretrained.pt", map_location=device))

    # 6）提取每个病人的 embedding（窗口）
    patient_seqs = extract_patient_window_embeddings(
        model=amcpc,
        windows=X_windows,
        window_to_patient=window_to_patient,
        num_patients=num_patients,
        device=device,
        batch_size=args.batch_size,
    )

    # 7）LSTM 分类
    train_lstm_classifier(
        patient_seqs=patient_seqs,
        labels=y_patient,
        static_features=static_features,
        device=device,
        hidden_dim=args.lstm_hidden,
        num_layers=args.lstm_layers,
        bidirectional=args.lstm_bidirectional,
        test_size=args.test_size,
        num_epochs=args.epochs_cls,
        lr=args.lr_cls,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
