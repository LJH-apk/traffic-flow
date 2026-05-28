"""省份字符分类器（31类）—— 模型定义 + 推理接口。

训练好的权重保存在同目录下的 province_clf.pt。
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

PROVINCE_CHARS = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
IMG_SIZE = 48
_LABELS  = list(PROVINCE_CHARS)
_WEIGHT  = Path(__file__).parent / "province_clf.pt"

_CONF_THRESH = 0.85  # 低于此阈值时不覆盖 HyperLPR3 结果


class ProvinceCLF(nn.Module):
    def __init__(self, n_classes: int = 31):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 24×24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 12×12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                 # 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


_model_cache: "ProvinceCLF | None" = None


def load_model(device: str = "cpu") -> "ProvinceCLF | None":
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _WEIGHT.exists():
        return None
    m = ProvinceCLF(n_classes=len(_LABELS))
    m.load_state_dict(torch.load(_WEIGHT, map_location=device, weights_only=True))
    m.eval()
    m.to(device)
    _model_cache = m
    return m


def predict(gray_crop: np.ndarray, device: str = "cpu") -> tuple[str, float]:
    """输入灰度省份字符 crop（任意尺寸），返回 (省份字符, 置信度)。

    模型未加载时返回 ("", 0.0)。
    """
    m = load_model(device)
    if m is None:
        return "", 0.0
    img = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t   = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(m(t.to(device)), dim=1)[0]
    idx = int(probs.argmax())
    return _LABELS[idx], float(probs[idx])


def is_available() -> bool:
    return _WEIGHT.exists()
