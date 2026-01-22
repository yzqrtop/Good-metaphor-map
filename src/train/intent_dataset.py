from pathlib import Path
from typing import List, Tuple
import json

from torch.utils.data import Dataset


class IntentDataset(Dataset):
    def __init__(self, path: Path):
        """Initialize intent classification dataset"""
        self.data = self.load_jsonl(path)

    def load_jsonl(self, path: Path) -> List[dict]:
        """Load JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        return data

    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get single sample"""
        sample = self.data[idx]
        # Concatenate all fields as input text
        text = " ".join([sample.get(k, "") for k in ["E", "V", "EE", "VE", "C", "Sc", "Sy"]])
        # Get intent label
        intent = sample.get("intent", "unknown")
        return text, intent