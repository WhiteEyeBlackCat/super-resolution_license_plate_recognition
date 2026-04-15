import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path


def build_track_dataset(root_dir, num_frames=5, categories=("brazilian", "mercosur")):
    return UFPRPlateTrackDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        categories=categories,
        return_pil=True,
    )


def split_track_dataset(dataset, train_ratio=0.8, seed=42):
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    print(f"total tracks = {n_total}")
    print(f"train tracks = {len(train_dataset)}")
    print(f"val tracks   = {len(val_dataset)}")
    return train_dataset, val_dataset


def build_dataloader(dataset, processor, batch_size=2, shuffle=True):
    collator = TrackCollator(processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )



class TrackCollator:
    def __init__(self, processor):
        self.processor = processor

    def _build_prompt_and_full_text(self, label):
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "只輸出車牌，禁止其他文字。如'ABC1234'，但'您的答案是ABC1234'則是非法回答。車牌會是7個字的英文字母及數字排列組合"},
                ],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "只輸出車牌，禁止其他文字。如'ABC1234'，但'您的答案是ABC1234'則是非法回答。車牌會是7個字的英文字母及數字排列組合"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return prompt_text, full_text

    def __call__(self, batch):
        flat_lr_images = []
        flat_full_texts = []
        flat_prompt_texts = []

        plate_ids = []
        labels = []
        num_views = []

        for sample in batch:
            plate_id = sample["plate_id"]
            label = sample["text_label"]
            lr_imgs = sample["lr_images"]
            hr_imgs = sample.get("hr_images", [])

            prompt_text, full_text = self._build_prompt_and_full_text(label)

            plate_ids.append(plate_id)
            labels.append(label)
            num_views.append(len(lr_imgs))

            for lr_img in lr_imgs:
                flat_lr_images.append(lr_img)
                flat_full_texts.append(full_text)
                flat_prompt_texts.append(prompt_text)

        reg_inputs = self.processor(
            text=flat_full_texts,
            images=flat_lr_images,
            padding=True,
            return_tensors="pt",
        )

        labels_tensor = reg_inputs["input_ids"].clone()
        if "attention_mask" in reg_inputs:
            labels_tensor[reg_inputs["attention_mask"] == 0] = -100

        for i in range(len(flat_lr_images)):
            prompt_inputs = self.processor(
                text=[flat_prompt_texts[i]],
                images=[flat_lr_images[i]],
                padding=False,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels_tensor[i, :prompt_len] = -100

        reg_inputs["labels"] = labels_tensor

        if not hasattr(self, "_debug_done"):
            self._debug_done = True

        return {
            "reg_inputs": reg_inputs,
            "lr_images_flat": flat_lr_images,
            "hr_images_flat": [],
            "plate_ids": plate_ids,
            "text_labels": labels,
            "num_views": num_views,
        }
    


class UFPRPlateTrackDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_frames=5,
        categories=("brazilian", "mercosur"),
        return_pil=False,
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.categories = categories
        self.return_pil = return_pil
        self.samples = []

        for category in self.categories:
            cat_dir = self.root_dir / category
            if not cat_dir.exists():
                continue

            for plate_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
                if self._is_valid_plate_dir(plate_dir):
                    self.samples.append(plate_dir)

        print(f"[UFPRPlateTrackDataset] valid tracks = {len(self.samples)}")

    def _is_valid_plate_dir(self, plate_dir):
        if not (plate_dir / "hr-001.json").exists():
            return False

        for i in range(1, self.num_frames + 1):
            idx = f"{i:03d}"
            if not (plate_dir / f"lr-{idx}.png").exists():
                return False
        return True

    def _load_label(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ["plate", "text", "label", "value"]:
            if key in data and isinstance(data[key], str):
                return data[key].strip().upper()

        raise KeyError(f"Cannot find plate text in {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        plate_dir = self.samples[idx]
        label = self._load_label(plate_dir / "hr-001.json")

        lr_imgs = []
        for i in range(1, self.num_frames + 1):
            frame_idx = f"{i:03d}"
            lr_path = plate_dir / f"lr-{frame_idx}.png"

            lr_img = Image.open(lr_path).convert("RGB")

            lr_imgs.append(lr_img)

        return {
            "plate_id": f"{plate_dir.parent.name}/{plate_dir.name}",
            "text_label": label,
            "lr_images": lr_imgs,   # list[PIL], len=5
            "hr_images": [],        # MVC 版本不需要 HR 圖
        }
