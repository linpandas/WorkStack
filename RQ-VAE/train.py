from torch.utils.data import Dataset, DataLoader
import torch
from rqvae import RQVAE
import os
import numpy as np
import json
import argparse


def get_cache_dir(file_path: str, cache_dir: str = None) -> str:
    if cache_dir:
        return cache_dir
    return f"{file_path}.cache"


def get_cache_paths(cache_dir: str):
    return {
        "embeddings": os.path.join(cache_dir, "embeddings.dat"),
        "ids": os.path.join(cache_dir, "ids.txt"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def get_source_signature(file_path: str):
    stat = os.stat(file_path)
    return {
        "source_path": os.path.abspath(file_path),
        "source_size": stat.st_size,
        "source_mtime": stat.st_mtime,
    }


def load_cache_metadata(cache_dir: str):
    metadata_path = get_cache_paths(cache_dir)["metadata"]
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_is_valid(file_path: str, cache_dir: str) -> bool:
    paths = get_cache_paths(cache_dir)
    if not all(os.path.exists(path) for path in paths.values()):
        return False

    metadata = load_cache_metadata(cache_dir)
    if not metadata:
        return False

    source_signature = get_source_signature(file_path)
    for key, value in source_signature.items():
        if metadata.get(key) != value:
            return False

    expected_size = metadata["num_rows"] * metadata["dim"] * np.dtype(np.float32).itemsize
    return os.path.getsize(paths["embeddings"]) == expected_size


def build_memmap_cache(file_path: str, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    paths = get_cache_paths(cache_dir)

    tmp_embeddings_path = f"{paths['embeddings']}.tmp"
    tmp_ids_path = f"{paths['ids']}.tmp"
    tmp_metadata_path = f"{paths['metadata']}.tmp"

    num_rows = 0
    dim = None

    print(f"Building embedding cache at {cache_dir}")
    with open(file_path, "r", encoding="utf-8") as input_file, \
            open(tmp_embeddings_path, "wb") as embeddings_file, \
            open(tmp_ids_path, "w", encoding="utf-8") as ids_file:
        for line_num, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            item_id = parts[0]
            vec_str = parts[1].strip().strip("[]")
            vec = np.fromstring(vec_str, sep=",", dtype=np.float32)
            if vec.size == 0:
                continue

            if dim is None:
                dim = vec.size
            elif vec.size != dim:
                raise ValueError(
                    f"Line {line_num} has dim {vec.size}, expected {dim}"
                )

            vec.tofile(embeddings_file)
            ids_file.write(f"{item_id}\n")
            num_rows += 1

            if num_rows % 100000 == 0:
                print(f"Cached {num_rows} embeddings")

    if num_rows == 0:
        raise ValueError(f"No valid embeddings loaded from {file_path}")

    metadata = {
        **get_source_signature(file_path),
        "num_rows": num_rows,
        "dim": dim,
        "dtype": "float32",
    }
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    os.replace(tmp_embeddings_path, paths["embeddings"])
    os.replace(tmp_ids_path, paths["ids"])
    os.replace(tmp_metadata_path, paths["metadata"])

    print(f"Cache ready: {num_rows} embeddings with dimension {dim}")

def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


class EmbeddingDataset(Dataset):
    """从memmap缓存读取embedding数据，避免一次性占用大量内存"""

    def __init__(self, file_path: str, cache_dir: str = None, rebuild_cache: bool = False):
        self.file_path = file_path
        self.cache_dir = get_cache_dir(file_path, cache_dir)

        if rebuild_cache or not cache_is_valid(file_path, self.cache_dir):
            build_memmap_cache(file_path, self.cache_dir)

        self.paths = get_cache_paths(self.cache_dir)
        metadata = load_cache_metadata(self.cache_dir)
        self.num_rows = metadata["num_rows"]
        self.dim = metadata["dim"]
        self.ids_path = self.paths["ids"]
        self._data = None

        print(f"Loaded memmap cache: {self.num_rows} embeddings with dimension {self.dim}")

    @property
    def data(self):
        if self._data is None:
            self._data = np.memmap(
                self.paths["embeddings"],
                dtype=np.float32,
                mode="r",
                shape=(self.num_rows, self.dim),
            )
        return self._data

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())


def train_rqvae(
        data_file: str,
        model_save_path: str = 'rqvae_model.pt',
        batch_size: int = 4096,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir: str = None,
        rebuild_cache: bool = False,
        num_workers: int = 2,
):
    """训练RQ-VAE模型"""

    # 加载数据
    dataset = EmbeddingDataset(
        data_file,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
        persistent_workers=num_workers > 0,
    )

    # 创建模型
    model = RQVAE(n_embed=[1024, 128, 16], embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)

            # Forward
            recon, quant_loss, code = model(batch)
            loss_dict = model.compute_loss(recon, quant_loss, code, batch)
            loss = loss_dict['loss_total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_dict['loss_recon'].item()
            total_quant_loss += loss_dict['loss_latent'].item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_quant = total_quant_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Loss: {avg_loss:.6f} "
              f"Recon: {avg_recon:.6f} "
              f"Quant: {avg_quant:.6f}")

        # 每50个epoch保存一次
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)
            print(f"Model saved to {model_save_path}")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    print(f"Final model saved to {model_save_path}")

    return model

def test_rqvae(
        model: RQVAE,
        data_file: str,
        device: str = 'cpu',
        cache_dir: str = None,
        num_workers: int = 2,
):
    """测试模型并检查codebook使用率"""
    dataset = EmbeddingDataset(data_file, cache_dir=cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
        persistent_workers=num_workers > 0,
    )

    model.eval()
    model.to(device)

    all_codes = [[] for _ in range(model.quantizer.n_layer)]

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            codes = model.get_codes(batch)

            for i in range(codes.shape[-1]):
                all_codes[i].extend(codes[:, i].cpu().numpy().tolist())

    # 检查每个量化器的codebook使用率
    print("\n=== Codebook Usage Statistics ===")
    for i, codes in enumerate(all_codes):
        unique_codes = len(set(codes))
        n_embed = model.quantizer.n_embed[i]
        usage_rate = unique_codes / n_embed * 100
        print(f"Quantizer {i + 1}: {unique_codes}/{n_embed} "
              f"codes used ({usage_rate:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE from embedding.txt.")
    parser.add_argument(
        "--data-file",
        default=os.environ.get("DATA_FILE", "/ossfs/node_70805357/workspace/embedding.txt"),
        help="Embedding file. Format: supply_code:[dim1,dim2,...]",
    )

    parser.add_argument(
        "--model-save-path",
        default=os.environ.get("MODEL_SAVE_PATH", "/ossfs/node_70805357/workspace/rqvae_model.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "4096")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "1000")))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("LEARNING_RATE", "1e-4")),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR"))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "2")),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        default=str_to_bool(os.environ.get("REBUILD_CACHE", "0")),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=str_to_bool(os.environ.get("SKIP_TEST", "0")),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if os.path.exists(args.data_file):
        model = train_rqvae(
            data_file=args.data_file,
            model_save_path=args.model_save_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
            num_workers=args.num_workers,
        )

        if not args.skip_test:
            test_rqvae(
                model,
                args.data_file,
                device=args.device,
                cache_dir=args.cache_dir,
                num_workers=args.num_workers,
            )
    else:
        print(f"Please create a data file at {args.data_file}")
        print("Format: id:[dim1,dim2,...,dim4096]")
        print("Example: sample_001:[0.1,0.2,0.3,...,0.5]")