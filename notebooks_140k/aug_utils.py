"""Utilitários de augmentation/dataset em módulo importável.

Definir estas classes/funções aqui (em vez de numa célula do notebook) permite
usar DataLoader com num_workers>0 no Windows: o método 'spawn' consegue picklar
as classes por referência ao módulo. Classes definidas no __main__ de um notebook
NÃO picklam sob spawn → deadlock. Este módulo resolve isso.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_preprocess(img, method, value):
    """Aplica um método de degradação a uma imagem PIL."""
    if method == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(value))
        buf.seek(0)
        return Image.open(buf).copy()
    if method == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if method == "downscale":
        w, h = img.size
        small = img.resize((max(1, int(w / value)), max(1, int(h / value))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if method == "noise":
        arr = np.array(img, dtype=np.float32)
        arr = arr + np.random.normal(0, float(value) * 255, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if method == "median":
        size = int(value)
        size = size + 1 if size % 2 == 0 else size  # MedianFilter exige tamanho ímpar
        return img.filter(ImageFilter.MedianFilter(size=max(3, size)))
    return img


class FixedPreprocess:
    """Aplica (method, value) FIXO a toda imagem. Usado no sweep (fase 1)."""

    def __init__(self, method, value):
        self.method = method
        self.value = value

    def __call__(self, img):
        return apply_preprocess(img, self.method, self.value)


class RandomAugment:
    """Por imagem: com prob p_apply, sorteia um método do pool e um valor da faixa.

    ranges: dict {method: (low, high)} — faixa uniforme amostrada por imagem.
    """

    def __init__(self, pool, p_apply, ranges):
        self.pool = pool
        self.p_apply = p_apply
        self.ranges = ranges

    def __call__(self, img):
        if random.random() < self.p_apply:
            m = random.choice(self.pool)
            lo, hi = self.ranges[m]
            v = random.uniform(lo, hi)
            img = apply_preprocess(img, m, v)
        return img


class PathListDataset(Dataset):
    """Dataset sobre uma lista fixa de (caminho, label), com transform configurável."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def artifact_split(files, which="test", frac=0.5, seed=777):
    """Particiona deterministicamente uma lista de arquivos em duas metades DISJUNTAS.

    Evita vazamento metodológico: o ArtiFact é usado tanto para *selecionar*
    augmentation/HP (notebooks 01d/01e/01f) quanto para o *número final*
    cross-generator (03/04). Se forem os mesmos dados, o AUC reportado fica
    otimista. Com este split:
      - which='dev'  -> metade de SELEÇÃO (sweep/grid/hp_search, 01x);
      - which='test' -> metade de TESTE FINAL (03/04), nunca vista na seleção.

    A partição é fixa (depende só de `seed`), então é reprodutível e disjunta.
    """
    order = list(files)
    random.Random(seed).shuffle(order)
    cut = int(len(order) * frac)
    return order[:cut] if which == "dev" else order[cut:]


def clean_transform(image_size):
    """Resize + ToTensor + Normalize (sem augmentation). Para eval e ArtiFact."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(image_size, augment=None):
    """Augmentation (opcional) + Resize + flip + ToTensor + Normalize."""
    steps = []
    if augment is not None:
        steps.append(augment)
    steps += [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)
