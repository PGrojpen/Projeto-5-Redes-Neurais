"""Utilitários de augmentation/dataset em módulo importável.

Definir estas classes/funções aqui (em vez de numa célula do notebook) permite
usar DataLoader com num_workers>0 no Windows: o método 'spawn' consegue picklar
as classes por referência ao módulo. Classes definidas no __main__ de um notebook
NÃO picklam sob spawn → deadlock. Este módulo resolve isso.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
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
    # ---- familia espectral cirurgica ----
    if method == "whiten":
        # remocao de picos espectrais: puxa a magnitude de cada frequencia para a MEDIA
        # RADIAL (azimutal) dela. Remove a grade/picos periodicos do gerador, mas preserva
        # o decaimento radial de energia (DC, brilho e a estrutura 1/f natural ficam intactos).
        # value (0..1): 0 = identidade, 1 = espectro radialmente simetrico (sem picos).
        s = float(value)
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.indices((h, w))
        r = np.hypot(xx - w // 2, yy - h // 2).astype(int)
        rmax = int(r.max()) + 1
        nr = np.bincount(r.ravel(), minlength=rmax)
        out = np.empty_like(arr)
        for c in range(3):
            F = np.fft.fftshift(np.fft.fft2(arr[..., c]))
            mag = np.abs(F) + 1e-8
            radial = np.bincount(r.ravel(), mag.ravel(), minlength=rmax) / np.maximum(nr, 1)
            new_mag = mag * (radial[r] / mag) ** s   # s=1 -> magnitude = media radial (sem picos)
            rec = np.fft.ifft2(np.fft.ifftshift(new_mag * np.exp(1j * np.angle(F))))
            out[..., c] = np.real(rec)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if method == "bars":
        # ataque BARS (Wesselkamp et al. 2022): zera uma borda de espessura `value`
        # do espectro centralizado = brick-wall low-pass nas frequencias mais altas,
        # onde mora a digital de upsampling do gerador.
        width = int(value)
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        if width > 0:
            h, w = arr.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)
            mask[:width, :] = 0.0; mask[-width:, :] = 0.0
            mask[:, :width] = 0.0; mask[:, -width:] = 0.0
            for c in range(3):
                F = np.fft.fftshift(np.fft.fft2(arr[..., c]))
                arr[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # ---- familia cor / canais ----
    if method == "grayscale":
        g = float(value)  # blend 0..1 (1 = cinza total)
        gray = img.convert("L").convert("RGB")
        return gray if g >= 1.0 else Image.blend(img.convert("RGB"), gray, g)
    if method == "chanshuffle":
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        p = perms[int(value) % len(perms)]
        return Image.fromarray(np.array(img.convert("RGB"))[..., list(p)])
    if method == "posterize":
        return ImageOps.posterize(img.convert("RGB"), int(value))  # value = bits (1..8)
    if method == "histeq":
        return ImageOps.equalize(img.convert("RGB"))               # value ignorado
    if method == "gamma":
        g = float(value)
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        return Image.fromarray((np.clip(arr ** g, 0, 1) * 255).astype(np.uint8))
    if method == "saturation":
        return ImageEnhance.Color(img.convert("RGB")).enhance(float(value))  # 1=identidade
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
