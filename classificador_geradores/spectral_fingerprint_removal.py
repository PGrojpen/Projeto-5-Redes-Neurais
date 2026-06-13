"""
spectral_fingerprint_removal.py
================================

Reimplementação enxuta (numpy/PIL, sem TensorFlow) de dois ataques de remoção
de fingerprint de GAN no domínio da frequência, inspirados em:

    Wesselkamp et al., "Misleading Deep-Fake Detection with GAN Fingerprints",
    DLS 2022.  https://arxiv.org/abs/2205.12543

Dois ataques implementados:

  1. BARS (untargeted) — não precisa de nada pré-calculado.
     Zera uma faixa de altas frequências (borda do espectro centralizado).
     É onde os artefatos de upsampling do gerador costumam concentrar energia.
     Único hiperparâmetro: `width` (largura da borda removida).

  2. MEAN (targeted) — precisa de um conjunto de fakes + um de reais.
     Fingerprint = média(espectro_fake) - média(espectro_real), por bin de
     frequência (complexo). A remoção subtrai `factor * fingerprint` do
     espectro de cada fake. `factor` controla a força do ataque.

Uso típico no contexto de detecção de deepfake:
  - FASE 1 (diagnóstico): aplique nos fakes do conjunto de TESTE, recalcule a
    AUC do detector e meça a queda. Queda grande => o modelo lia o fingerprint,
    não a semântica.
  - FASE 2 (mitigação): use a mesma transformação como data augmentation no
    TREINO (aplicada com probabilidade p a cada batch) para forçar o detector
    a aprender features que sobrevivem à remoção, melhorando o cross-dataset.

CLI:
  # Ataque bars numa pasta de fakes
  python spectral_fingerprint_removal.py bars FAKES_DIR -o OUT_DIR --width 12

  # 1) calcular o fingerprint mean a partir de fakes + reais
  python spectral_fingerprint_removal.py fit-mean FAKES_DIR REAIS_DIR -o fp.npy --size 256
  # 2) aplicar o ataque mean
  python spectral_fingerprint_removal.py mean FAKES_DIR fp.npy -o OUT_DIR --factor 1.0

  # medir PSNR médio entre original e manipulado (para calibrar força)
  python spectral_fingerprint_removal.py psnr ORIG_DIR MANIP_DIR
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Carrega imagem como float32 em [0,1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: str | Path) -> None:
    """Salva array float [0,1] (H,W,3) como uint8."""
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------- #
# Domínio da frequência (por canal)                                           #
# --------------------------------------------------------------------------- #
def to_spectrum(img: np.ndarray) -> np.ndarray:
    """FFT2 centralizada (fftshift) por canal. Retorna complexo (H,W,3)."""
    spec = np.fft.fft2(img, axes=(0, 1))
    return np.fft.fftshift(spec, axes=(0, 1))


def from_spectrum(spec: np.ndarray) -> np.ndarray:
    """Inverte to_spectrum. Retorna imagem real (H,W,3)."""
    spec = np.fft.ifftshift(spec, axes=(0, 1))
    img = np.fft.ifft2(spec, axes=(0, 1))
    return np.real(img)


# --------------------------------------------------------------------------- #
# Ataque 1: BARS (untargeted)                                                 #
# --------------------------------------------------------------------------- #
def bars_mask(h: int, w: int, width: int) -> np.ndarray:
    """
    Máscara (H,W) com 0 numa borda de espessura `width` (altas frequências,
    pós-fftshift => bordas do array) e 1 no resto.
    """
    mask = np.ones((h, w), dtype=np.float32)
    if width <= 0:
        return mask
    mask[:width, :] = 0.0
    mask[-width:, :] = 0.0
    mask[:, :width] = 0.0
    mask[:, -width:] = 0.0
    return mask


def attack_bars(img: np.ndarray, width: int = 10) -> np.ndarray:
    """Remove altas frequências numa borda de largura `width`."""
    spec = to_spectrum(img)
    h, w = img.shape[:2]
    mask = bars_mask(h, w, width)[:, :, None]
    return from_spectrum(spec * mask)


# --------------------------------------------------------------------------- #
# Ataque 2: MEAN (targeted)                                                   #
# --------------------------------------------------------------------------- #
def fit_mean_fingerprint(
    fake_dir: str | Path,
    real_dir: str | Path,
    size: int = 256,
    max_images: int | None = 5000,
) -> np.ndarray:
    """
    Calcula fingerprint = média(espectro_fake) - média(espectro_real).
    Todas as imagens são redimensionadas para `size` x `size`.
    Retorna array complexo (size, size, 3).
    """

    def mean_spectrum(folder: str | Path) -> np.ndarray:
        paths = list_images(folder)
        if max_images is not None:
            paths = paths[:max_images]
        if not paths:
            raise ValueError(f"Nenhuma imagem encontrada em {folder}")
        acc = np.zeros((size, size, 3), dtype=np.complex128)
        for p in paths:
            acc += to_spectrum(load_image(p, size=size))
        return acc / len(paths)

    fake_mean = mean_spectrum(fake_dir)
    real_mean = mean_spectrum(real_dir)
    return (fake_mean - real_mean).astype(np.complex64)


def attack_mean(img: np.ndarray, fingerprint: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Subtrai factor*fingerprint do espectro da imagem.
    A imagem é redimensionada ao tamanho do fingerprint se necessário,
    e a saída é devolvida nesse tamanho.
    """
    fp_h, fp_w = fingerprint.shape[:2]
    if img.shape[:2] != (fp_h, fp_w):
        img = np.asarray(
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
                (fp_w, fp_h), Image.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0
    spec = to_spectrum(img)
    return from_spectrum(spec - factor * fingerprint)


# --------------------------------------------------------------------------- #
# Qualidade: PSNR                                                             #
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR em dB entre dois arrays float [0,1] de mesmo shape."""
    mse = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# --------------------------------------------------------------------------- #
# Processamento em lote                                                       #
# --------------------------------------------------------------------------- #
def process_folder(in_dir, out_dir, transform, size=None) -> list[float]:
    """
    Aplica `transform(img)->img` a todas as imagens de in_dir, salva em out_dir
    com o mesmo nome, e devolve a lista de PSNRs (original vs manipulado).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psnrs = []
    for p in list_images(in_dir):
        orig = load_image(p, size=size)
        manip = transform(orig)
        # alinhar shapes para PSNR (o mean pode ter redimensionado)
        ref = orig
        if ref.shape != manip.shape:
            ref = load_image(p, size=manip.shape[0])
        psnrs.append(psnr(ref, manip))
        save_image(manip, out_dir / p.name)
    return psnrs


def _report(psnrs: list[float], out_dir) -> None:
    if psnrs:
        arr = np.array([x for x in psnrs if np.isfinite(x)])
        print(f"  imagens processadas : {len(psnrs)}")
        print(f"  PSNR medio          : {arr.mean():.2f} dB")
        print(f"  PSNR min / max      : {arr.min():.2f} / {arr.max():.2f} dB")
    print(f"  salvo em            : {out_dir}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_bars = sub.add_parser("bars", help="Ataque bars (untargeted) numa pasta")
    p_bars.add_argument("fakes_dir")
    p_bars.add_argument("-o", "--output", required=True)
    p_bars.add_argument("--width", type=int, default=10)
    p_bars.add_argument("--size", type=int, default=None)

    p_fit = sub.add_parser("fit-mean", help="Calcula o fingerprint mean")
    p_fit.add_argument("fakes_dir")
    p_fit.add_argument("reais_dir")
    p_fit.add_argument("-o", "--output", required=True, help="arquivo .npy de saida")
    p_fit.add_argument("--size", type=int, default=256)
    p_fit.add_argument("--max-images", type=int, default=5000)

    p_mean = sub.add_parser("mean", help="Aplica o ataque mean")
    p_mean.add_argument("fakes_dir")
    p_mean.add_argument("fingerprint", help="arquivo .npy do fingerprint")
    p_mean.add_argument("-o", "--output", required=True)
    p_mean.add_argument("--factor", type=float, default=1.0)

    p_psnr = sub.add_parser("psnr", help="PSNR medio entre duas pastas pareadas por nome")
    p_psnr.add_argument("orig_dir")
    p_psnr.add_argument("manip_dir")

    args = ap.parse_args()

    if args.cmd == "bars":
        print(f"[bars] width={args.width}")
        ps = process_folder(args.fakes_dir, args.output,
                             lambda im: attack_bars(im, width=args.width),
                             size=args.size)
        _report(ps, args.output)

    elif args.cmd == "fit-mean":
        print(f"[fit-mean] size={args.size} max_images={args.max_images}")
        fp = fit_mean_fingerprint(args.fakes_dir, args.reais_dir,
                                  size=args.size, max_images=args.max_images)
        np.save(args.output, fp)
        mag = np.abs(fp)
        print(f"  fingerprint shape   : {fp.shape}")
        print(f"  magnitude media/max : {mag.mean():.4f} / {mag.max():.4f}")
        print(f"  salvo em            : {args.output}")

    elif args.cmd == "mean":
        fp = np.load(args.fingerprint)
        print(f"[mean] factor={args.factor} fingerprint={fp.shape}")
        ps = process_folder(args.fakes_dir, args.output,
                            lambda im: attack_mean(im, fp, factor=args.factor))
        _report(ps, args.output)

    elif args.cmd == "psnr":
        orig = {p.name: p for p in list_images(args.orig_dir)}
        vals = []
        for q in list_images(args.manip_dir):
            if q.name in orig:
                a = load_image(orig[q.name])
                b = load_image(q)
                if a.shape != b.shape:
                    a = load_image(orig[q.name], size=b.shape[0])
                vals.append(psnr(a, b))
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals):
            print(f"  pares comparados : {len(vals)}")
            print(f"  PSNR medio       : {vals.mean():.2f} dB")
        else:
            print("  nenhum par encontrado (nomes precisam bater)")


if __name__ == "__main__":
    main()
