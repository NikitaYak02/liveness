"""Текстурные признаки маски и региона вставки на кропе лица.

Глобальные признаки описывают материал: у кожи больше мелкой текстуры, чем у
бумаги, латекса, силикона и пересъёмки. Признаки по регионам сравнивают лоб,
щёки, нос и подбородок: вставка и вырез отличаются от соседней кожи шумом,
резкостью и рисунком LBP.

Каждый признак — одно число. Знак (что считать атакой) выбирается на обучении
при подборе порога, а не здесь.
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "lbp_entropy",
    "ms_lbp_entropy",
    "cb_lbp_entropy",
    "cr_lbp_entropy",
    "sat_lbp_entropy",
    "lpq_entropy",
    "dog_energy",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_asm",
    "fft_hf_ratio",
    "haar_hf_ratio",
    "gabor_energy",
    "hog_entropy",
    "lap_var",
    "noise_std",
    "region_lbp_chi2",
    "noise_inconsistency",
    "blur_inconsistency",
    "entropy_range",
)

# Что признак измеряет и откуда он в детекции масок / склеек.
FEATURE_INFO: dict[str, dict[str, str]] = {
    "lbp_entropy": {
        "title": "Энтропия равномерного LBP",
        "family": "global",
        "measures": "Богатство микротекстуры яркости. Бумага и экран беднее кожи.",
        "source": "Chingovska, Anjos, Marcel 2012",
    },
    "ms_lbp_entropy": {
        "title": "Энтропия мультимасштабного LBP",
        "family": "global",
        "measures": "То же на радиусах 1, 2 и 3 пикселя: зерно печати видно не на одном масштабе.",
        "source": "Määttä, Hadid, Pietikäinen 2011",
    },
    "cb_lbp_entropy": {
        "title": "LBP энтропия канала Cb",
        "family": "global",
        "measures": "Текстура синей цветности. У печати и маски цветность глаже, чем у кожи.",
        "source": "Boulkenafet, Komulainen, Hadid 2015",
    },
    "cr_lbp_entropy": {
        "title": "LBP энтропия канала Cr",
        "family": "global",
        "measures": "Текстура красной цветности, самый устойчивый цветовой канал в color-LBP.",
        "source": "Boulkenafet, Komulainen, Hadid 2015",
    },
    "sat_lbp_entropy": {
        "title": "LBP энтропия насыщенности",
        "family": "global",
        "measures": "Рисунок насыщенности HSV. Маска и повторная съёмка выравнивают насыщенность.",
        "source": "Boulkenafet, Komulainen, Hadid 2016",
    },
    "lpq_entropy": {
        "title": "Энтропия LPQ",
        "family": "global",
        "measures": "Локальная фаза низких частот. Устойчива к размытию, которым печать гасит поры.",
        "source": "Ojansivu, Heikkilä 2008; применяется в PAD",
    },
    "dog_energy": {
        "title": "Энергия разности гауссиан",
        "family": "global",
        "measures": "Средние частоты после DoG. Печать и экран теряют полосу, где лежат поры.",
        "source": "Tan et al. 2010",
    },
    "glcm_contrast": {
        "title": "Контраст GLCM",
        "family": "global",
        "measures": "Контраст соседних пикселей по матрице совместной встречаемости.",
        "source": "Haralick 1973; PAD-обзоры текстуры",
    },
    "glcm_homogeneity": {
        "title": "Однородность GLCM",
        "family": "global",
        "measures": "Насколько соседние пиксели похожи. Бумага и латекс однороднее кожи.",
        "source": "Haralick 1973",
    },
    "glcm_asm": {
        "title": "Энергия GLCM (ASM)",
        "family": "global",
        "measures": "Регулярность совместной встречаемости. Повторяющееся зерно печати повышает ASM.",
        "source": "Haralick 1973",
    },
    "fft_hf_ratio": {
        "title": "Доля высоких частот FFT",
        "family": "global",
        "measures": "Доля энергии спектра вне низких частот. У живого лица её больше.",
        "source": "Li, He, Sun, Tan 2004",
    },
    "haar_hf_ratio": {
        "title": "Энергия деталей Хаара",
        "family": "global",
        "measures": "Энергия LH/HL/HH одного уровня Хаара относительно LL.",
        "source": "вейвлет-текстура в PAD",
    },
    "gabor_energy": {
        "title": "Энергия фильтров Габора",
        "family": "global",
        "measures": "Отклик коротковолновых Габоров по четырём ориентациям, нормированный на контраст.",
        "source": "Gabor-текстура, face PAD",
    },
    "hog_entropy": {
        "title": "Энтропия HOG",
        "family": "global",
        "measures": "Разнообразие ориентаций градиента. Кожа даёт более равномерную розу направлений.",
        "source": "Dalal, Triggs; HOG в PAD",
    },
    "lap_var": {
        "title": "Дисперсия Лапласиана",
        "family": "global",
        "measures": "Резкость кропа. Печать, экран и многие маски мягче кожи на том же масштабе.",
        "source": "Galbally, Marcel, Fierrez 2014 (IQM)",
    },
    "noise_std": {
        "title": "СКО высокочастотного остатка",
        "family": "global",
        "measures": "Амплитуда остатка после слабого размытия: поры и шум сенсора против гладкого материала.",
        "source": "остаток в IQM и в поиске склеек",
    },
    "region_lbp_chi2": {
        "title": "χ² LBP между регионами лица",
        "family": "insertion",
        "measures": "Расхождение гистограмм LBP лба, щёк, носа и подбородка. Вставка имеет другой рисунок.",
        "source": "несогласованность регионов при splice / partial attack",
    },
    "noise_inconsistency": {
        "title": "Разброс шума по регионам",
        "family": "insertion",
        "measures": "Разброс СКО шумового остатка по тем же регионам. У склейки шум не один на всё лицо.",
        "source": "Lyu, Pan, Zhang 2005; Mahdian, Saic",
    },
    "blur_inconsistency": {
        "title": "Разброс резкости по регионам",
        "family": "insertion",
        "measures": "Разброс дисперсии Лапласиана. Вырезанные глаза и наклеенная бумага резкие по-разному.",
        "source": "несогласованность размытия на границе вставки",
    },
    "entropy_range": {
        "title": "Размах локальной энтропии",
        "family": "insertion",
        "measures": "Разница энтропий яркости регионов. Гладкая вставка рядом с кожей растягивает этот размах.",
        "source": "локальная энтропия как карта склейки",
    },
}


def _entropy(hist: np.ndarray) -> float:
    total = float(hist.sum())
    if total <= 0:
        return float("nan")
    probability = hist[hist > 0] / total
    return float(-(probability * np.log2(probability)).sum())


def _uniform_lbp_hist(gray: np.ndarray, radius: int) -> np.ndarray:
    image = gray.astype(np.float32)
    height, width = image.shape
    if height <= 2 * radius + 1 or width <= 2 * radius + 1:
        return np.zeros(10, dtype=np.float64)
    center = image[radius : height - radius, radius : width - radius]
    code = np.zeros(center.shape, dtype=np.uint8)
    offsets = (
        (0, radius),
        (-radius, radius),
        (-radius, 0),
        (-radius, -radius),
        (0, -radius),
        (radius, -radius),
        (radius, 0),
        (radius, radius),
    )
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = image[radius + dy : radius + dy + center.shape[0], radius + dx : radius + dx + center.shape[1]]
        code |= (neighbor >= center).astype(np.uint8) << bit
    values = code.astype(np.uint16)
    rotated = ((values << 1) & 0xFF) | (values >> 7)
    transitions = np.bitwise_count(values ^ rotated)
    ones = np.bitwise_count(values)
    labels = np.where(transitions <= 2, ones, 9).astype(np.int32)
    return np.bincount(labels.ravel(), minlength=10).astype(np.float64)


def _lbp_entropy(gray: np.ndarray, radius: int = 1) -> float:
    return _entropy(_uniform_lbp_hist(gray, radius))


def _lpq_filters(win: int = 7) -> list[np.ndarray]:
    alpha = 1.0 / win
    half = (win - 1) / 2.0
    coords = np.arange(-half, half + 1.0, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    filters = []
    for fu, fv in ((alpha, 0.0), (0.0, alpha), (alpha, alpha), (alpha, -alpha)):
        phase = (2.0 * np.pi * (fu * xx + fv * yy)).astype(np.float32)
        filters.append(np.cos(phase))
        filters.append(np.sin(phase))
    return filters


_LPQ = _lpq_filters()


def _lpq_entropy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    code = np.zeros(image.shape, dtype=np.uint8)
    for bit, kernel in enumerate(_LPQ):
        response = cv2.filter2D(image, cv2.CV_32F, kernel)
        code |= (response >= 0).astype(np.uint8) << bit
    margin = 3
    if code.shape[0] <= 2 * margin or code.shape[1] <= 2 * margin:
        return float("nan")
    hist = np.bincount(code[margin:-margin, margin:-margin].ravel(), minlength=256).astype(np.float64)
    return _entropy(hist)


def _dog_energy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    narrow = cv2.GaussianBlur(image, (0, 0), 0.8)
    wide = cv2.GaussianBlur(image, (0, 0), 1.6)
    dog = narrow - wide
    return float(np.mean(dog * dog) / (np.mean(wide * wide) + 1e-6))


def _glcm(gray: np.ndarray) -> tuple[float, float, float]:
    quantized = np.clip((gray.astype(np.float32) * (15.0 / 255.0)).astype(np.int32), 0, 15)
    acc = np.zeros((16, 16), dtype=np.float64)
    pairs = (
        (quantized[:, :-1].ravel(), quantized[:, 1:].ravel()),
        (quantized[:-1, :].ravel(), quantized[1:, :].ravel()),
    )
    for left, right in pairs:
        np.add.at(acc, (left, right), 1.0)
    symmetric = acc + acc.T
    diagonal = np.arange(16)
    symmetric[diagonal, diagonal] = acc[diagonal, diagonal]
    total = float(symmetric.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    probability = symmetric / total
    diff = (np.arange(16, dtype=np.float64)[:, None] - np.arange(16, dtype=np.float64)[None, :])
    contrast = float(np.sum(probability * diff * diff))
    homogeneity = float(np.sum(probability / (1.0 + np.abs(diff))))
    asm = float(np.sum(probability * probability))
    return contrast, homogeneity, asm


def _fft_hf_ratio(gray: np.ndarray) -> float:
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    height, width = magnitude.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(yy - height / 2.0, xx - width / 2.0)
    limit = 0.5 * min(height, width)
    total = float(magnitude[radius <= limit].sum())
    high = float(magnitude[(radius >= 0.25 * limit) & (radius <= limit)].sum())
    return high / (total + 1e-8)


def _haar_hf_ratio(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    image = image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]
    upper_left = image[0::2, 0::2]
    upper_right = image[0::2, 1::2]
    lower_left = image[1::2, 0::2]
    lower_right = image[1::2, 1::2]
    low = upper_left + upper_right + lower_left + lower_right
    horizontal = upper_left + upper_right - lower_left - lower_right
    vertical = upper_left - upper_right + lower_left - lower_right
    diagonal = upper_left - upper_right - lower_left + lower_right
    high = float(np.mean(horizontal * horizontal + vertical * vertical + diagonal * diagonal))
    return high / (float(np.mean(low * low)) + 1e-8)


def _gabor_kernels() -> list[np.ndarray]:
    kernels = []
    for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
        kernels.append(cv2.getGaborKernel((21, 21), 3.0, theta, 4.0, 0.5, 0, ktype=cv2.CV_32F))
    return kernels


_GABOR = _gabor_kernels()


def _gabor_energy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    energy = 0.0
    for kernel in _GABOR:
        response = cv2.filter2D(image, cv2.CV_32F, kernel)
        energy += float(np.mean(response * response))
    return (energy / len(_GABOR)) / (float(np.var(image)) + 1e-6)


def _hog_entropy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    bins = np.clip((angle / 20.0).astype(np.int32), 0, 8)
    hist = np.bincount(bins.ravel(), weights=magnitude.ravel(), minlength=9).astype(np.float64)
    return _entropy(hist)


def _laplacian_var(gray: np.ndarray) -> float:
    response = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(response))


def _noise_std(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    residual = image - cv2.GaussianBlur(image, (0, 0), 1.0)
    return float(np.std(residual))


def _gray_entropy(gray: np.ndarray) -> float:
    values = np.clip(gray.astype(np.int32), 0, 255) // 8
    return _entropy(np.bincount(values.ravel(), minlength=32).astype(np.float64))


def _chi2(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    left = hist_a / (float(hist_a.sum()) + 1e-8)
    right = hist_b / (float(hist_b.sum()) + 1e-8)
    return float(0.5 * np.sum((left - right) ** 2 / (left + right + 1e-8)))


def _relative_spread(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return float("nan")
    return float(np.std(array) / (np.median(array) + 1e-6))


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def compute_features(face_bgr: np.ndarray, regions: list[np.ndarray]) -> dict[str, float]:
    """Признаки одного выровненного кропа и списка региональных кропов."""
    gray = _to_gray(face_bgr)
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    contrast, homogeneity, asm = _glcm(gray)
    region_gray = [_to_gray(patch) for patch in regions]

    lbp_hists = [_uniform_lbp_hist(patch, 1) for patch in region_gray]
    distances = [
        _chi2(lbp_hists[i], lbp_hists[j])
        for i in range(len(lbp_hists))
        for j in range(i + 1, len(lbp_hists))
        if lbp_hists[i].sum() > 0 and lbp_hists[j].sum() > 0
    ]
    noise_values = []
    blur_values = []
    entropies = []
    for patch in region_gray:
        patch_f = patch.astype(np.float32)
        residual = patch_f - cv2.GaussianBlur(patch_f, (0, 0), 1.0)
        noise_values.append(float(np.std(residual) / (float(np.mean(patch_f)) + 1e-6)))
        lap = cv2.Laplacian(patch_f, cv2.CV_32F, ksize=3)
        blur_values.append(float(np.var(lap) / (float(np.var(patch_f)) + 1e-6)))
        entropies.append(_gray_entropy(patch))
    finite_entropy = [value for value in entropies if np.isfinite(value)]
    entropy_range = (
        float(max(finite_entropy) - min(finite_entropy)) if len(finite_entropy) >= 3 else float("nan")
    )

    return {
        "lbp_entropy": _lbp_entropy(gray, 1),
        "ms_lbp_entropy": float(
            np.nanmean([_lbp_entropy(gray, radius) for radius in (1, 2, 3)])
        ),
        "cb_lbp_entropy": _lbp_entropy(ycrcb[:, :, 2], 1),
        "cr_lbp_entropy": _lbp_entropy(ycrcb[:, :, 1], 1),
        "sat_lbp_entropy": _lbp_entropy(hsv[:, :, 1], 1),
        "lpq_entropy": _lpq_entropy(gray),
        "dog_energy": _dog_energy(gray),
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_asm": asm,
        "fft_hf_ratio": _fft_hf_ratio(gray),
        "haar_hf_ratio": _haar_hf_ratio(gray),
        "gabor_energy": _gabor_energy(gray),
        "hog_entropy": _hog_entropy(gray),
        "lap_var": _laplacian_var(gray),
        "noise_std": _noise_std(gray),
        "region_lbp_chi2": float(np.mean(distances)) if distances else float("nan"),
        "noise_inconsistency": _relative_spread(noise_values),
        "blur_inconsistency": _relative_spread(blur_values),
        "entropy_range": entropy_range,
    }


def _self_check() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (192, 192, 3), dtype=np.uint8)
    smooth = np.full((192, 192, 3), 140, dtype=np.uint8)
    smooth[40:150, 40:150] = 180
    regions = [noisy[20 + 8 * i : 84 + 8 * i, 20:84] for i in range(5)]
    features = compute_features(noisy, regions)
    missing = [name for name in FEATURE_NAMES if name not in features or not np.isfinite(features[name])]
    if missing:
        raise RuntimeError(f"Самопроверка: нет конечных значений {missing}")
    noisy_lbp = compute_features(noisy, regions)["lbp_entropy"]
    smooth_lbp = compute_features(smooth, regions)["lbp_entropy"]
    if not noisy_lbp > smooth_lbp:
        raise RuntimeError("Самопроверка: у шума энтропия LBP должна быть выше, чем у гладкого поля")
