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
    "coalbp_entropy",
    "clbp_entropy",
    "bsif_entropy",
    "wld_entropy",
    "ltp_entropy",
    "ida_specular",
    "surf_entropy",
    "rdwt_glcm_entropy",
    "context_lbp_chi2",
    "glcm_correlation",
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
    "coalbp_entropy": {
        "title": "Энтропия color CoALBP",
        "family": "global",
        "measures": "Совместная встречаемость соседних равномерных LBP в Y, Cr и Cb. У печати соседние коды повторяются.",
        "source": "Boulkenafet, Komulainen, Hadid 2016",
    },
    "clbp_entropy": {
        "title": "Энтропия совместного CLBP",
        "family": "global",
        "measures": "Совместное распределение знака и модуля локального перепада. У гладкого материала модуль схлопывается.",
        "source": "Guo, Zhang, Zhang 2010",
    },
    "bsif_entropy": {
        "title": "Энтропия кода BSIF",
        "family": "global",
        "measures": "Энтропия 8-битного кода из знаков ICA-фильтров 7×7. Фильтры обучены процедурой BSIF на 1/f патчах.",
        "source": "Kannala, Rahtu 2012; Arashloo, Kittler, Christmas 2015",
    },
    "wld_entropy": {
        "title": "Энтропия WLD",
        "family": "global",
        "measures": "Совместная гистограмма дифференциального возбуждения и ориентации градиента.",
        "source": "Chen et al. 2010; Gragnaniello et al. 2015",
    },
    "ltp_entropy": {
        "title": "Энтропия LTP",
        "family": "global",
        "measures": "Энтропия равномерных верхнего и нижнего тернарных кодов с порогом 5. Мелкий шум кожи сохраняется.",
        "source": "Tan, Triggs 2010",
    },
    "ida_specular": {
        "title": "Зеркальная доля IDA",
        "family": "global",
        "measures": "Средний положительный остаток после вычитания плавного освещения. Блик экрана и глянца выше, чем у кожи.",
        "source": "Wen, Han, Jain 2015",
    },
    "surf_entropy": {
        "title": "Энтропия ориентаций SURF",
        "family": "global",
        "measures": "Средняя энтропия ориентаций отклика Хаара по клеткам. У кожи нет одного господствующего направления.",
        "source": "Boulkenafet, Komulainen, Hadid 2017",
    },
    "rdwt_glcm_entropy": {
        "title": "Энтропия GLCM на RDWT",
        "family": "global",
        "measures": "Энтропия совместной встречаемости детальных подполос избыточного Хаара.",
        "source": "Agarwal, Singh, Vatsa 2016",
    },
    "context_lbp_chi2": {
        "title": "χ² LBP лица и фона",
        "family": "context",
        "measures": "Расхождение гистограмм LBP лица и участка кадра вне лица. На листе и экране это один материал.",
        "source": "Komulainen, Hadid, Pietikäinen 2013",
    },
    "glcm_correlation": {
        "title": "Корреляция GLCM",
        "family": "global",
        "measures": "Линейная связь яркостей соседних пикселей. Бумага и латекс связывают соседей сильнее, чем кожа.",
        "source": "Schwartz, Rocha, Pedrini 2011",
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


def _glcm(gray: np.ndarray) -> tuple[float, float, float, float]:
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
    levels = np.arange(16, dtype=np.float64)
    mean_row = float(np.sum(probability.sum(axis=1) * levels))
    mean_col = float(np.sum(probability.sum(axis=0) * levels))
    std_row = float(np.sqrt(np.sum(probability.sum(axis=1) * (levels - mean_row) ** 2)))
    std_col = float(np.sqrt(np.sum(probability.sum(axis=0) * (levels - mean_col) ** 2)))
    correlation = float(
        np.sum((levels[:, None] - mean_row) * (levels[None, :] - mean_col) * probability)
        / (std_row * std_col + 1e-12)
    )
    return contrast, homogeneity, asm, correlation


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


_OFFSETS = (
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _neighbors(image: np.ndarray, radius: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    height, width = image.shape
    if height <= 2 * radius + 1 or width <= 2 * radius + 1:
        return None
    center = image[radius : height - radius, radius : width - radius]
    neighbors = [
        image[
            radius + dy * radius : radius + dy * radius + center.shape[0],
            radius + dx * radius : radius + dx * radius + center.shape[1],
        ]
        for dy, dx in _OFFSETS
    ]
    return center, neighbors


def _uniform_labels(code: np.ndarray) -> np.ndarray:
    values = code.astype(np.uint16)
    rotated = ((values << 1) & 0xFF) | (values >> 7)
    transitions = np.bitwise_count(values ^ rotated)
    ones = np.bitwise_count(values)
    return np.where(transitions <= 2, ones, 9).astype(np.int32)


def _coalbp_entropy(channel: np.ndarray, radius: int = 1, distance: int = 2) -> float:
    packed = _neighbors(channel.astype(np.float32), radius)
    if packed is None:
        return float("nan")
    center, neighbors = packed
    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= (neighbor >= center).astype(np.uint8) << bit
    labels = _uniform_labels(code)
    entropies = []
    for dy, dx in ((0, distance), (distance, 0), (distance, distance), (distance, -distance)):
        y0, x0 = max(dy, 0), max(dx, 0)
        y1 = labels.shape[0] + min(dy, 0)
        x1 = labels.shape[1] + min(dx, 0)
        if y1 <= y0 or x1 <= x0:
            continue
        left = labels[y0:y1, x0:x1]
        right = labels[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        joint = np.zeros((10, 10), dtype=np.float64)
        np.add.at(joint, (left.ravel(), right.ravel()), 1.0)
        entropies.append(_entropy(joint.ravel()))
    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def _clbp_entropy(gray: np.ndarray, radius: int = 1) -> float:
    packed = _neighbors(gray.astype(np.float32), radius)
    if packed is None:
        return float("nan")
    center, neighbors = packed
    sign = np.zeros(center.shape, dtype=np.uint8)
    magnitudes = []
    for bit, neighbor in enumerate(neighbors):
        diff = neighbor - center
        magnitudes.append(np.abs(diff))
        sign |= (diff >= 0).astype(np.uint8) << bit
    stacked = np.stack(magnitudes, axis=0)
    threshold = float(stacked.mean())
    magnitude = np.zeros(center.shape, dtype=np.uint8)
    for bit in range(8):
        magnitude |= (stacked[bit] >= threshold).astype(np.uint8) << bit
    sign_labels = _uniform_labels(sign)
    magnitude_labels = _uniform_labels(magnitude)
    joint = np.zeros((10, 10), dtype=np.float64)
    np.add.at(joint, (sign_labels.ravel(), magnitude_labels.ravel()), 1.0)
    return _entropy(joint.ravel())


# ICA-фильтры 7×7, нулевое среднее и единичная энергия.
# Обучены один раз процедурой BSIF (PCA-whitening + FastICA, logcosh)
# на 1/f патчах с фиксированным seed. Таблица Оулу в репозиторий не входит.
_BSIF = np.array(
    [
        [
            [0.00437, 0.03110, 0.07809, 0.12171, 0.13348, 0.11583, 0.08591],
            [-0.03001, -0.03478, -0.01997, 0.00575, 0.02212, 0.03779, 0.04480],
            [-0.04255, -0.11369, -0.17315, -0.18799, -0.15090, -0.06874, 0.00239],
            [-0.00028, -0.11939, -0.25405, -0.31272, -0.25171, -0.10219, 0.02583],
            [0.10867, -0.01670, -0.20054, -0.28242, -0.19899, -0.01672, 0.11486],
            [0.22391, 0.12963, -0.04447, -0.11845, -0.04634, 0.11923, 0.22054],
            [0.26302, 0.20992, 0.09149, 0.03712, 0.09277, 0.21134, 0.25507],
        ],
        [
            [-0.21535, -0.07715, 0.14625, 0.25198, 0.15811, -0.05615, -0.17834],
            [-0.28486, -0.13078, 0.15125, 0.29943, 0.20836, -0.02820, -0.17205],
            [-0.27343, -0.16771, 0.06486, 0.21910, 0.19435, 0.03884, -0.08667],
            [-0.17093, -0.13071, -0.02674, 0.07056, 0.08990, 0.04962, -0.00650],
            [0.02421, -0.00049, -0.05013, -0.06949, -0.04375, 0.00615, 0.01991],
            [0.20966, 0.15026, -0.01693, -0.13391, -0.13666, -0.05111, 0.00601],
            [0.24947, 0.19936, 0.02726, -0.10844, -0.13401, -0.06879, -0.01560],
        ],
        [
            [-0.08128, -0.13189, -0.16650, -0.15997, -0.11142, -0.03988, 0.00699],
            [0.00043, -0.02070, -0.04804, -0.04269, 0.00665, 0.06092, 0.08297],
            [0.10900, 0.14170, 0.14154, 0.14450, 0.16084, 0.16857, 0.13991],
            [0.15485, 0.21427, 0.22734, 0.22159, 0.20549, 0.17109, 0.10996],
            [0.08621, 0.12695, 0.13574, 0.11770, 0.07745, 0.03233, -0.00858],
            [-0.05046, -0.05344, -0.06181, -0.09715, -0.13103, -0.15893, -0.15931],
            [-0.14181, -0.17924, -0.20555, -0.23674, -0.25787, -0.26844, -0.23225],
        ],
        [
            [-0.22858, -0.17504, -0.02307, 0.11129, 0.13673, 0.07877, -0.00005],
            [-0.14757, -0.05707, 0.08658, 0.15632, 0.08641, -0.04214, -0.11418],
            [0.04445, 0.14363, 0.19411, 0.12899, -0.04732, -0.21341, -0.24407],
            [0.15114, 0.22919, 0.20581, 0.04726, -0.14788, -0.28027, -0.25900],
            [0.09883, 0.12580, 0.07934, -0.02458, -0.11130, -0.13411, -0.09634],
            [-0.04673, -0.06471, -0.07135, -0.04592, 0.03040, 0.11492, 0.14464],
            [-0.13261, -0.15910, -0.13711, -0.02389, 0.12235, 0.25109, 0.25935],
        ],
        [
            [0.08450, 0.08720, 0.05943, 0.05707, 0.10067, 0.15379, 0.15337],
            [0.02843, 0.01967, 0.00270, 0.00513, 0.04683, 0.09025, 0.08386],
            [-0.08237, -0.07160, -0.04126, -0.02273, -0.02435, -0.04877, -0.07068],
            [-0.17155, -0.11493, 0.00176, 0.04788, -0.02144, -0.15765, -0.21803],
            [-0.20174, -0.08357, 0.12067, 0.19448, 0.05857, -0.18312, -0.29667],
            [-0.16140, -0.00349, 0.22867, 0.31081, 0.14660, -0.13905, -0.29274],
            [-0.09463, 0.04778, 0.23961, 0.29305, 0.15201, -0.08747, -0.22553],
        ],
        [
            [-0.27123, -0.25272, -0.16439, -0.05491, 0.03032, 0.07828, 0.08168],
            [-0.25126, -0.19805, -0.08978, 0.00725, 0.06226, 0.07195, 0.06094],
            [-0.16674, -0.07617, 0.03183, 0.09241, 0.08286, 0.03368, 0.01688],
            [-0.13139, -0.01641, 0.10987, 0.15232, 0.10917, 0.02671, 0.00184],
            [-0.18710, -0.07075, 0.08545, 0.16827, 0.15899, 0.09163, 0.05257],
            [-0.26836, -0.17846, -0.00123, 0.14571, 0.20824, 0.18288, 0.13678],
            [-0.28055, -0.21727, -0.06497, 0.09750, 0.18927, 0.20338, 0.17080],
        ],
        [
            [-0.13843, -0.18483, -0.15704, -0.00985, 0.19748, 0.38651, 0.42006],
            [-0.07250, -0.12070, -0.13888, -0.05979, 0.11017, 0.29359, 0.35094],
            [0.06778, 0.03408, -0.05344, -0.10008, -0.06031, 0.05225, 0.13767],
            [0.15345, 0.13429, 0.01982, -0.12120, -0.18062, -0.14351, -0.04235],
            [0.11646, 0.10301, 0.01173, -0.10490, -0.17729, -0.16898, -0.09031],
            [0.00599, -0.00411, -0.03261, -0.06466, -0.08253, -0.07351, -0.03216],
            [-0.05474, -0.07269, -0.06347, -0.03886, -0.00400, 0.02237, 0.03069],
        ],
        [
            [-0.24943, -0.30875, -0.30394, -0.23723, -0.13679, -0.03656, 0.00987],
            [-0.17426, -0.22318, -0.22491, -0.17403, -0.08251, 0.00444, 0.04229],
            [-0.00879, -0.02457, -0.04434, -0.03743, -0.00826, 0.03001, 0.04244],
            [0.14826, 0.16228, 0.12326, 0.06728, 0.03094, 0.01553, 0.00623],
            [0.23293, 0.25316, 0.19117, 0.10236, 0.01893, -0.03224, -0.04022],
            [0.23347, 0.24157, 0.17471, 0.07401, -0.01880, -0.07580, -0.08209],
            [0.18965, 0.17782, 0.11813, 0.03465, -0.03500, -0.07977, -0.08651],
        ],
    ],
    dtype=np.float32,
)


def _bsif_entropy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    code = np.zeros(image.shape, dtype=np.uint8)
    for bit in range(8):
        response = cv2.filter2D(image, cv2.CV_32F, _BSIF[bit])
        code |= (response > 0).astype(np.uint8) << bit
    margin = 3
    if code.shape[0] <= 2 * margin or code.shape[1] <= 2 * margin:
        return float("nan")
    hist = np.bincount(code[margin:-margin, margin:-margin].ravel(), minlength=256).astype(np.float64)
    return _entropy(hist)


def _wld_entropy(gray: np.ndarray, radius: int = 1) -> float:
    packed = _neighbors(gray.astype(np.float32), radius)
    if packed is None:
        return float("nan")
    center, neighbors = packed
    excitation = np.arctan(sum(neighbor - center for neighbor in neighbors) / (center + 1e-6))
    orientation = np.arctan2(neighbors[6] - neighbors[2], neighbors[0] - neighbors[4])
    excitation_bins = np.clip(((excitation + np.pi / 2.0) / np.pi * 6.0).astype(np.int32), 0, 5)
    orientation_bins = np.clip(((orientation + np.pi) / (2.0 * np.pi) * 8.0).astype(np.int32), 0, 7)
    hist = np.zeros((6, 8), dtype=np.float64)
    np.add.at(hist, (excitation_bins.ravel(), orientation_bins.ravel()), 1.0)
    return _entropy(hist.ravel())


def _ltp_entropy(gray: np.ndarray, radius: int = 1, threshold: float = 5.0) -> float:
    packed = _neighbors(gray.astype(np.float32), radius)
    if packed is None:
        return float("nan")
    center, neighbors = packed
    upper = np.zeros(center.shape, dtype=np.uint8)
    lower = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        upper |= (neighbor >= center + threshold).astype(np.uint8) << bit
        lower |= (neighbor <= center - threshold).astype(np.uint8) << bit
    entropies = []
    for code in (upper, lower):
        labels = _uniform_labels(code)
        entropies.append(_entropy(np.bincount(labels.ravel(), minlength=10).astype(np.float64)))
    return float(np.nanmean(entropies))


def _ida_specular(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    illumination = cv2.GaussianBlur(image, (0, 0), 4.0)
    specular = np.maximum(image - illumination, 0.0)
    return float(np.mean(specular) / (float(np.mean(image)) + 1e-6))


def _surf_entropy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    kernel = np.zeros((8, 8), dtype=np.float32)
    kernel[:, :4] = -1.0
    kernel[:, 4:] = 1.0
    kernel /= 32.0
    horizontal = cv2.filter2D(image, cv2.CV_32F, kernel)
    vertical = cv2.filter2D(image, cv2.CV_32F, kernel.T)
    magnitude = np.hypot(horizontal, vertical)[4:-4:4, 4:-4:4]
    angle = (np.degrees(np.arctan2(vertical, horizontal)) + 180.0)[4:-4:4, 4:-4:4] % 360.0
    cell = 4
    height = magnitude.shape[0] // cell * cell
    width = magnitude.shape[1] // cell * cell
    if height < cell or width < cell:
        return float("nan")
    magnitude = magnitude[:height, :width]
    bins = np.clip((angle[:height, :width] / 90.0).astype(np.int32), 0, 3)
    entropies = []
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            weights = magnitude[y : y + cell, x : x + cell]
            if float(weights.sum()) <= 1e-6:
                continue
            hist = np.bincount(bins[y : y + cell, x : x + cell].ravel(), weights=weights.ravel(), minlength=4)
            entropies.append(_entropy(hist.astype(np.float64)))
    if len(entropies) < 4:
        return float("nan")
    return float(np.mean(entropies))


def _band_glcm_entropy(band: np.ndarray) -> float:
    quantized = np.clip(((band.astype(np.float32) + 255.0) * (15.0 / 510.0)).astype(np.int32), 0, 15)
    if quantized.shape[0] < 2 or quantized.shape[1] < 2:
        return float("nan")
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
    return _entropy(symmetric.ravel())


def _rdwt_glcm_entropy(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    horizontal = image[:, :-1] - image[:, 1:]
    vertical = image[:-1, :] - image[1:, :]
    return float(np.nanmean([_band_glcm_entropy(horizontal), _band_glcm_entropy(vertical)]))


def _context_chi2(gray: np.ndarray, context_bgr: np.ndarray | None) -> float:
    if context_bgr is None:
        return float("nan")
    face_hist = _uniform_lbp_hist(gray, 1)
    context_hist = _uniform_lbp_hist(_to_gray(context_bgr), 1)
    if face_hist.sum() <= 0 or context_hist.sum() <= 0:
        return float("nan")
    return _chi2(face_hist, context_hist)


def _relative_spread(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return float("nan")
    return float(np.std(array) / (np.median(array) + 1e-6))


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def compute_features(
    face_bgr: np.ndarray,
    regions: list[np.ndarray],
    context_bgr: np.ndarray | None = None,
) -> dict[str, float]:
    """Признаки одного выровненного кропа, региональных кропов и фона вне лица."""
    gray = _to_gray(face_bgr)
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    contrast, homogeneity, asm, correlation = _glcm(gray)
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
        "coalbp_entropy": float(
            np.nanmean([_coalbp_entropy(gray), _coalbp_entropy(ycrcb[:, :, 1]), _coalbp_entropy(ycrcb[:, :, 2])])
        ),
        "clbp_entropy": _clbp_entropy(gray),
        "bsif_entropy": _bsif_entropy(gray),
        "wld_entropy": _wld_entropy(gray),
        "ltp_entropy": _ltp_entropy(gray),
        "ida_specular": _ida_specular(gray),
        "surf_entropy": _surf_entropy(gray),
        "rdwt_glcm_entropy": _rdwt_glcm_entropy(gray),
        "context_lbp_chi2": _context_chi2(gray, context_bgr),
        "glcm_correlation": correlation,
    }


def _self_check() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (192, 192, 3), dtype=np.uint8)
    smooth = np.full((192, 192, 3), 140, dtype=np.uint8)
    smooth[40:150, 40:150] = 180
    regions = [noisy[20 + 8 * i : 84 + 8 * i, 20:84] for i in range(5)]
    flat_context = np.full((64, 64, 3), 90, dtype=np.uint8)
    features = compute_features(noisy, regions, flat_context)
    missing = [name for name in FEATURE_NAMES if name not in features or not np.isfinite(features[name])]
    if missing:
        raise RuntimeError(f"Самопроверка: нет конечных значений {missing}")
    noisy_features = compute_features(noisy, regions, noisy[:64, :64])
    smooth_features = compute_features(smooth, regions, flat_context)
    if not noisy_features["lbp_entropy"] > smooth_features["lbp_entropy"]:
        raise RuntimeError("Самопроверка: у шума энтропия LBP должна быть выше, чем у гладкого поля")
    for name in ("coalbp_entropy", "clbp_entropy", "bsif_entropy", "wld_entropy", "ltp_entropy"):
        if not noisy_features[name] > smooth_features[name]:
            raise RuntimeError(f"Самопроверка: у шума {name} должна быть выше, чем у гладкого поля")
    spike = smooth.copy()
    spike[80:100, 80:100] = 255
    if not compute_features(spike, regions)["ida_specular"] > smooth_features["ida_specular"]:
        raise RuntimeError("Самопроверка: блик должен поднимать зеркальную долю")
    if not features["context_lbp_chi2"] > noisy_features["context_lbp_chi2"]:
        raise RuntimeError("Самопроверка: LBP лица и гладкого фона должны расходиться сильнее, чем у одного шума")
