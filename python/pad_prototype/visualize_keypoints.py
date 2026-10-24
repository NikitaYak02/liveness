"""Один кадр на входе, тот же кадр с точками проверки атаки на выходе.

Голубые точки задают плоскость гомографии: углы глаз и рта.
Красная точка — кончик носа, проба.
Жёлтые — контроль: щёки, лоб, подбородок, переносица.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pad_prototype.run import (
    BRIDGE,
    CHEEK_A,
    CHEEK_B,
    CHIN,
    EYE_INNER_A,
    EYE_INNER_B,
    EYE_OUTER_A,
    EYE_OUTER_B,
    FOREHEAD,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    FaceTracker,
)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

# BGR для OpenCV-кругов. Подписи рисуются через PIL в RGB.
PLANE = ((255, 200, 0), (0, 200, 255), "плоскость")
PROBE = ((0, 0, 255), (255, 0, 0), "нос")
CONTROL = ((0, 220, 255), (255, 220, 0), "контроль")

# Направление подписи от точки: x вправо кадра, y вниз.
POINTS = (
    (EYE_OUTER_A, PLANE, "внешний угол правого глаза", (-1.0, 0.45)),
    (EYE_INNER_A, PLANE, "внутренний угол правого глаза", (-1.0, -0.85)),
    (EYE_OUTER_B, PLANE, "внешний угол левого глаза", (1.0, 0.45)),
    (EYE_INNER_B, PLANE, "внутренний угол левого глаза", (1.0, -0.85)),
    (MOUTH_L, PLANE, "правый угол рта", (-0.85, 0.7)),
    (MOUTH_R, PLANE, "левый угол рта", (0.85, 0.7)),
    (NOSE, PROBE, "кончик носа", (0.15, 1.0)),
    (CHEEK_A, CONTROL, "правая щека", (-1.0, 0.15)),
    (CHEEK_B, CONTROL, "левая щека", (1.0, 0.15)),
    (FOREHEAD, CONTROL, "лоб", (0.0, -1.0)),
    (CHIN, CONTROL, "подбородок", (0.0, 1.0)),
    (BRIDGE, CONTROL, "переносица", (-0.95, -0.25)),
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _draw(bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    used = np.stack([points[index] for index, _, _, _ in POINTS])
    center = used.mean(axis=0)
    span = float(np.linalg.norm(used.max(axis=0) - used.min(axis=0)))
    radius = max(4, int(round(span * 0.012)))
    font = _font(max(12, int(round(span * 0.02))))
    legend_font = _font(max(14, int(round(span * 0.026))))

    canvas = bgr.copy()
    for index, (bgr_color, _, _), _name, _direction in POINTS:
        x, y = points[index]
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, bgr_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    placed: list[tuple[float, float, float, float]] = []
    for index, (_, rgb, group), name, raw_direction in POINTS:
        x, y = float(points[index][0]), float(points[index][1])
        direction = np.array(raw_direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        text = name
        probe = draw.textbbox((0, 0), text, font=font)
        width, height = float(probe[2] - probe[0]), float(probe[3] - probe[1])
        left = top = 0.0
        rect = (0.0, 0.0, 0.0, 0.0)
        for gap in (radius + 14, radius + 36, radius + 64, radius + 96, radius + 130):
            half = abs(direction[0]) * width / 2 + abs(direction[1]) * height / 2
            anchor = np.array([x, y]) + direction * (gap + half + 6)
            left = float(anchor[0] - width / 2)
            top = float(anchor[1] - height / 2)
            ink = draw.textbbox((left, top), text, font=font)
            rect = (ink[0] - 14, ink[1] - 8, ink[2] + 14, ink[3] + 8)
            inside = rect[0] >= 0 and rect[1] >= 0 and rect[2] <= image.width and rect[3] <= image.height
            if inside and not any(_overlaps(rect, other) for other in placed):
                break
        placed.append(rect)
        draw.rectangle((rect[0] + 8, rect[1] + 4, rect[2] - 8, rect[3] - 4), fill=(0, 0, 0))
        draw.text((left, top), text, font=font, fill=rgb)

    legend = (
        ("плоскость  глаза и углы рта", PLANE[1]),
        ("нос  кончик, проба", PROBE[1]),
        ("контроль  щёки, лоб, подбородок, переносица", CONTROL[1]),
    )
    x0, y0 = 16, 16
    line_h = max(22, int(round(span * 0.04)))
    pad = 8
    widths = []
    for line, _color in legend:
        box = draw.textbbox((0, 0), line, font=legend_font)
        widths.append(box[2] - box[0])
    box_w = max(widths) + pad * 2 + 18
    box_h = line_h * len(legend) + pad
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    for i, (line, color) in enumerate(legend):
        cy = y0 + pad + i * line_h + line_h // 2
        draw.ellipse((x0 + 8, cy - 6, x0 + 20, cy + 6), fill=color)
        draw.text((x0 + 26, y0 + pad + i * line_h), line, font=legend_font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Нарисовать точки проверки атаки на одном кадре")
    parser.add_argument("--image", type=Path, required=True, help="Входное изображение")
    parser.add_argument("--out", type=Path, required=True, help="Куда сохранить кадр с точками")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise SystemExit(f"Не удалось прочитать изображение: {args.image}")
    points = FaceTracker().detect(bgr)
    if points is None:
        raise SystemExit(f"Лицо не найдено: {args.image}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.out.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(args.out), _draw(bgr, points), [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok = cv2.imwrite(str(args.out), _draw(bgr, points))
    if not ok:
        raise SystemExit(f"Не удалось записать: {args.out}")
    print(f"Точки: {args.out}")


if __name__ == "__main__":
    main()
