"""Общие константы и утилиты для пайплайна обучения face keypoints."""

import os
import re
from pathlib import Path

import numpy as np

KP_NAMES = [
    # Indices 0..10 stay stable: skeleton_edges and older checkpoints depend on them.
    'kp_eye_left_outer', 'kp_eye_right_inner', 'kp_eye_left_inner', 'kp_eye_right_outer',
    'kp_upper_lip_top', 'kp_under_lip_bottom', 'kp_mouth_right', 'kp_mouth_left', 'kp_nose_tip',
    'kp_pupil_left', 'kp_pupil_right',
    'kp_eyebrow_left_outer', 'kp_eyebrow_right_outer',
    'kp_eyebrow_left_inner', 'kp_eyebrow_right_inner',
    'kp_eyebrow_left_center', 'kp_eyebrow_right_center',
    'kp_eyelid_left_upper_outer', 'kp_eyelid_right_upper_outer',
    'kp_eyelid_left_upper_inner', 'kp_eyelid_right_upper_inner',
    'kp_eyelid_left_lower_outer', 'kp_eyelid_right_lower_outer',
    'kp_eyelid_left_lower_inner', 'kp_eyelid_right_lower_inner',
    # 25..29: контрольные точки hpe_parallax. Индексы 0..24 не сдвигать.
    'kp_cheek_right', 'kp_cheek_left', 'kp_forehead', 'kp_chin', 'kp_nose_bridge',
]
# Все точки списка используются в обучении (count_point совпадает с длиной).
TRAIN_KP_NAMES = KP_NAMES
TRAIN_KP_SHORT_NAMES = [n[3:] if n.startswith('kp_') else n for n in TRAIN_KP_NAMES]
# 0..10 — скелет, от которого зависят skeleton_edges и старые чекпоинты.
BASE_KP_COUNT = 11
LIVENESS_EXTRA_SHORT_NAMES = [
    'cheek_right', 'cheek_left', 'forehead', 'chin', 'nose_bridge',
]
LIVENESS_KP_SHORT_NAMES = (
    TRAIN_KP_SHORT_NAMES[:BASE_KP_COUNT] + LIVENESS_EXTRA_SHORT_NAMES
)
NEW_KP_TO_TRAIN = {short: f'kp_{short}' for short in TRAIN_KP_SHORT_NAMES}

# Имена точек в новом формате разметки (без префикса kp_).
NEW_FORMAT_KP_NAMES = list(TRAIN_KP_SHORT_NAMES)
_OTHER_INDEX_RE = re.compile(r'_(\d+)$')

KP_MAP_INSIGHTFACE = {
    'kp_nose_tip': 86, 'kp_eye_left_outer': 93, 'kp_eye_left_inner': 89,
    'kp_eye_right_inner': 39, 'kp_eye_right_outer': 35, 'kp_under_lip_bottom': 53,
    'kp_upper_lip_top': 71, 'kp_mouth_right': 52, 'kp_mouth_left': 61,
    'kp_pupil_left': 88, 'kp_pupil_right': 38,
}

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
MISSING_PT = -1000
DEFAULT_FACE_ROTATION = 'nrotations90_0'
NROTATIONS90_TO_DEG = {
    'nrotations90_0': 0,
    'nrotations90_1': 90,
    'nrotations90_2': 180,
    'nrotations90_3': 270,
}


def parse_face_rotation_deg(value):
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        deg = int(value) % 360
        if deg in (0, 90, 180, 270):
            return deg
        raise ValueError(f'unsupported face_rotation degrees: {value}')
    if isinstance(value, str):
        if value in NROTATIONS90_TO_DEG:
            return NROTATIONS90_TO_DEG[value]
        raise ValueError(f'unsupported face_rotation: {value}')
    raise ValueError(f'unsupported face_rotation type: {type(value)}')


def face_rotation_deg(props):
    """Угол поворота ректа (0/90/180/270) из props.face_rotation или face_rect[1]."""
    if props.get('face_rotation'):
        return parse_face_rotation_deg(props['face_rotation'])
    rect = props.get('face_rect')
    if isinstance(rect, list) and len(rect) >= 2 and isinstance(rect[1], str):
        return parse_face_rotation_deg(rect[1])
    return 0


def rotate_point_to_upright(x, y, rotation_deg, img_w, img_h):
    """Image coords → upright (0°), inverse of facerecog2::RotatePoint."""
    if rotation_deg == 0:
        return float(x), float(y)
    if rotation_deg == 90:
        return float(y), float(img_h - x)
    if rotation_deg == 180:
        return float(img_w - x), float(img_h - y)
    if rotation_deg == 270:
        return float(img_w - y), float(x)
    raise ValueError(f'unsupported rotation: {rotation_deg}')


def unrotate_rect(rect, rotation_deg, img_w, img_h):
    x, y, w, h = rect
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = [rotate_point_to_upright(px, py, rotation_deg, img_w, img_h) for px, py in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    return x0, y0, max(xs) - x0, max(ys) - y0


def face_rotation_label(props):
    if props.get('face_rotation'):
        return props['face_rotation']
    rect = props.get('face_rect')
    if isinstance(rect, list) and len(rect) >= 2 and isinstance(rect[1], str):
        return rect[1]
    return None


def upright_props_and_rect(props, img_w, img_h):
    """Копия props с kp_* и face_rect, выпрямленная по face_rotation."""
    rot = face_rotation_deg(props)
    if rot == 0:
        rect = face_rect(props)
        return props, rect

    out = dict(props)
    for key, val in props.items():
        if not key.startswith('kp_') or key.startswith('other_kp_'):
            continue
        pt = read_point(props, key)
        if not pt:
            continue
        ux, uy = rotate_point_to_upright(pt[0], pt[1], rot, img_w, img_h)
        out[key] = [[ux, uy]]

    rect = face_rect(props)
    if rect:
        out['face_rect'] = [list(map(float, unrotate_rect(rect, rot, img_w, img_h)))]
    return out, face_rect(out)


def is_yaw_markup_path(markup_path):
    return 'yaw' in Path(markup_path).parts


def ensure_face_rotation(props, markup_path=None):
    """Проставить face_rotation по умолчанию (0°), если поле отсутствует в yaw-разметке."""
    if props.get('face_rotation'):
        return False
    if markup_path is not None and not is_yaw_markup_path(markup_path):
        return False
    props['face_rotation'] = DEFAULT_FACE_ROTATION
    return True


def is_image_file(name):
    return Path(name).suffix.lower() in IMG_EXT


def is_missing(x, y):
    return x <= -500 or y <= -500


def is_invalid_double_ext_name(name):
    """Имя вида photo.jpg.png — артефакт двойного расширения, невалидная разметка."""
    stem = Path(name).stem.lower()
    return any(stem.endswith(ext) for ext in IMG_EXT)


def swap_path_dir(path_str, from_dir, to_dir):
    """Меняет сегмент пути from_dir→to_dir (только целое имя каталога, не подстроку в имени файла)."""
    parts = [to_dir if p == from_dir else p for p in Path(path_str).parts]
    return str(Path(*parts))


def _markup_path_candidates(img_path):
    """Кандидаты markup: сначала старый replace, затем замена только каталога images."""
    ext = Path(img_path).suffix
    base = str(Path(img_path).with_suffix(''))
    old_style = [
        img_path.replace('images', 'markup') + '.json',
        base.replace('images', 'markup') + '.json',
    ]
    new_style = [
        swap_path_dir(img_path, 'images', 'markup') + '.json',
        swap_path_dir(base, 'images', 'markup') + '.json',
    ]
    seen = set()
    for path in old_style + new_style:
        if path in seen:
            continue
        seen.add(path)
        yield path


def resolve_markup_paths(img_path):
    return [p for p in _markup_path_candidates(img_path) if os.path.exists(p)]


def resolve_markup_path(img_path):
    found = resolve_markup_paths(img_path)
    if found:
        return found[0]
    return next(_markup_path_candidates(img_path))


def _image_base_candidates(markup_path):
    old_base = markup_path.replace('.json', '').replace('markup', 'images')
    new_base = swap_path_dir(str(Path(markup_path).with_suffix('')), 'markup', 'images')
    seen = set()
    for base in (old_base, new_base):
        if base in seen:
            continue
        seen.add(base)
        yield base


def resolve_image_path(markup_path):
    for base in _image_base_candidates(markup_path):
        for ext in IMG_EXT:
            path = base + ext
            if os.path.exists(path):
                return path
        if os.path.exists(base):
            return base
    return None


def read_point(props, name):
    val = props.get(name)
    if not val:
        return None
    x, y = float(val[0][0]), float(val[0][1])
    return None if is_missing(x, y) else (x, y)


def set_missing(props, name):
    props[name] = [[MISSING_PT, MISSING_PT]]


def face_rect(props):
    rect = props.get('face_rect')
    if not rect:
        return None
    x, y, w, h = map(float, rect[0])
    return None if w <= 0 or h <= 0 else (x, y, w, h)


def rect_info(props, img_w, img_h):
    """Размер face_rect и производные метрики для принятия решения."""
    rect = face_rect(props)
    if not rect:
        return None
    x, y, w, h = rect
    scale = min(128 / max(img_w, 1), 128 / max(img_h, 1))
    return {
        'x': round(x, 1), 'y': round(y, 1),
        'width': round(w, 1), 'height': round(h, 1),
        'area': round(w * h, 1),
        'area_frac': round(w * h / max(1, img_w * img_h), 4),
        'min_side': round(min(w, h), 1),
        'min_side_at_128': round(min(w, h) * scale, 1),
        'img_width': img_w, 'img_height': img_h,
    }


def format_rect_info(info):
    if not info:
        return 'face_rect: нет'
    return (
        f"face_rect: {info['width']:.0f}×{info['height']:.0f} px  "
        f"(area={info['area']:.0f}, {info['area_frac']*100:.1f}% кадра, "
        f"min_side={info['min_side']:.0f}, @128={info['min_side_at_128']:.0f})"
    )


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def in_rect(pt, rect, margin=0.0):
    x, y, w, h = rect
    mx, my = w * margin, h * margin
    return x - mx <= pt[0] <= x + w + mx and y - my <= pt[1] <= y + h + my


def landmarks_to_points(landmarks, w, h, kp_map=None):
    kp_map = kp_map or KP_MAP_INSIGHTFACE
    out = {}
    for name, idx in kp_map.items():
        if 0 <= idx < len(landmarks):
            out[name] = (
                float(np.clip(landmarks[idx, 0], 0, w - 1)),
                float(np.clip(landmarks[idx, 1], 0, h - 1)),
            )
    return out


def list_image_markup_pairs(root):
    """Список (img_path, markup_path) под root (images/ или сам root)."""
    root = Path(root)
    search = root / 'images' if (root / 'images').is_dir() else root
    pairs = []
    for dirpath, _, files in os.walk(search):
        for name in files:
            if not is_image_file(name):
                continue
            img = os.path.join(dirpath, name)
            mrk = resolve_markup_path(img)
            if os.path.exists(mrk):
                pairs.append((img, mrk))
    pairs.sort()
    return pairs


def pair_key(img_path, images_root):
    """Относительный ключ пары для сравнения между папками."""
    p = Path(img_path)
    try:
        return str(p.relative_to(images_root))
    except ValueError:
        return p.name


def rect_xywh_to_quad(x, y, w, h):
    """Axis-aligned rect (x, y, w, h) → TL, TR, BR, BL quad."""
    return [
        [float(x), float(y)],
        [float(x) + float(w), float(y)],
        [float(x) + float(w), float(y) + float(h)],
        [float(x), float(y) + float(h)],
    ]


def quad_aabb(quad):
    """Return (xmin, ymin, xmax, ymax) or None if the quad is empty/invalid."""
    if not quad or len(quad) < 2:
        return None
    xs, ys = [], []
    for pt in quad:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def is_new_face_markup(data):
    if not isinstance(data, dict):
        return False
    if data.get('markup_version') is not None:
        return True
    if isinstance(data.get('main_face'), dict):
        return True
    return any(
        isinstance(k, str) and k.startswith('other_face_') and isinstance(v, dict)
        for k, v in data.items()
    )


def _strip_other_index(name):
    match = _OTHER_INDEX_RE.search(name)
    if not match:
        return name, None
    return name[: match.start()], int(match.group(1))


def _old_point_to_xy(val):
    """[[x, y]] or [x, y] → (x, y) or None if missing."""
    if not val:
        return None
    if isinstance(val, dict) and 'x' in val and 'y' in val:
        x, y = float(val['x']), float(val['y'])
        return None if is_missing(x, y) else (x, y)
    pt = val[0] if isinstance(val[0], (list, tuple)) else val
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    x, y = float(pt[0]), float(pt[1])
    return None if is_missing(x, y) else (x, y)


def _old_point_to_new(val):
    xy = _old_point_to_xy(val)
    if xy is None:
        return None
    return {'x': xy[0], 'y': xy[1]}


def _old_rect_xywh_and_type(rect_field):
    """Parse old face_rect field. Returns (x, y, w, h, face_type) or None."""
    if not rect_field:
        return None
    # Main / suffix: [[x, y, w, h], "nrotations90_0", "doc"?]
    first = rect_field[0]
    if isinstance(first, (list, tuple)) and len(first) >= 4 and not isinstance(first[0], (list, tuple)):
        x, y, w, h = map(float, first[:4])
        if w <= 0 or h <= 0:
            return None
        face_type = 'doc'
        if len(rect_field) > 2 and isinstance(rect_field[2], str):
            face_type = rect_field[2]
        return x, y, w, h, face_type
    # List-format other_face_rect item: [x, y, w, h]
    if len(rect_field) >= 4 and all(isinstance(v, (int, float)) for v in rect_field[:4]):
        x, y, w, h = map(float, rect_field[:4])
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h, 'doc'
    return None


def _create_structure_face(quad, keypoints, face_type, is_confident=True):
    return {
        'attributes': [],
        'face_quad': quad or [],
        'face_type': face_type or 'doc',
        'is_confident': bool(is_confident),
        'keypoints': keypoints or {},
    }


def _suffix_other_indexes(props):
    indexes = set()
    for key in props:
        if not isinstance(key, str) or not key.startswith('other'):
            continue
        _, idx = _strip_other_index(key)
        if idx is not None:
            indexes.add(idx)
    return indexes


def _old_keypoint_short_names(props):
    names = set(NEW_FORMAT_KP_NAMES)
    for key in props:
        if not isinstance(key, str):
            continue
        if key.startswith('other_kp_'):
            rest = key[len('other_kp_'):]
        elif key.startswith('kp_'):
            rest = key[3:]
        else:
            continue
        short, _ = _strip_other_index(rest)
        if short:
            names.add(short)
    return names


def _keypoints_from_old_props(props, *, suffix='', list_idx=None):
    keypoints = {}
    for short in _old_keypoint_short_names(props):
        point = None
        if list_idx is None:
            old_key = f'kp_{short}{suffix}'
            old_key_2 = f'other_kp_{short}{suffix}'
            if old_key in props:
                point = _old_point_to_new(props[old_key])
            # other_kp_* without a numeric suffix is the list-format of extra faces,
            # not the main face. Only read it for suffixed other-face keys.
            if suffix and old_key_2 in props:
                point = _old_point_to_new(props[old_key_2])
        else:
            lst = props.get(f'other_kp_{short}', [])
            if isinstance(lst, list) and list_idx < len(lst):
                point = _old_point_to_new(lst[list_idx])
        if point is not None:
            keypoints[short] = point
    return keypoints


def _face_from_old_rect(rect_field, keypoints):
    parsed = _old_rect_xywh_and_type(rect_field)
    if parsed is None:
        return None
    x, y, w, h, face_type = parsed
    return _create_structure_face(rect_xywh_to_quad(x, y, w, h), keypoints, face_type)


def convert_old_markup_to_new(data):
    """Convert old props-based markup to main_face / other_face_* structure."""
    if is_new_face_markup(data):
        return data
    props = data.get('props') if isinstance(data, dict) else None
    result = {'markup_version': 1}
    if not isinstance(props, dict):
        return result

    main_kps = _keypoints_from_old_props(props, suffix='')
    main_face = _face_from_old_rect(props.get('face_rect'), main_kps)
    if main_face is not None:
        result['main_face'] = main_face

    suffix_indexes = _suffix_other_indexes(props)
    if suffix_indexes:
        for num in sorted(suffix_indexes):
            suf = f'_{num}'
            rect_field = props.get(f'face_rect{suf}', props.get(f'other_face_rect{suf}'))
            kps = _keypoints_from_old_props(props, suffix=suf)
            face = _face_from_old_rect(rect_field, kps)
            if face is not None:
                result[f'other_face_{num}'] = face
        return result

    other_rects = props.get('other_face_rect', [])
    if not isinstance(other_rects, list):
        other_rects = []
    n_other = len(other_rects)
    if n_other == 0:
        for key, val in props.items():
            if isinstance(key, str) and key.startswith('other_kp_') and isinstance(val, list):
                n_other = max(n_other, len(val))
    for idx in range(n_other):
        rect_field = other_rects[idx] if idx < len(other_rects) else None
        kps = _keypoints_from_old_props(props, list_idx=idx)
        face = _face_from_old_rect(rect_field, kps)
        if face is not None:
            result[f'other_face_{idx + 1}'] = face
    return result


def _other_face_sort_key(name):
    _, idx = _strip_other_index(name)
    return idx if idx is not None else 0


def iter_markup_faces(data):
    """Yield (face_key, face_dict) for main_face and other_face_* in order."""
    if not is_new_face_markup(data):
        data = convert_old_markup_to_new(data)
    if not isinstance(data, dict):
        return
    if isinstance(data.get('main_face'), dict):
        yield 'main_face', data['main_face']
    other_keys = [
        key for key, val in data.items()
        if isinstance(key, str) and key.startswith('other_face_') and isinstance(val, dict)
    ]
    for key in sorted(other_keys, key=_other_face_sort_key):
        yield key, data[key]


def read_new_keypoint(keypoints, short_name):
    if not keypoints:
        return None
    return _old_point_to_xy(keypoints.get(short_name))


def face_has_valid_train_keypoints(face):
    keypoints = (face or {}).get('keypoints') or {}
    for short in TRAIN_KP_SHORT_NAMES:
        if read_new_keypoint(keypoints, short) is not None:
            return True
    return False

