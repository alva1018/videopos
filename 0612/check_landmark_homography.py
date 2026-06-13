import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Check camera landmark pixel/map correspondences with homography reprojection errors."
    )
    parser.add_argument("--config", default=str(script_dir / "scenario_config.json"))
    parser.add_argument("--max-error-m", type=float, default=0.30)
    parser.add_argument("--max-loo-error-m", type=float, default=0.50)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def point_from_raw(raw: Sequence[float]) -> Point:
    return (float(raw[0]), float(raw[1]))


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix)
    return projected.reshape(-1, 2)


def fit_homography(image_points: Sequence[Point], map_points: Sequence[Point]) -> Optional[np.ndarray]:
    if len(image_points) < 4 or len(image_points) != len(map_points):
        return None
    src = np.array(image_points, dtype=np.float64)
    dst = np.array(map_points, dtype=np.float64)
    matrix, _ = cv2.findHomography(src, dst, 0)
    return matrix


def rmse(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return math.sqrt(sum(value * value for value in items) / len(items))


def leave_one_out_errors(image_points: List[Point], map_points: List[Point]) -> List[Optional[float]]:
    errors: List[Optional[float]] = []
    for idx in range(len(image_points)):
        train_image = image_points[:idx] + image_points[idx + 1 :]
        train_map = map_points[:idx] + map_points[idx + 1 :]
        matrix = fit_homography(train_image, train_map)
        if matrix is None:
            errors.append(None)
            continue
        predicted = project_points(np.array([image_points[idx]], dtype=np.float64), matrix)[0]
        target = np.array(map_points[idx], dtype=np.float64)
        errors.append(float(np.linalg.norm(predicted - target)))
    return errors


def camera_correspondences(
    camera: Dict[str, Any],
    landmarks: Dict[str, Sequence[float]],
) -> Tuple[List[str], List[Point], List[Point]]:
    landmark_pixels = camera.get("landmark_pixels", {})
    ids: List[str] = []
    image_points: List[Point] = []
    map_points: List[Point] = []
    missing = []

    for landmark_id, pixel in landmark_pixels.items():
        key = str(landmark_id)
        if key not in landmarks:
            missing.append(key)
            continue
        ids.append(key)
        image_points.append(point_from_raw(pixel))
        map_points.append(point_from_raw(landmarks[key]))

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{camera.get('camera_id', '<unknown>')} references missing landmarks: {joined}")
    return ids, image_points, map_points


def print_camera_report(
    camera: Dict[str, Any],
    landmarks: Dict[str, Sequence[float]],
    max_error_m: float,
    max_loo_error_m: float,
) -> int:
    camera_id = camera.get("camera_id", "<unknown>")
    ids, image_points, map_points = camera_correspondences(camera, landmarks)
    matrix = fit_homography(image_points, map_points)
    if matrix is None:
        print(f"\n{camera_id}: FAILED - need at least 4 matched landmarks")
        return 1

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(map_points, dtype=np.float64)
    predicted = project_points(src, matrix)
    errors = np.linalg.norm(predicted - dst, axis=1)
    loo_errors = leave_one_out_errors(image_points, map_points)

    print(f"\n{camera_id}")
    print(f"  landmarks: {len(ids)}")
    print(f"  mean error: {float(errors.mean()):.4f} m")
    print(f"  RMSE:       {rmse(float(v) for v in errors):.4f} m")
    print(f"  max error:  {float(errors.max()):.4f} m")

    suspicious_count = 0
    rows = sorted(
        zip(ids, image_points, map_points, predicted.tolist(), errors.tolist(), loo_errors),
        key=lambda item: max(float(item[4]), float(item[5] or 0.0)),
        reverse=True,
    )
    for landmark_id, pixel, target, pred, err, loo_err in rows:
        flags = []
        if err > max_error_m:
            flags.append("all-point")
        if loo_err is not None and loo_err > max_loo_error_m:
            flags.append("leave-one-out")
        if flags:
            suspicious_count += 1
        flag_text = f"  <-- suspicious: {', '.join(flags)}" if flags else ""
        loo_text = "n/a" if loo_err is None else f"{loo_err:.4f}"
        print(
            "  "
            f"id={landmark_id:>2} pixel=({pixel[0]:.1f}, {pixel[1]:.1f}) "
            f"target=({target[0]:.3f}, {target[1]:.3f}) "
            f"pred=({pred[0]:.3f}, {pred[1]:.3f}) "
            f"err={err:.4f}m loo={loo_text}m"
            f"{flag_text}"
        )

    if suspicious_count:
        print(f"  result: CHECK {suspicious_count} suspicious landmark(s)")
        return 1

    print("  result: OK")
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    landmarks = config.get("landmarks", {})
    if not landmarks:
        raise ValueError("Config must contain a non-empty landmarks object.")

    status = 0
    for camera in config.get("cameras", []):
        status |= print_camera_report(
            camera,
            landmarks,
            max_error_m=args.max_error_m,
            max_loo_error_m=args.max_loo_error_m,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
