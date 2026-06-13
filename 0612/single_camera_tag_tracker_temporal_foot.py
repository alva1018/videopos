import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Single-camera AprilTag + person tracking debugger with foot-point overlays."
    )
    parser.add_argument("--config", default=str(script_dir / "scenario_config.json"))
    parser.add_argument("--camera-id", default="cam_a")
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--video", default=None, help="Override the camera video path from config.")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker, usually botsort.yaml or bytetrack.yaml.")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=None, help="Override tag size in meters.")
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--apriltag-threads", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tag-fps", type=float, default=5.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=str(script_dir / "outputs_single_camera_tag_tracker"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true", default=True)
    parser.add_argument("--draw-projection-roi", action="store_true", default=True)
    parser.add_argument("--no-map-projection", action="store_true", help="Do not project foot points through camera homography.")
    parser.add_argument("--foot-mode", choices=["bbox", "temporal"], default="temporal")
    parser.add_argument("--max-foot-speed-mps", type=float, default=2.5)
    parser.add_argument("--min-speed-history", type=int, default=4)
    parser.add_argument("--occlusion-x-overlap", type=float, default=0.35)
    parser.add_argument("--truncated-height-ratio", type=float, default=0.65)
    parser.add_argument("--foot-stale-sec", type=float, default=1.0)
    return parser.parse_args()


@dataclass
class FootHistory:
    reliable: List[Tuple[float, Point, Point, float]] = field(default_factory=list)

    def add_reliable(self, timestamp: float, foot: Point, map_point: Point, bbox_height: float) -> None:
        self.reliable.append((timestamp, foot, map_point, bbox_height))
        self.reliable = self.reliable[-8:]

    def last(self) -> Optional[Tuple[float, Point, Point, float]]:
        if not self.reliable:
            return None
        return self.reliable[-1]

    def median_height(self) -> Optional[float]:
        if not self.reliable:
            return None
        heights = sorted(item[3] for item in self.reliable)
        mid = len(heights) // 2
        if len(heights) % 2:
            return heights[mid]
        return (heights[mid - 1] + heights[mid]) * 0.5

    def predict_map(self, timestamp: float) -> Optional[Point]:
        if not self.reliable:
            return None
        if len(self.reliable) == 1:
            return self.reliable[-1][2]

        t1, _, p1, _ = self.reliable[-2]
        t2, _, p2, _ = self.reliable[-1]
        dt = max(1e-6, t2 - t1)
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        horizon = max(0.0, timestamp - t2)
        return (p2[0] + vx * horizon, p2[1] + vy * horizon)


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"OpenCV is required. Install with: \"{sys.executable}\" -m pip install opencv-python") from exc
    return cv2


def load_yolo(model_path: str) -> Any:
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        print("ultralytics is not installed in the Python that is running this script.")
        print(f"Python executable: {sys.executable}")
        print(f"Install with: \"{sys.executable}\" -m pip install ultralytics")
        raise SystemExit(2) from exc
    return YOLO(model_path)


def load_apriltag_detector() -> Any:
    try:
        from pupil_apriltags import Detector  # type: ignore
    except ImportError as exc:
        print("pupil_apriltags is not installed in the Python that is running this script.")
        print(f"Python executable: {sys.executable}")
        print(f"Install with: \"{sys.executable}\" -m pip install pupil-apriltags")
        raise SystemExit(2) from exc
    return Detector


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_camera(config: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    for camera in config.get("cameras", []):
        if camera.get("camera_id") == camera_id:
            return camera
    raise ValueError(f"Camera id not found in config: {camera_id}")


def resolve_path(path: str, base_dir: Path) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item
    return base_dir / item


def draw_text(
    cv2: Any,
    frame: Any,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_polyline(cv2: Any, frame: Any, points: Sequence[Sequence[float]], color: Tuple[int, int, int], thickness: int) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness, lineType=cv2.LINE_AA)


def convex_hull_order(points: Sequence[Tuple[str, Point]]) -> List[str]:
    """Return landmark ids ordered around the map-space outer boundary."""
    if len(points) <= 2:
        return [item[0] for item in points]

    unique = sorted(
        {item[0]: (float(item[1][0]), float(item[1][1])) for item in points}.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )

    def cross(origin: Point, a: Point, b: Point) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Tuple[str, Point]] = []
    for item in unique:
        while len(lower) >= 2 and cross(lower[-2][1], lower[-1][1], item[1]) <= 0:
            lower.pop()
        lower.append(item)

    upper: List[Tuple[str, Point]] = []
    for item in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2][1], upper[-1][1], item[1]) <= 0:
            upper.pop()
        upper.append(item)

    hull = lower[:-1] + upper[:-1]
    if len(hull) >= 3:
        return [item[0] for item in hull]

    cx = sum(item[1][0] for item in points) / len(points)
    cy = sum(item[1][1] for item in points) / len(points)
    return [
        item[0]
        for item in sorted(
            points,
            key=lambda item: math.atan2(item[1][1] - cy, item[1][0] - cx),
        )
    ]


def projection_polygon_from_landmarks(config: Dict[str, Any], camera: Dict[str, Any]) -> List[Point]:
    ids = [str(item) for item in camera.get("projection_landmark_ids", [])]
    landmarks = config.get("landmarks", {})
    landmark_pixels = camera.get("landmark_pixels", {})
    if len(ids) < 3:
        return []

    map_points: List[Tuple[str, Point]] = []
    for landmark_id in ids:
        if landmark_id not in landmarks or landmark_id not in landmark_pixels:
            return []
        coord = landmarks[landmark_id]
        map_points.append((landmark_id, (float(coord[0]), float(coord[1]))))

    ordered_ids = convex_hull_order(map_points)
    polygon: List[Point] = []
    for landmark_id in ordered_ids:
        pixel = landmark_pixels[landmark_id]
        polygon.append((float(pixel[0]), float(pixel[1])))
    return polygon


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    palette = [
        (66, 133, 244),
        (52, 168, 83),
        (251, 188, 5),
        (234, 67, 53),
        (171, 71, 188),
        (0, 188, 212),
        (255, 112, 67),
        (124, 179, 66),
    ]
    return palette[track_id % len(palette)]


def bbox_contains(bbox: BBox, point: Point) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_bbox(bbox: BBox, scale: float, frame_w: int, frame_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return (
        max(0.0, cx - w * 0.5),
        max(0.0, cy - h * 0.5),
        min(float(frame_w), cx + w * 0.5),
        min(float(frame_h), cy + h * 0.5),
    )


def upper_body_roi(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    upper = (x1, y1, x2, y1 + (y2 - y1) * 0.58)
    return expand_bbox(upper, 1.35, frame_w, frame_h)


def detect_tags_in_roi(
    detector: Any,
    gray: Any,
    roi: BBox,
    camera_params: Sequence[float],
    tag_size: float,
) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    if x2 <= x1 or y2 <= y1:
        return []

    roi_gray = gray[y1:y2, x1:x2]
    roi_params = [
        float(camera_params[0]),
        float(camera_params[1]),
        float(camera_params[2]) - x1,
        float(camera_params[3]) - y1,
    ]
    detections = detector.detect(
        roi_gray,
        estimate_tag_pose=True,
        camera_params=roi_params,
        tag_size=tag_size,
    )

    tags: List[Dict[str, Any]] = []
    for det in detections:
        center = (float(det.center[0] + x1), float(det.center[1] + y1))
        corners = (det.corners + np.array([x1, y1])).astype(float).tolist()
        distance = None
        if getattr(det, "pose_t", None) is not None:
            distance = float(np.linalg.norm(det.pose_t))
        tags.append(
            {
                "tag_id": int(det.tag_id),
                "center": center,
                "corners": corners,
                "distance_m": distance,
            }
        )
    return tags


def parse_yolo_tracks(results: Iterable[Any]) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    fallback_id = 1
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = None
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        for idx, (bbox_arr, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = [float(v) for v in bbox_arr.tolist()]
            track_id = int(ids[idx]) if ids is not None else fallback_id
            fallback_id += 1
            foot = ((x1 + x2) * 0.5, y2)
            tracks.append(
                {
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                    "foot": foot,
                    "raw_foot": foot,
                    "foot_source": "bbox_bottom",
                    "foot_quality": "unscored",
                    "foot_occluded": False,
                    "map_point_raw": None,
                    "confidence": float(conf),
                    "tag_id": None,
                    "tag_center": None,
                    "map_point": None,
                }
            )
    return tracks


def bind_tags_to_tracks(tags: Sequence[Dict[str, Any]], tracks: List[Dict[str, Any]]) -> None:
    for tag in tags:
        center = tag["center"]
        candidates = [track for track in tracks if bbox_contains(track["bbox"], center)]
        if not candidates:
            continue
        track = min(
            candidates,
            key=lambda item: math.dist(item["center"], center),
        )
        track["tag_id"] = int(tag["tag_id"])
        track["tag_center"] = center


def homography_matrix(camera: Dict[str, Any]) -> Optional[np.ndarray]:
    homography = camera.get("homography", {})
    image_points = homography.get("image_points", [])
    map_points = homography.get("map_points", [])
    if not homography.get("enabled", False) or len(image_points) < 4 or len(image_points) != len(map_points):
        return None
    matrix, _ = cv2_find_homography(image_points, map_points)
    return matrix


def cv2_find_homography(image_points: Sequence[Sequence[float]], map_points: Sequence[Sequence[float]]) -> Tuple[Any, Any]:
    import cv2  # type: ignore

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(map_points, dtype=np.float64)
    return cv2.findHomography(src, dst, 0)


def project_point(point: Point, matrix: Optional[np.ndarray]) -> Optional[Point]:
    if matrix is None:
        return None
    vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = matrix @ vec
    denom = float(projected[2])
    if abs(denom) < 1e-9:
        return None
    return (float(projected[0] / denom), float(projected[1] / denom))


def inverse_homography(matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def bbox_height(track: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = track["bbox"]
    return max(0.0, float(y2 - y1))


def x_overlap_ratio(a: BBox, b: BBox) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_width = max(1.0, min(ax2 - ax1, bx2 - bx1))
    return overlap / min_width


def mark_occluded_tracks(tracks: List[Dict[str, Any]], min_x_overlap: float) -> None:
    for track in tracks:
        track["foot_occluded"] = False
        track["occluder_track_id"] = None

    for candidate in tracks:
        cx1, cy1, cx2, cy2 = candidate["bbox"]
        for possible_occluder in tracks:
            if candidate is possible_occluder:
                continue
            ox1, oy1, ox2, oy2 = possible_occluder["bbox"]
            if x_overlap_ratio(candidate["bbox"], possible_occluder["bbox"]) < min_x_overlap:
                continue

            candidate_bottom_inside_occluder = oy1 <= cy2 <= oy2
            occluder_is_lower = oy2 > cy2
            candidate_top_visible = cy1 < oy1
            if candidate_bottom_inside_occluder and occluder_is_lower and candidate_top_visible:
                candidate["foot_occluded"] = True
                candidate["occluder_track_id"] = possible_occluder["track_id"]
                break


def update_foot_estimates(
    tracks: List[Dict[str, Any]],
    histories: Dict[int, FootHistory],
    timestamp: float,
    homography: Optional[np.ndarray],
    inverse_map_to_image: Optional[np.ndarray],
    foot_mode: str,
    max_speed_mps: float,
    min_speed_history: int,
    stale_sec: float,
    truncated_height_ratio: float,
) -> None:
    for track in tracks:
        track_id = int(track["track_id"])
        history = histories.setdefault(track_id, FootHistory())
        raw_foot = track["raw_foot"]
        raw_map = project_point(raw_foot, homography)
        track["map_point_raw"] = raw_map
        track["foot"] = raw_foot
        track["map_point"] = raw_map
        track["foot_source"] = "bbox_bottom"
        track["foot_quality"] = "reliable"

        if foot_mode == "bbox" or raw_map is None:
            if raw_map is None:
                track["foot_quality"] = "invalid"
            elif not track.get("foot_occluded"):
                history.add_reliable(timestamp, raw_foot, raw_map, bbox_height(track))
            continue

        reasons: List[str] = []
        if track.get("foot_occluded"):
            reasons.append("occluded")

        median_height = history.median_height()
        if median_height is not None and bbox_height(track) < median_height * truncated_height_ratio:
            reasons.append("truncated")

        last = history.last()
        if last is not None and len(history.reliable) >= min_speed_history:
            last_t, _, last_map, _ = last
            dt = max(1e-6, timestamp - last_t)
            speed = math.dist(raw_map, last_map) / dt
            if speed > max_speed_mps:
                reasons.append("speed_jump")

        if not reasons:
            history.add_reliable(timestamp, raw_foot, raw_map, bbox_height(track))
            continue

        predicted = history.predict_map(timestamp)
        if predicted is None or last is None:
            track["map_point"] = None
            track["foot_quality"] = "invalid"
            track["foot_source"] = "+".join(reasons)
            continue

        age = timestamp - last[0]
        if age > stale_sec:
            track["map_point"] = None
            track["foot_quality"] = "stale"
            track["foot_source"] = "+".join(reasons)
            continue

        track["map_point"] = predicted
        predicted_foot = project_point(predicted, inverse_map_to_image)
        if predicted_foot is not None:
            track["foot"] = predicted_foot
        track["foot_quality"] = "occluded" if "occluded" in reasons else "predicted"
        track["foot_source"] = "predicted_map:" + "+".join(reasons)


def write_jsonl_row(fp: Any, row: Dict[str, Any]) -> None:
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def rounded_point(point: Optional[Point], digits: int = 1) -> Optional[List[float]]:
    if point is None:
        return None
    return [round(float(point[0]), digits), round(float(point[1]), digits)]


def draw_scene_guides(
    cv2: Any,
    frame: Any,
    config: Dict[str, Any],
    camera: Dict[str, Any],
    draw_landmarks: bool,
    draw_projection_roi: bool,
) -> None:
    if draw_projection_roi:
        polygon = projection_polygon_from_landmarks(config, camera) or camera.get("projection_image_polygon", [])
        draw_polyline(cv2, frame, polygon, (255, 180, 40), 3)
        if polygon:
            x, y = polygon[0]
            draw_text(cv2, frame, "projection ROI", (int(x), int(y) - 10), (255, 180, 40), 0.55)

    if draw_landmarks:
        for landmark_id, pixel in camera.get("landmark_pixels", {}).items():
            x, y = int(round(pixel[0])), int(round(pixel[1]))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
            draw_text(cv2, frame, str(landmark_id), (x + 7, y - 7), (0, 255, 255), 0.5)


def draw_tags(cv2: Any, frame: Any, tags: Sequence[Dict[str, Any]]) -> None:
    for tag in tags:
        corners = tag.get("corners", [])
        if corners:
            draw_polyline(cv2, frame, corners, (0, 255, 0), 2)
        cx, cy = tag["center"]
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1, cv2.LINE_AA)
        label = f"tag:{tag['tag_id']}"
        if tag.get("distance_m") is not None:
            label += f" {tag['distance_m']:.2f}m"
        draw_text(cv2, frame, label, (int(cx) + 8, int(cy) - 8), (0, 255, 0), 0.55)


def draw_tracks(cv2: Any, frame: Any, tracks: Sequence[Dict[str, Any]]) -> None:
    for track in tracks:
        track_id = int(track["track_id"])
        color = color_for_track(track_id)
        x1, y1, x2, y2 = [int(round(v)) for v in track["bbox"]]
        raw_foot = track.get("raw_foot", track["foot"])
        raw_foot_px = (int(round(raw_foot[0])), int(round(raw_foot[1])))
        foot = track["foot"]
        foot_px = (int(round(foot[0])), int(round(foot[1])))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(frame, raw_foot_px, 4, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, foot_px, 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(frame, (foot_px[0] - 10, foot_px[1]), (foot_px[0] + 10, foot_px[1]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (foot_px[0], foot_px[1] - 10), (foot_px[0], foot_px[1] + 10), (0, 0, 255), 2, cv2.LINE_AA)

        label = (
            f"track:{track_id} conf:{track['confidence']:.2f} "
            f"foot:({foot[0]:.0f},{foot[1]:.0f}) {track.get('foot_quality', '')}"
        )
        if track.get("tag_id") is not None:
            label += f" tag:{track['tag_id']}"
        draw_text(cv2, frame, label, (x1, max(18, y1 - 8)), color, 0.55)

        map_point = track.get("map_point")
        if map_point is not None:
            draw_text(
                cv2,
                frame,
                f"map:({map_point[0]:.2f},{map_point[1]:.2f}) {track.get('foot_source', '')}",
                (x1, min(frame.shape[0] - 12, y2 + 22)),
                (230, 230, 230),
                0.5,
            )


def main() -> int:
    args = parse_args()
    cv2 = load_cv2()
    Detector = load_apriltag_detector()
    config_path = Path(args.config)
    base_dir = Path(args.base_dir)
    config = load_config(config_path)
    camera = find_camera(config, args.camera_id)
    video_path = resolve_path(args.video or camera["video_path"], base_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_yolo(args.yolo_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(args.start_sec * fps)))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{args.camera_id}_tag_tracks.jsonl"
    video_writer = None
    if args.save_video:
        video_path_out = output_dir / f"{args.camera_id}_tag_tracks.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path_out), fourcc, fps, (frame_w, frame_h))

    tag_size = float(args.tag_size if args.tag_size is not None else camera.get("tag_size", 0.1))
    detector = Detector(
        families=args.tag_family,
        nthreads=args.apriltag_threads,
        quad_decimate=args.quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    h_matrix = None if args.no_map_projection else homography_matrix(camera)
    inverse_h_matrix = inverse_homography(h_matrix)
    tag_interval = max(1, int(round(fps / max(args.tag_fps, 1e-6))))
    foot_histories: Dict[int, FootHistory] = {}

    print(f"Camera: {args.camera_id}")
    print(f"Video: {video_path}")
    print(f"Frame size: {frame_w}x{frame_h}, fps={fps:.3f}")
    print(f"Output JSONL: {jsonl_path}")

    processed = 0
    frame_idx = start_frame - 1
    with jsonl_path.open("w", encoding="utf-8") as fp:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % max(1, args.frame_step) != 0:
                continue
            if args.max_frames is not None and processed >= args.max_frames:
                break
            processed += 1
            timestamp = frame_idx / fps

            results = model.track(
                frame,
                classes=[0],
                conf=args.yolo_conf,
                persist=True,
                tracker=args.tracker,
                verbose=False,
            )
            tracks = parse_yolo_tracks(results)

            tags: List[Dict[str, Any]] = []
            if frame_idx % tag_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rois = [upper_body_roi(track["bbox"], frame_w, frame_h) for track in tracks]
                if not rois:
                    rois = [(0.0, 0.0, float(frame_w), float(frame_h))]
                for roi in rois:
                    tags.extend(detect_tags_in_roi(detector, gray, roi, camera["camera_params"], tag_size))

            bind_tags_to_tracks(tags, tracks)
            mark_occluded_tracks(tracks, args.occlusion_x_overlap)
            update_foot_estimates(
                tracks,
                foot_histories,
                timestamp,
                h_matrix,
                inverse_h_matrix,
                args.foot_mode,
                args.max_foot_speed_mps,
                args.min_speed_history,
                args.foot_stale_sec,
                args.truncated_height_ratio,
            )
            for track in tracks:
                write_jsonl_row(
                    fp,
                    {
                        "timestamp": round(timestamp, 3),
                        "frame_idx": frame_idx,
                        "camera_id": args.camera_id,
                        "track_id": int(track["track_id"]),
                        "tag_id": track.get("tag_id"),
                        "bbox": [round(float(v), 1) for v in track["bbox"]],
                        "center": rounded_point(track["center"]),
                        "raw_foot": rounded_point(track.get("raw_foot")),
                        "foot": rounded_point(track["foot"]),
                        "foot_source": track.get("foot_source"),
                        "foot_quality": track.get("foot_quality"),
                        "foot_occluded": bool(track.get("foot_occluded")),
                        "occluder_track_id": track.get("occluder_track_id"),
                        "map_point_raw": rounded_point(track.get("map_point_raw"), 3),
                        "map_point": rounded_point(track.get("map_point"), 3),
                        "confidence": round(float(track["confidence"]), 3),
                    },
                )

            draw_scene_guides(cv2, frame, config, camera, args.draw_landmarks, args.draw_projection_roi)
            draw_tags(cv2, frame, tags)
            draw_tracks(cv2, frame, tracks)
            draw_text(
                cv2,
                frame,
                f"{args.camera_id} frame:{frame_idx} time:{timestamp:.2f}s tracks:{len(tracks)} tags:{len(tags)}",
                (16, 32),
                (255, 255, 255),
                0.7,
            )

            if video_writer is not None:
                video_writer.write(frame)
            if args.show:
                cv2.imshow("single camera tag tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Processed frames: {processed}")
    print(f"Wrote tracks: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
