import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CONFIRMED_WINDOW_SEC = 1.0
TRACKED_WINDOW_SEC = 5.0


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class AreaConfig:
    name: str
    polygon: List[Point]
    position: Point
    priority: int = 0


@dataclass
class CameraConfig:
    camera_id: str
    video_path: str
    camera_params: List[float]
    tag_size: float
    areas: List[AreaConfig]
    door_rois: Dict[str, List[Point]] = field(default_factory=dict)


@dataclass
class LocalTrack:
    local_track_id: int
    bbox: BBox
    is_real_person: bool = False
    tag_id: Optional[int] = None
    last_seen_ts: float = 0.0
    last_tag_seen_ts: Optional[float] = None
    area: str = "Unknown"
    position: Point = (0.0, 0.0)
    confidence: float = 0.0


@dataclass
class VisualCameraState:
    camera: CameraConfig
    cap: Any
    fps: float
    detector: Any
    tracks: List[LocalTrack] = field(default_factory=list)
    next_track_id: int = 1
    frame_idx: int = -1
    next_output_sec: int = 0
    last_frame: Optional[Any] = None
    last_tags: List[Dict[str, Any]] = field(default_factory=list)
    last_tag_rois: List[BBox] = field(default_factory=list)
    last_people: List[Tuple[BBox, float]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)


def load_optional_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to process videos. Install it with: "
            "pip install opencv-python"
        ) from exc
    return cv2


def load_optional_apriltag_detector():
    try:
        from pupil_apriltags import Detector  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pupil_apriltags is required for tag detection. Install it with: "
            "pip install pupil-apriltags"
        ) from exc
    return Detector


def load_optional_yolo(model_path: Optional[str]):
    if not model_path:
        return None
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        print("YOLO disabled: ultralytics is not installed in the Python that is running this script.")
        print(f"Python executable: {sys.executable}")
        print(f"Install with: \"{sys.executable}\" -m pip install ultralytics")
        return None
    return YOLO(model_path)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def normalize_polygon_points(points: Sequence[Sequence[float]]) -> List[Point]:
    normalized = [(float(point[0]), float(point[1])) for point in points]
    if len(normalized) <= 2:
        return normalized
    cx = sum(point[0] for point in normalized) / len(normalized)
    cy = sum(point[1] for point in normalized) / len(normalized)
    return sorted(normalized, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))


def is_door_area(name: str) -> bool:
    return name.lower().startswith("door")


def is_point_in_lower_polygon_band(point: Point, polygon: Sequence[Point], ratio: float = 0.62) -> bool:
    if not polygon:
        return False
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    return point[1] >= min_y + (max_y - min_y) * ratio


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    if not polygon:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def nearest_area_by_pixel(point: Point, areas: Sequence[AreaConfig]) -> Optional[AreaConfig]:
    if not areas:
        return None
    return min(areas, key=lambda area: math.dist(point, polygon_centroid(area.polygon)))


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


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


def area_for_point(point: Point, areas: Sequence[AreaConfig]) -> Tuple[str, Point]:
    matches = [area for area in areas if point_in_polygon(point, area.polygon)]
    if matches:
        best = max(matches, key=lambda area: area.priority)
        return best.name, best.position
    if not areas:
        return "Unknown", point
    nearest = min(
        areas,
        key=lambda area: math.dist(point, area.position),
    )
    return nearest.name, nearest.position


def area_for_track(track: LocalTrack, camera: CameraConfig) -> Tuple[str, Point]:
    foot = ((track.bbox[0] + track.bbox[2]) * 0.5, track.bbox[3])
    matches = [area for area in camera.areas if point_in_polygon(foot, area.polygon)]
    non_door_areas = [area for area in camera.areas if not is_door_area(area.name)]
    if matches:
        non_doors = [area for area in matches if not is_door_area(area.name)]
        door_candidates = [
            area
            for area in matches
            if is_door_area(area.name) and track.is_real_person and is_point_in_lower_polygon_band(foot, area.polygon)
        ]
        candidates = door_candidates or non_doors
        if candidates:
            best = max(candidates, key=lambda area: area.priority)
            return best.name, best.position
        nearest_non_door = nearest_area_by_pixel(foot, non_door_areas)
        if nearest_non_door is not None:
            return nearest_non_door.name, nearest_non_door.position
    if not track.is_real_person:
        if non_door_areas:
            nearest = nearest_area_by_pixel(foot, non_door_areas)
            if nearest is None:
                return "Unknown", foot
            return nearest.name, nearest.position
    nearest = nearest_area_by_pixel(foot, camera.areas)
    if nearest is not None:
        return nearest.name, nearest.position
    return "Unknown", foot


def camera_from_dict(raw: Dict[str, Any]) -> CameraConfig:
    areas = [
        AreaConfig(
            name=item["name"],
            polygon=normalize_polygon_points(item["polygon"]),
            position=tuple(item["position"]),
            priority=int(item.get("priority", 0)),
        )
        for item in raw.get("areas", [])
    ]
    door_rois = {
        name: normalize_polygon_points(polygon)
        for name, polygon in raw.get("door_rois", {}).items()
    }
    return CameraConfig(
        camera_id=raw["camera_id"],
        video_path=raw["video_path"],
        camera_params=[float(v) for v in raw.get("camera_params", [720, 720, 960, 540])],
        tag_size=float(raw.get("tag_size", 0.1)),
        areas=areas,
        door_rois=door_rois,
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_people_yolo(model: Any, frame: Any, conf: float) -> List[Tuple[BBox, float]]:
    if model is None:
        return []
    results = model.predict(frame, classes=[0], conf=conf, verbose=False)
    people: List[Tuple[BBox, float]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            people.append(((xyxy[0], xyxy[1], xyxy[2], xyxy[3]), score))
    return people


def update_tracks(
    tracks: List[LocalTrack],
    detections: List[Tuple[BBox, float]],
    timestamp: float,
    next_track_id: int,
    min_iou: float,
) -> Tuple[List[LocalTrack], int]:
    unmatched_tracks = set(range(len(tracks)))
    for bbox, score in detections:
        best_idx = None
        best_iou = 0.0
        for idx in unmatched_tracks:
            iou = bbox_iou(tracks[idx].bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= min_iou:
            track = tracks[best_idx]
            track.bbox = bbox
            track.is_real_person = True
            track.last_seen_ts = timestamp
            track.confidence = max(track.confidence, score)
            unmatched_tracks.remove(best_idx)
        else:
            tracks.append(
                LocalTrack(
                    local_track_id=next_track_id,
                    bbox=bbox,
                    is_real_person=True,
                    last_seen_ts=timestamp,
                    confidence=score,
                )
            )
            next_track_id += 1
    tracks = [
        track
        for track in tracks
        if timestamp - track.last_seen_ts <= TRACKED_WINDOW_SEC
        or (
            track.last_tag_seen_ts is not None
            and timestamp - track.last_tag_seen_ts <= TRACKED_WINDOW_SEC
        )
    ]
    return tracks, next_track_id


def detect_tags_in_roi(
    detector: Any,
    gray: Any,
    roi: BBox,
    camera_params: List[float],
    tag_size: float,
) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    if x2 <= x1 or y2 <= y1:
        return []
    roi_gray = gray[y1:y2, x1:x2]
    roi_params = [
        camera_params[0],
        camera_params[1],
        camera_params[2] - x1,
        camera_params[3] - y1,
    ]
    detections = detector.detect(
        roi_gray,
        estimate_tag_pose=True,
        camera_params=roi_params,
        tag_size=tag_size,
    )
    tags = []
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


def bind_tags_to_tracks(
    tags: List[Dict[str, Any]],
    tracks: List[LocalTrack],
    timestamp: float,
    camera: CameraConfig,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for tag in tags:
        center = tag["center"]
        candidates = [track for track in tracks if bbox_contains(track.bbox, center)]
        if candidates:
            track = min(
                candidates,
                key=lambda item: math.dist(
                    ((item.bbox[0] + item.bbox[2]) * 0.5, (item.bbox[1] + item.bbox[3]) * 0.35),
                    center,
                ),
            )
        else:
            x, y = center
            size = 120.0
            track = LocalTrack(
                local_track_id=-1,
                bbox=(x - size, y - size, x + size, y + size),
                is_real_person=False,
                last_seen_ts=timestamp,
                confidence=0.75,
            )

        track.tag_id = tag["tag_id"]
        track.last_tag_seen_ts = timestamp
        area, position = area_for_track(track, camera)
        track.area = area
        track.position = position
        track.confidence = max(track.confidence, 0.92)
        observations.append(local_observation(camera, track, timestamp, "CONFIRMED"))
    return observations


def state_for_track(track: LocalTrack, timestamp: float) -> Optional[str]:
    if track.tag_id is None or track.last_tag_seen_ts is None:
        return None
    age = timestamp - track.last_tag_seen_ts
    if age <= CONFIRMED_WINDOW_SEC:
        return "CONFIRMED"
    if age <= TRACKED_WINDOW_SEC:
        return "TRACKED"
    return None


def local_observation(
    camera: CameraConfig,
    track: LocalTrack,
    timestamp: float,
    state: str,
) -> Dict[str, Any]:
    assert track.tag_id is not None
    last_tag_seen_ms = 0
    if track.last_tag_seen_ts is not None:
        last_tag_seen_ms = int(max(0.0, timestamp - track.last_tag_seen_ts) * 1000)
    state_penalty = 1.0 if state == "CONFIRMED" else 0.62
    return {
        "timestamp": round(timestamp, 3),
        "camera_id": camera.camera_id,
        "local_track_id": track.local_track_id,
        "person_id": track.tag_id,
        "state": state,
        "area": track.area,
        "position": [round(track.position[0], 3), round(track.position[1], 3)],
        "bbox": [round(v, 1) for v in track.bbox],
        "confidence": round(min(0.99, track.confidence * state_penalty), 3),
        "last_tag_seen_ms": last_tag_seen_ms,
    }


def color_for_name(name: str) -> Tuple[int, int, int]:
    palette = [
        (70, 190, 255),
        (80, 220, 120),
        (255, 170, 70),
        (220, 120, 255),
        (255, 90, 90),
        (120, 220, 220),
        (180, 180, 255),
    ]
    return palette[abs(hash(name)) % len(palette)]


def color_for_person(person_id: Optional[int]) -> Tuple[int, int, int]:
    if person_id == 0:
        return (0, 0, 255)
    if person_id == 1:
        return (255, 0, 0)
    return (255, 255, 255)


def draw_polyline(cv2: Any, frame: Any, points: Sequence[Point], color: Tuple[int, int, int], thickness: int) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness, lineType=cv2.LINE_AA)


def draw_text(
    cv2: Any,
    frame: Any,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_debug_overlays(
    cv2: Any,
    frame: Any,
    state: VisualCameraState,
    timestamp: float,
    args: argparse.Namespace,
    toggles: Dict[str, bool],
) -> Any:
    camera = state.camera
    if toggles["polygons"]:
        for area in camera.areas:
            color = color_for_name(area.name)
            draw_polyline(cv2, frame, area.polygon, color, 2)
            if area.polygon:
                px, py = area.polygon[0]
                draw_text(cv2, frame, area.name, (int(px), int(py) - 6), color, 0.5)

    if toggles["door_rois"]:
        for name, polygon in camera.door_rois.items():
            draw_polyline(cv2, frame, polygon, (0, 255, 255), 3)
            if polygon:
                px, py = polygon[0]
                draw_text(cv2, frame, f"ROI {name}", (int(px), int(py) - 18), (0, 255, 255), 0.5)

    if toggles["tag_rois"]:
        for roi in state.last_tag_rois:
            x1, y1, x2, y2 = [int(v) for v in roi]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 80), 1, cv2.LINE_AA)

    for tag in state.last_tags:
        corners = tag.get("corners", [])
        if corners:
            draw_polyline(cv2, frame, [(float(x), float(y)) for x, y in corners], (0, 255, 0), 2)
        cx, cy = tag["center"]
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1, cv2.LINE_AA)
        dist = tag.get("distance_m")
        label = f"tag {tag['tag_id']}"
        if dist is not None:
            label += f" {dist:.2f}m"
        draw_text(cv2, frame, label, (int(cx) + 8, int(cy) - 8), (0, 255, 0), 0.55)

    for bbox, score in state.last_people:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 2, cv2.LINE_AA)
        draw_text(cv2, frame, f"YOLO person {score:.2f}", (x1, max(18, y1 - 8)), (255, 255, 255), 0.5)

    for track in state.tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        track_state = state_for_track(track, timestamp) or "UNKNOWN"
        color = color_for_person(track.tag_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        foot = (int((track.bbox[0] + track.bbox[2]) * 0.5), int(track.bbox[3]))
        cv2.circle(frame, foot, 6, (0, 0, 255), -1, cv2.LINE_AA)
        label_id = f"id:{track.tag_id}" if track.tag_id is not None else "id:?"
        source = "YOLO" if track.is_real_person else "TAG"
        label = f"t{track.local_track_id} {label_id} {track_state} {track.area} {source}"
        draw_text(cv2, frame, label, (x1, max(18, y1 - 8)), color, 0.55)

    yolo_status = "ON" if args.yolo_model else "OFF"
    draw_text(
        cv2,
        frame,
        f"YOLO:{yolo_status} people={len(state.last_people)} tracks={len(state.tracks)} tags={len(state.last_tags)}",
        (12, 54),
        (210, 255, 210) if args.yolo_model else (180, 180, 180),
        0.58,
    )
    draw_text(
        cv2,
        frame,
        f"{camera.camera_id}  t={timestamp:.2f}s  frame={state.frame_idx}",
        (12, 28),
        (255, 255, 255),
        0.65,
    )
    return frame


def resize_tile(cv2: Any, frame: Any, width: int, height: int) -> Any:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    tile[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return tile


def make_grid(cv2: Any, tiles: Sequence[Any], width: int, height: int) -> Any:
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    padded = list(tiles[:4])
    while len(padded) < 4:
        padded.append(blank.copy())
    top = cv2.hconcat([padded[0], padded[1]])
    bottom = cv2.hconcat([padded[2], padded[3]])
    return cv2.vconcat([top, bottom])


def draw_fused_summary(cv2: Any, grid: Any, fused_rows: Sequence[Dict[str, Any]]) -> None:
    x, y = 18, 34
    draw_text(cv2, grid, "FUSED", (x, y), (255, 255, 255), 0.65)
    for idx, row in enumerate(fused_rows[:8]):
        text = (
            f"p{row['person_id']} {row['state']} {row['area']} "
            f"conf={row['confidence']} cams={','.join(row['source_cameras'])}"
        )
        draw_text(cv2, grid, text, (x, y + 26 * (idx + 1)), (210, 255, 210), 0.55)


def process_camera(
    camera: CameraConfig,
    args: argparse.Namespace,
    yolo_model: Any,
) -> List[Dict[str, Any]]:
    video_path = camera.video_path
    if not os.path.isabs(video_path):
        video_path = os.path.join(args.base_dir, video_path)
    if not os.path.exists(video_path):
        print(f"{camera.camera_id}: missing video, skipped: {video_path}")
        return []

    cv2 = load_optional_cv2()
    Detector = load_optional_apriltag_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{camera.camera_id}: could not open video, skipped: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or args.default_fps
    detector = Detector(
        families=args.tag_family,
        nthreads=args.apriltag_threads,
        quad_decimate=args.quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    observations: List[Dict[str, Any]] = []
    tracks: List[LocalTrack] = []
    next_track_id = 1
    frame_idx = -1
    next_output_sec = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.frame_step != 0:
            continue
        timestamp = frame_idx / fps
        frame_h, frame_w = frame.shape[:2]

        if yolo_model is not None and frame_idx % max(1, int(fps / args.yolo_fps)) == 0:
            people = detect_people_yolo(yolo_model, frame, args.yolo_conf)
            tracks, next_track_id = update_tracks(
                tracks,
                people,
                timestamp,
                next_track_id,
                args.track_iou,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tag_rois: List[BBox]
        if tracks:
            tag_rois = [upper_body_roi(track.bbox, frame_w, frame_h) for track in tracks]
        else:
            tag_rois = [(0.0, 0.0, float(frame_w), float(frame_h))]

        if frame_idx % max(1, int(fps / args.tag_fps)) == 0:
            tags: List[Dict[str, Any]] = []
            for roi in tag_rois:
                tags.extend(
                    detect_tags_in_roi(
                        detector,
                        gray,
                        roi,
                        camera.camera_params,
                        camera.tag_size,
                    )
                )
            observations.extend(bind_tags_to_tracks(tags, tracks, timestamp, camera))

        current_sec = int(timestamp)
        if current_sec >= next_output_sec:
            for track in tracks:
                state = state_for_track(track, timestamp)
                if state:
                    area, position = area_for_track(track, camera)
                    track.area = area
                    track.position = position
                    observations.append(local_observation(camera, track, timestamp, state))
            next_output_sec = current_sec + 1

    cap.release()
    print(f"{camera.camera_id}: {len(observations)} local observations")
    return observations


def open_visual_states(
    cameras: Sequence[CameraConfig],
    args: argparse.Namespace,
) -> List[VisualCameraState]:
    cv2 = load_optional_cv2()
    Detector = load_optional_apriltag_detector()
    states: List[VisualCameraState] = []
    for camera in cameras:
        video_path = camera.video_path
        if not os.path.isabs(video_path):
            video_path = os.path.join(args.base_dir, video_path)
        if not os.path.exists(video_path):
            print(f"{camera.camera_id}: missing video, skipped: {video_path}")
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{camera.camera_id}: could not open video, skipped: {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or args.default_fps
        detector = Detector(
            families=args.tag_family,
            nthreads=args.apriltag_threads,
            quad_decimate=args.quad_decimate,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        states.append(VisualCameraState(camera=camera, cap=cap, fps=fps, detector=detector))
    return states


def process_visual_frame(
    cv2: Any,
    state: VisualCameraState,
    args: argparse.Namespace,
    yolo_model: Any,
) -> Optional[List[Dict[str, Any]]]:
    ok, frame = state.cap.read()
    if not ok:
        state.cap.release()
        state.last_frame = None
        return None
    state.frame_idx += 1
    state.last_frame = frame
    if state.frame_idx % args.frame_step != 0:
        return []

    timestamp = state.frame_idx / state.fps
    frame_h, frame_w = frame.shape[:2]
    observations: List[Dict[str, Any]] = []

    if yolo_model is not None and state.frame_idx % max(1, int(state.fps / args.yolo_fps)) == 0:
        people = detect_people_yolo(yolo_model, frame, args.yolo_conf)
        state.last_people = people
        state.tracks, state.next_track_id = update_tracks(
            state.tracks,
            people,
            timestamp,
            state.next_track_id,
            args.track_iou,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if state.tracks:
        tag_rois = [upper_body_roi(track.bbox, frame_w, frame_h) for track in state.tracks]
    else:
        tag_rois = [(0.0, 0.0, float(frame_w), float(frame_h))]
    state.last_tag_rois = tag_rois

    if state.frame_idx % max(1, int(state.fps / args.tag_fps)) == 0:
        tags: List[Dict[str, Any]] = []
        for roi in tag_rois:
            tags.extend(
                detect_tags_in_roi(
                    state.detector,
                    gray,
                    roi,
                    state.camera.camera_params,
                    state.camera.tag_size,
                )
            )
        state.last_tags = tags
        new_observations = bind_tags_to_tracks(tags, state.tracks, timestamp, state.camera)
        observations.extend(new_observations)

    current_sec = int(timestamp)
    if current_sec >= state.next_output_sec:
        for track in state.tracks:
            track_state = state_for_track(track, timestamp)
            if track_state:
                area, position = area_for_track(track, state.camera)
                track.area = area
                track.position = position
                observations.append(local_observation(state.camera, track, timestamp, track_state))
        state.next_output_sec = current_sec + 1

    state.observations.extend(observations)
    return observations


def run_visualization(
    cameras: Sequence[CameraConfig],
    config: Dict[str, Any],
    args: argparse.Namespace,
    yolo_model: Any,
) -> int:
    cv2 = load_optional_cv2()
    states = open_visual_states(cameras, args)
    if not states:
        print("No videos could be opened for visualization.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_output_dir, exist_ok=True)
    window_name = "Multi-camera AprilTag Debug"
    paused = False
    step_once = False
    screenshot_idx = 0
    toggles = {
        "polygons": args.show_polygons,
        "door_rois": args.show_door_rois,
        "tag_rois": args.show_tag_rois,
        "fused": args.show_fused,
    }
    all_observations: List[Dict[str, Any]] = []
    latest_fused: List[Dict[str, Any]] = []

    print("Controls: Space pause/play | N step | Q/Esc quit | P polygons | D doors | T tag ROIs | F fused | S screenshot")
    try:
        while states:
            advance = (not paused) or step_once
            if advance:
                still_active: List[VisualCameraState] = []
                for state in states:
                    new_obs = process_visual_frame(cv2, state, args, yolo_model)
                    if new_obs is not None:
                        all_observations.extend(new_obs)
                        still_active.append(state)
                states = still_active
                step_once = False

            current_second = 0
            for state in states:
                if state.frame_idx >= 0:
                    current_second = max(current_second, int(state.frame_idx / state.fps))
            if all_observations:
                fused = fuse_observations(all_observations, config, args.prediction_window_sec)
                latest_fused = [row for row in fused if int(row["timestamp"]) == current_second]

            tiles = []
            for state in states[:4]:
                frame = state.last_frame.copy() if state.last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                timestamp = max(0.0, state.frame_idx / state.fps) if state.frame_idx >= 0 else 0.0
                draw_debug_overlays(cv2, frame, state, timestamp, args, toggles)
                tiles.append(resize_tile(cv2, frame, args.vis_width, args.vis_height))
            grid = make_grid(cv2, tiles, args.vis_width, args.vis_height)
            if toggles["fused"]:
                draw_fused_summary(cv2, grid, latest_fused)
            if paused:
                draw_text(cv2, grid, "PAUSED", (grid.shape[1] - 130, 34), (0, 255, 255), 0.75)

            cv2.imshow(window_name, grid)
            key = cv2.waitKey(1 if not paused else 40) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("n"), ord("N")):
                paused = True
                step_once = True
            elif key in (ord("p"), ord("P")):
                toggles["polygons"] = not toggles["polygons"]
            elif key in (ord("d"), ord("D")):
                toggles["door_rois"] = not toggles["door_rois"]
            elif key in (ord("t"), ord("T")):
                toggles["tag_rois"] = not toggles["tag_rois"]
            elif key in (ord("f"), ord("F")):
                toggles["fused"] = not toggles["fused"]
            elif key in (ord("s"), ord("S")):
                screenshot_idx += 1
                path = os.path.join(args.debug_output_dir, f"debug_{screenshot_idx:04d}.jpg")
                cv2.imwrite(path, grid)
                print(f"Saved screenshot: {path}")
    finally:
        for state in states:
            state.cap.release()
        cv2.destroyAllWindows()

    fused = fuse_observations(all_observations, config, args.prediction_window_sec)
    local_path = os.path.join(args.output_dir, "local_observations.jsonl")
    fused_jsonl = os.path.join(args.output_dir, "fused_positions.jsonl")
    fused_csv = os.path.join(args.output_dir, "fused_positions.csv")
    write_jsonl(local_path, all_observations)
    write_jsonl(fused_jsonl, fused)
    write_csv(fused_csv, fused)
    print(f"Wrote {len(all_observations)} visual local observations: {local_path}")
    print(f"Wrote {len(fused)} visual fused positions: {fused_jsonl}")
    return 0


def state_rank(state: str) -> int:
    return {
        "CONFIRMED": 4,
        "TRACKED": 3,
        "PREDICTED": 2,
        "CANDIDATE": 1,
        "LOST": 0,
    }.get(state, 0)


def add_topology_hints(row: Dict[str, Any], topology: Dict[str, Any]) -> Dict[str, Any]:
    doors = topology.get("doors", {})
    area = row.get("area")
    if area in doors:
        row["candidate_exits"] = doors[area].get("connects", [])
    else:
        row["candidate_exits"] = []
    return row


def fill_predicted_gaps(
    fused: List[Dict[str, Any]],
    topology: Dict[str, Any],
    prediction_window_sec: int,
) -> List[Dict[str, Any]]:
    if not fused:
        return []
    by_person: Dict[int, List[Dict[str, Any]]] = {}
    for row in fused:
        by_person.setdefault(int(row["person_id"]), []).append(row)

    completed: List[Dict[str, Any]] = []
    for person_id, rows in by_person.items():
        rows = sorted(rows, key=lambda item: int(item["timestamp"]))
        for idx, row in enumerate(rows):
            completed.append(add_topology_hints(row, topology))
            if idx + 1 >= len(rows):
                continue
            current_ts = int(row["timestamp"])
            next_ts = int(rows[idx + 1]["timestamp"])
            gap_end = min(next_ts, current_ts + prediction_window_sec + 1)
            for ts in range(current_ts + 1, gap_end):
                if ts >= next_ts:
                    break
                age = ts - current_ts
                confidence = max(0.05, float(row["confidence"]) * (0.72 ** age))
                predicted = {
                    "timestamp": ts,
                    "person_id": person_id,
                    "state": "PREDICTED",
                    "area": row["area"],
                    "position": row["position"],
                    "confidence": round(confidence, 3),
                    "source_cameras": row["source_cameras"],
                    "last_tag_seen_ms": int(row["last_tag_seen_ms"]) + age * 1000,
                }
                completed.append(add_topology_hints(predicted, topology))
    return sorted(completed, key=lambda item: (int(item["timestamp"]), int(item["person_id"])))


def fuse_observations(
    observations: Iterable[Dict[str, Any]],
    topology: Dict[str, Any],
    prediction_window_sec: int,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for obs in observations:
        second = int(float(obs["timestamp"]))
        key = (second, int(obs["person_id"]))
        buckets.setdefault(key, []).append(obs)

    fused: List[Dict[str, Any]] = []
    for (second, person_id), items in sorted(buckets.items()):
        weights = [max(0.05, float(item["confidence"])) for item in items]
        total = sum(weights)
        x = sum(item["position"][0] * weight for item, weight in zip(items, weights)) / total
        y = sum(item["position"][1] * weight for item, weight in zip(items, weights)) / total
        best = max(items, key=lambda item: (state_rank(item["state"]), item["confidence"]))
        cameras = sorted({item["camera_id"] for item in items})
        last_tag_seen_ms = min(int(item["last_tag_seen_ms"]) for item in items)
        confidence = min(0.99, sum(weights) / max(1.0, len(items)) + 0.05 * (len(cameras) - 1))
        fused.append(
            add_topology_hints(
                {
                    "timestamp": second,
                    "person_id": person_id,
                    "state": best["state"],
                    "area": best["area"],
                    "position": [round(x, 3), round(y, 3)],
                    "confidence": round(confidence, 3),
                    "source_cameras": cameras,
                    "last_tag_seen_ms": last_tag_seen_ms,
                },
                topology,
            )
        )
    return fill_predicted_gaps(fused, topology, prediction_window_sec)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestamp",
        "person_id",
        "state",
        "area",
        "position",
        "confidence",
        "source_cameras",
        "candidate_exits",
        "last_tag_seen_ms",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["position"] = json.dumps(out["position"])
            out["source_cameras"] = json.dumps(out["source_cameras"])
            out["candidate_exits"] = json.dumps(out.get("candidate_exits", []))
            writer.writerow(out)


def validate_config(config: Dict[str, Any]) -> None:
    cameras = config.get("cameras", [])
    if not cameras:
        raise ValueError("Config must contain at least one camera.")
    ids = [camera["camera_id"] for camera in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("camera_id values must be unique.")
    for raw_camera in cameras:
        if not raw_camera.get("areas"):
            raise ValueError(f"{raw_camera['camera_id']} must define at least one area.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline multi-camera AprilTag validation pipeline."
    )
    parser.add_argument("--config", default="scenario_config.json")
    parser.add_argument("--base-dir", default=os.getcwd())
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-fps", type=float, default=2.0)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--tag-fps", type=float, default=2.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--default-fps", type=float, default=30.0)
    parser.add_argument("--track-iou", type=float, default=0.25)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--quad-decimate", type=float, default=2.0)
    parser.add_argument("--apriltag-threads", type=int, default=4)
    parser.add_argument("--prediction-window-sec", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-width", type=int, default=960)
    parser.add_argument("--vis-height", type=int, default=540)
    parser.add_argument("--show-polygons", action="store_true", default=True)
    parser.add_argument("--show-door-rois", action="store_true", default=True)
    parser.add_argument("--show-tag-rois", action="store_true", default=True)
    parser.add_argument("--show-fused", action="store_true", default=True)
    parser.add_argument("--debug-output-dir", default="outputs_debug")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    cameras = [camera_from_dict(item) for item in config["cameras"]]

    print(f"Loaded {len(cameras)} cameras from {args.config}")
    if args.dry_run:
        for camera in cameras:
            print(f"{camera.camera_id}: {camera.video_path}, {len(camera.areas)} areas")
        return 0

    if args.visualize and args.output_dir == "outputs":
        args.output_dir = "outputs_visual"

    os.makedirs(args.output_dir, exist_ok=True)
    yolo_model = load_optional_yolo(args.yolo_model)

    if args.visualize:
        return run_visualization(cameras, config, args, yolo_model)

    all_observations: List[Dict[str, Any]] = []
    for camera in cameras:
        all_observations.extend(process_camera(camera, args, yolo_model))

    fused = fuse_observations(all_observations, config, args.prediction_window_sec)
    local_path = os.path.join(args.output_dir, "local_observations.jsonl")
    fused_jsonl = os.path.join(args.output_dir, "fused_positions.jsonl")
    fused_csv = os.path.join(args.output_dir, "fused_positions.csv")
    write_jsonl(local_path, all_observations)
    write_jsonl(fused_jsonl, fused)
    write_csv(fused_csv, fused)
    print(f"Wrote {len(all_observations)} local observations: {local_path}")
    print(f"Wrote {len(fused)} fused positions: {fused_jsonl}")
    print(f"Wrote fused CSV: {fused_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
