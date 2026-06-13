import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import single_camera_tag_tracker as tracker


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


PERSON_COLORS = {
    0: (0, 0, 255),
    1: (255, 0, 0),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Single-camera homography viewer with camera frame on the left and floorplan map on the right."
    )
    parser.add_argument("--config", default=str(script_dir / "scenario_config.json"))
    parser.add_argument("--camera-id", default="cam_a")
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--video", default=None)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=None)
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--apriltag-threads", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tag-fps", type=float, default=5.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--view-height", type=int, default=720)
    parser.add_argument("--output-dir", default=str(script_dir / "outputs_single_camera_homography"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    return parser.parse_args()


def matrix_from_config(raw: Sequence[Sequence[float]]) -> np.ndarray:
    return np.array(raw, dtype=np.float64)


def apply_homography(point: Point, matrix: Optional[np.ndarray]) -> Optional[Point]:
    if matrix is None:
        return None
    vec = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    projected = matrix @ vec
    denom = float(projected[2])
    if abs(denom) < 1e-9:
        return None
    return (float(projected[0] / denom), float(projected[1] / denom))


def color_for_person(person_id: int) -> Tuple[int, int, int]:
    return PERSON_COLORS.get(person_id, (255, 255, 255))


def resize_to_height(cv2: Any, image: Any, height: int) -> Any:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * height / max(1, h))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_map_polyline(
    cv2: Any,
    image: Any,
    points: Sequence[Sequence[float]],
    world_to_pixel: Optional[np.ndarray],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    pixels: List[Point] = []
    for point in points:
        pixel = apply_homography((float(point[0]), float(point[1])), world_to_pixel)
        if pixel is not None:
            pixels.append(pixel)
    tracker.draw_polyline(cv2, image, pixels, color, thickness)


def draw_floorplan_base(cv2: Any, floorplan_base: Any, config: Dict[str, Any], camera: Dict[str, Any]) -> Any:
    floorplan = floorplan_base.copy()
    calibration = config.get("map", {}).get("calibration", {})
    world_to_pixel = matrix_from_config(calibration["world_to_pixel_homography"]) if calibration.get("world_to_pixel_homography") else None

    for region in config.get("map_regions", []):
        draw_map_polyline(cv2, floorplan, region.get("polygon", []), world_to_pixel, (80, 220, 120), 3)
        polygon = region.get("polygon", [])
        if polygon:
            label_pixel = apply_homography((float(polygon[0][0]), float(polygon[0][1])), world_to_pixel)
            if label_pixel is not None:
                tracker.draw_text(cv2, floorplan, region.get("name", "region"), (int(label_pixel[0]) + 8, int(label_pixel[1]) - 8), (80, 220, 120), 0.65)

    visibility = camera.get("map_visibility", {}).get("polygon", [])
    draw_map_polyline(cv2, floorplan, visibility, world_to_pixel, (255, 180, 40), 5)

    for landmark_id, coord in config.get("landmarks", {}).items():
        pixel = apply_homography((float(coord[0]), float(coord[1])), world_to_pixel)
        if pixel is None:
            continue
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        cv2.circle(floorplan, (x, y), 7, (0, 255, 255), -1, cv2.LINE_AA)
        tracker.draw_text(cv2, floorplan, str(landmark_id), (x + 8, y - 8), (0, 255, 255), 0.55)

    camera_pixel = apply_homography((float(camera["position"][0]), float(camera["position"][1])), world_to_pixel)
    if camera_pixel is not None:
        x, y = int(round(camera_pixel[0])), int(round(camera_pixel[1]))
        cv2.circle(floorplan, (x, y), 10, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(floorplan, (x, y), 14, (0, 0, 0), 2, cv2.LINE_AA)
        tracker.draw_text(cv2, floorplan, camera["camera_id"], (x + 14, y - 8), (0, 255, 255), 0.65)

    tracker.draw_text(cv2, floorplan, "floorplanB homography", (20, 36), (255, 255, 255), 0.85)
    return floorplan


def update_person_memory(
    tracks: Sequence[Dict[str, Any]],
    track_to_person: Dict[int, int],
) -> None:
    active_track_ids = {int(track["track_id"]) for track in tracks}
    for track_id in list(track_to_person):
        if track_id not in active_track_ids:
            del track_to_person[track_id]

    for track in tracks:
        tag_id = track.get("tag_id")
        if tag_id in (0, 1):
            track_to_person[int(track["track_id"])] = int(tag_id)


def draw_camera_people(cv2: Any, frame: Any, tracks: Sequence[Dict[str, Any]], track_to_person: Dict[int, int]) -> None:
    for track in tracks:
        track_id = int(track["track_id"])
        person_id = track_to_person.get(track_id)
        if person_id not in (0, 1):
            continue

        color = color_for_person(person_id)
        x1, y1, x2, y2 = [int(round(v)) for v in track["bbox"]]
        foot = track["foot"]
        fx, fy = int(round(foot[0])), int(round(foot[1]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 8, color, -1, cv2.LINE_AA)
        label = f"p{person_id} t{track_id} foot:({foot[0]:.0f},{foot[1]:.0f})"
        if track.get("map_point") is not None:
            mp = track["map_point"]
            label += f" map:({mp[0]:.2f},{mp[1]:.2f})"
        tracker.draw_text(cv2, frame, label, (x1, max(22, y1 - 8)), color, 0.65)


def draw_map_people(
    cv2: Any,
    floorplan: Any,
    tracks: Sequence[Dict[str, Any]],
    track_to_person: Dict[int, int],
    world_to_pixel: Optional[np.ndarray],
) -> None:
    for track in tracks:
        person_id = track_to_person.get(int(track["track_id"]))
        if person_id not in (0, 1) or track.get("map_point") is None:
            continue
        pixel = apply_homography(track["map_point"], world_to_pixel)
        if pixel is None:
            continue
        color = color_for_person(person_id)
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        cv2.circle(floorplan, (x, y), 17, color, -1, cv2.LINE_AA)
        cv2.circle(floorplan, (x, y), 22, (0, 0, 0), 3, cv2.LINE_AA)
        label = f"p{person_id} ({track['map_point'][0]:.2f},{track['map_point'][1]:.2f})"
        tracker.draw_text(cv2, floorplan, label, (x + 24, y - 10), color, 0.7)


def main() -> int:
    args = parse_args()
    cv2 = tracker.load_cv2()
    Detector = tracker.load_apriltag_detector()
    config_path = Path(args.config)
    base_dir = Path(args.base_dir)
    config = tracker.load_config(config_path)
    camera = tracker.find_camera(config, args.camera_id)

    video_path = tracker.resolve_path(args.video or camera["video_path"], base_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    map_path = tracker.resolve_path(config.get("map", {}).get("image_path", "floorplanB.png"), base_dir)
    floorplan_base = cv2.imread(str(map_path))
    if floorplan_base is None:
        raise FileNotFoundError(f"Could not load floorplan image: {map_path}")

    yolo_model = tracker.load_yolo(args.yolo_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(args.start_sec * fps)))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
    image_to_map = tracker.homography_matrix(camera)
    calibration = config.get("map", {}).get("calibration", {})
    world_to_pixel = matrix_from_config(calibration["world_to_pixel_homography"]) if calibration.get("world_to_pixel_homography") else None
    tag_interval = max(1, int(round(fps / max(args.tag_fps, 1e-6))))
    track_to_person: Dict[int, int] = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video:
        left_w = max(1, int(round(frame_w * args.view_height / max(1, frame_h))))
        right_w = max(1, int(round(floorplan_base.shape[1] * args.view_height / max(1, floorplan_base.shape[0]))))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = output_dir / f"{args.camera_id}_homography_view.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (left_w + right_w, args.view_height))

    print(f"Camera: {args.camera_id}")
    print(f"Video: {video_path}")
    print(f"Map: {map_path}")
    print("Showing only p0 and p1 map localization.")

    processed = 0
    frame_idx = start_frame - 1
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

        results = yolo_model.track(
            frame,
            classes=[0],
            conf=args.yolo_conf,
            persist=True,
            tracker=args.tracker,
            verbose=False,
        )
        tracks = tracker.parse_yolo_tracks(results)

        tags: List[Dict[str, Any]] = []
        if frame_idx % tag_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rois = [tracker.upper_body_roi(track["bbox"], frame_w, frame_h) for track in tracks]
            if not rois:
                rois = [(0.0, 0.0, float(frame_w), float(frame_h))]
            for roi in rois:
                tags.extend(tracker.detect_tags_in_roi(detector, gray, roi, camera["camera_params"], tag_size))

        tracker.bind_tags_to_tracks(tags, tracks)
        update_person_memory(tracks, track_to_person)
        for track in tracks:
            track["map_point"] = tracker.project_point(track["foot"], image_to_map)

        tracker.draw_scene_guides(cv2, frame, config, camera, True, True)
        tracker.draw_tags(cv2, frame, tags)
        draw_camera_people(cv2, frame, tracks, track_to_person)
        tracker.draw_text(
            cv2,
            frame,
            f"{args.camera_id} frame:{frame_idx} time:{timestamp:.2f}s p0/p1 only",
            (16, 32),
            (255, 255, 255),
            0.75,
        )

        floorplan = draw_floorplan_base(cv2, floorplan_base, config, camera)
        draw_map_people(cv2, floorplan, tracks, track_to_person, world_to_pixel)

        left = resize_to_height(cv2, frame, args.view_height)
        right = resize_to_height(cv2, floorplan, args.view_height)
        view = np.hstack([left, right])

        if writer is not None:
            writer.write(view)
        if args.show:
            cv2.imshow("single camera homography viewer", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Processed frames: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
