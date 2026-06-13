import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import single_camera_homography_viewer as single_view
import single_camera_tag_tracker as tracker


Point = Tuple[float, float]


@dataclass
class CameraState:
    camera: Dict[str, Any]
    cap: Any
    fps: float
    frame_w: int
    frame_h: int
    detector: Any
    tag_size: float
    yolo_model: Any
    image_to_map: Optional[np.ndarray]
    frame_idx: int
    track_to_person: Dict[int, int] = field(default_factory=dict)
    last_frame: Optional[Any] = None
    last_tracks: List[Dict[str, Any]] = field(default_factory=list)
    last_tags: List[Dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Multi-camera homography viewer with fused p0/p1 map positions."
    )
    parser.add_argument("--config", default=str(script_dir / "scenario_config.json"))
    parser.add_argument("--base-dir", default=str(script_dir))
    parser.add_argument("--camera-ids", default="cam_a,cam_b,cam_c")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=None)
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--apriltag-threads", type=int, default=4)
    parser.add_argument("--tag-fps", type=float, default=5.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-tile-width", type=int, default=640)
    parser.add_argument("--camera-tile-height", type=int, default=360)
    parser.add_argument("--map-scale", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(script_dir / "outputs_multi_camera_homography"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    return parser.parse_args()


def selected_cameras(config: Dict[str, Any], camera_ids: str) -> List[Dict[str, Any]]:
    wanted = [item.strip() for item in camera_ids.split(",") if item.strip()]
    cameras = []
    for camera_id in wanted:
        cameras.append(tracker.find_camera(config, camera_id))
    return cameras


def open_camera_state(
    cv2: Any,
    Detector: Any,
    camera: Dict[str, Any],
    base_dir: Path,
    args: argparse.Namespace,
) -> CameraState:
    video_path = tracker.resolve_path(camera["video_path"], base_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"{camera['camera_id']}: video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"{camera['camera_id']}: could not open video: {video_path}")

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
    return CameraState(
        camera=camera,
        cap=cap,
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
        detector=detector,
        tag_size=tag_size,
        yolo_model=tracker.load_yolo(args.yolo_model),
        image_to_map=tracker.homography_matrix(camera),
        frame_idx=start_frame - 1,
    )


def process_camera_frame(
    cv2: Any,
    state: CameraState,
    frame: Any,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    timestamp = state.frame_idx / state.fps
    results = state.yolo_model.track(
        frame,
        classes=[0],
        conf=args.yolo_conf,
        persist=True,
        tracker=args.tracker,
        verbose=False,
    )
    tracks = tracker.parse_yolo_tracks(results)

    tags: List[Dict[str, Any]] = []
    tag_interval = max(1, int(round(state.fps / max(args.tag_fps, 1e-6))))
    if state.frame_idx % tag_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rois = [tracker.upper_body_roi(track["bbox"], state.frame_w, state.frame_h) for track in tracks]
        if not rois:
            rois = [(0.0, 0.0, float(state.frame_w), float(state.frame_h))]
        for roi in rois:
            tags.extend(
                tracker.detect_tags_in_roi(
                    state.detector,
                    gray,
                    roi,
                    state.camera["camera_params"],
                    state.tag_size,
                )
            )

    tracker.bind_tags_to_tracks(tags, tracks)
    single_view.update_person_memory(tracks, state.track_to_person)
    for track in tracks:
        track["map_point"] = tracker.project_point(track["foot"], state.image_to_map)

    state.last_frame = frame
    state.last_tracks = tracks
    state.last_tags = tags

    observations: List[Dict[str, Any]] = []
    for track in tracks:
        person_id = state.track_to_person.get(int(track["track_id"]))
        if person_id not in (0, 1) or track.get("map_point") is None:
            continue
        observations.append(
            {
                "camera_id": state.camera["camera_id"],
                "person_id": person_id,
                "track_id": int(track["track_id"]),
                "map_point": track["map_point"],
                "confidence": float(track.get("confidence", 0.0)),
                "timestamp": timestamp,
            }
        )
    return observations


def draw_camera_tile(
    cv2: Any,
    state: CameraState,
    config: Dict[str, Any],
    width: int,
    height: int,
) -> Any:
    frame = state.last_frame.copy()
    tracker.draw_scene_guides(cv2, frame, config, state.camera, True, True)
    tracker.draw_tags(cv2, frame, state.last_tags)
    single_view.draw_camera_people(cv2, frame, state.last_tracks, state.track_to_person)
    tracker.draw_text(
        cv2,
        frame,
        f"{state.camera['camera_id']} frame:{state.frame_idx} p0/p1 only",
        (16, 32),
        (255, 255, 255),
        0.7,
    )
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def black_tile(cv2: Any, width: int, height: int, text: str) -> Any:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    tracker.draw_text(cv2, tile, text, (18, 34), (220, 220, 220), 0.65)
    return tile


def compose_camera_grid(cv2: Any, states: Sequence[CameraState], config: Dict[str, Any], tile_w: int, tile_h: int) -> Any:
    tiles = []
    for state in states:
        if state.last_frame is None:
            tiles.append(black_tile(cv2, tile_w, tile_h, state.camera["camera_id"]))
        else:
            tiles.append(draw_camera_tile(cv2, state, config, tile_w, tile_h))
    while len(tiles) < 4:
        tiles.append(black_tile(cv2, tile_w, tile_h, ""))
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def fuse_observations(observations: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    fused: Dict[int, Dict[str, Any]] = {}
    for person_id in (0, 1):
        items = [item for item in observations if item["person_id"] == person_id]
        if not items:
            continue
        weights = [max(0.05, float(item.get("confidence", 0.0))) for item in items]
        total = sum(weights)
        x = sum(item["map_point"][0] * weight for item, weight in zip(items, weights)) / total
        y = sum(item["map_point"][1] * weight for item, weight in zip(items, weights)) / total
        fused[person_id] = {
            "map_point": (x, y),
            "source_cameras": [item["camera_id"] for item in items],
            "count": len(items),
        }
    return fused


def draw_multi_camera_floorplan(
    cv2: Any,
    floorplan_base: Any,
    config: Dict[str, Any],
    states: Sequence[CameraState],
    observations: Sequence[Dict[str, Any]],
    fused: Dict[int, Dict[str, Any]],
) -> Any:
    floorplan = floorplan_base.copy()
    calibration = config.get("map", {}).get("calibration", {})
    world_to_pixel = single_view.matrix_from_config(calibration["world_to_pixel_homography"]) if calibration.get("world_to_pixel_homography") else None

    for region in config.get("map_regions", []):
        single_view.draw_map_polyline(cv2, floorplan, region.get("polygon", []), world_to_pixel, (80, 220, 120), 3)

    for state in states:
        visibility = state.camera.get("map_visibility", {}).get("polygon", [])
        single_view.draw_map_polyline(cv2, floorplan, visibility, world_to_pixel, (255, 180, 40), 4)
        camera_pixel = single_view.apply_homography((float(state.camera["position"][0]), float(state.camera["position"][1])), world_to_pixel)
        if camera_pixel is not None:
            cx, cy = int(round(camera_pixel[0])), int(round(camera_pixel[1]))
            cv2.circle(floorplan, (cx, cy), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(floorplan, (cx, cy), 14, (0, 0, 0), 2, cv2.LINE_AA)
            tracker.draw_text(cv2, floorplan, state.camera["camera_id"], (cx + 14, cy - 8), (0, 255, 255), 0.65)

    for landmark_id, coord in config.get("landmarks", {}).items():
        pixel = single_view.apply_homography((float(coord[0]), float(coord[1])), world_to_pixel)
        if pixel is None:
            continue
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        cv2.circle(floorplan, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)
        tracker.draw_text(cv2, floorplan, str(landmark_id), (x + 8, y - 8), (0, 255, 255), 0.5)

    for obs in observations:
        pixel = single_view.apply_homography(obs["map_point"], world_to_pixel)
        if pixel is None:
            continue
        color = single_view.color_for_person(int(obs["person_id"]))
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        cv2.circle(floorplan, (x, y), 10, color, 2, cv2.LINE_AA)
        tracker.draw_text(cv2, floorplan, obs["camera_id"], (x + 12, y + 18), color, 0.45)

    for person_id, item in fused.items():
        pixel = single_view.apply_homography(item["map_point"], world_to_pixel)
        if pixel is None:
            continue
        color = single_view.color_for_person(person_id)
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        cv2.circle(floorplan, (x, y), 22, color, -1, cv2.LINE_AA)
        cv2.circle(floorplan, (x, y), 28, (0, 0, 0), 3, cv2.LINE_AA)
        label = f"FUSED p{person_id} ({item['map_point'][0]:.2f},{item['map_point'][1]:.2f}) cams:{','.join(item['source_cameras'])}"
        tracker.draw_text(cv2, floorplan, label, (x + 30, y - 10), color, 0.7)

    tracker.draw_text(cv2, floorplan, "multi-camera fused map: p0/p1", (20, 36), (255, 255, 255), 0.85)
    return floorplan


def fit_on_canvas(cv2: Any, image: Any, canvas_w: int, canvas_h: int, title: str = "") -> Any:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x = (canvas_w - out_w) // 2
    y = (canvas_h - out_h) // 2
    canvas[y:y + out_h, x:x + out_w] = resized
    if title:
        tracker.draw_text(cv2, canvas, title, (18, 32), (255, 255, 255), 0.7)
    return canvas


def main() -> int:
    args = parse_args()
    cv2 = tracker.load_cv2()
    Detector = tracker.load_apriltag_detector()
    config_path = Path(args.config)
    base_dir = Path(args.base_dir)
    config = tracker.load_config(config_path)
    cameras = selected_cameras(config, args.camera_ids)

    map_path = tracker.resolve_path(config.get("map", {}).get("image_path", "floorplanB.png"), base_dir)
    floorplan_base = cv2.imread(str(map_path))
    if floorplan_base is None:
        raise FileNotFoundError(f"Could not load floorplan image: {map_path}")

    states = [open_camera_state(cv2, Detector, camera, base_dir, args) for camera in cameras]
    if not states:
        raise ValueError("No cameras selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_w = args.camera_tile_width * 2
    grid_h = args.camera_tile_height * 2
    map_canvas_w = max(1, int(round(grid_w * args.map_scale)))
    map_canvas_h = max(1, int(round(grid_h * args.map_scale)))
    view_w = grid_w + map_canvas_w
    view_h = grid_h

    writer = None
    if args.save_video:
        out_path = output_dir / "multi_camera_homography_fused.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, states[0].fps, (view_w, view_h))

    print(f"Cameras: {', '.join(state.camera['camera_id'] for state in states)}")
    print(f"Map: {map_path}")
    print(f"Map canvas scale: {args.map_scale:.2f}")
    print("Fusing and displaying only p0 and p1.")

    processed = 0
    while True:
        frames = []
        ok_all = True
        for state in states:
            ok, frame = state.cap.read()
            if not ok:
                ok_all = False
                break
            state.frame_idx += 1
            frames.append((state, frame))
        if not ok_all:
            break

        if any(state.frame_idx % max(1, args.frame_step) != 0 for state, _ in frames):
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break
        processed += 1

        observations: List[Dict[str, Any]] = []
        for state, frame in frames:
            observations.extend(process_camera_frame(cv2, state, frame, config, args))
        fused = fuse_observations(observations)

        grid = compose_camera_grid(cv2, states, config, args.camera_tile_width, args.camera_tile_height)
        floorplan = draw_multi_camera_floorplan(cv2, floorplan_base, config, states, observations, fused)
        map_canvas = fit_on_canvas(cv2, floorplan, map_canvas_w, map_canvas_h, "map 50%")
        right_panel = np.zeros((view_h, map_canvas_w, 3), dtype=np.uint8)
        right_panel[:map_canvas_h, :] = map_canvas
        view = np.hstack([grid, right_panel])

        if writer is not None:
            writer.write(view)
        if args.show:
            cv2.imshow("multi camera homography viewer", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    for state in states:
        state.cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Processed fused frames: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
