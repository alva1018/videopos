# 0612 Three-Camera Positioning Test

This folder contains the current 0612 test scene for the offline multi-camera
person-position prototype. The scene uses three phone cameras, a new floorplan,
and numbered boundary landmarks for image-to-map localization.

## Files

- `scenario_config.json`
  - New scene configuration for `multi_camera_validate.py`.
  - Defines the floorplan calibration, map regions, landmarks, cameras,
    visible projection polygons, and homography input points.
- `check_landmark_homography.py`
  - Standalone checker for camera landmark pixel/map correspondences.
  - Fits one image-to-map homography per camera and reports reprojection error.
- `single_camera_tag_tracker.py`
  - Single-camera visual/debug tool for AprilTag recognition plus YOLO person
    tracking.
  - Draws person boxes, tag detections, projection ROI, landmark pixels, and
    red foot points.
  - Writes per-track JSONL rows with `bbox`, `center`, `foot`, optional
    `tag_id`, and optional homography-projected `map_point`.
- `single_camera_homography_viewer.py`
  - Single-camera homography viewer that reuses `single_camera_tag_tracker.py`.
  - Shows the camera frame on the left and `floorplanB.png` on the right.
  - Draws map landmarks, camera visibility polygon, and only `p0` / `p1`
    localization markers.
- `multi_camera_homography_viewer.py`
  - Multi-camera homography viewer that reuses `single_camera_tag_tracker.py`
    and `single_camera_homography_viewer.py`.
  - Opens `cam_a`, `cam_b`, and `cam_c`, runs independent YOLO trackers per
    camera, and fuses `p0` / `p1` map positions by weighted averaging.
  - Shows a 2x2 camera grid on the left and a 50% scaled floorplan map panel on
    the right.
- `multi_camera_homography_viewer_gpu.py`
  - Copy of the multi-camera viewer that explicitly runs YOLO on CUDA.
  - Defaults to `device=0`, FP16, and `imgsz=960`.
  - Intended for the RTX 4090 machine.
- `multi_camera_homography_viewer_gpu_fast.py`
  - Faster GPU viewer variant.
  - Runs YOLO/tracker every N frames, defaults tag detection to 1 FPS, caches
    the static floorplan layer, and uses smaller camera tiles.
  - Keeps the original GPU viewer unchanged.
- `floorplanB.png`
  - Current calibrated floorplan image.
  - Contains four blue calibration points.
- `floorplanB1.png`
  - Earlier floorplan image kept for reference.
- `cam_a.mp4`, `cam_b.mp4`, `cam_c.mp4`
  - Three active camera videos for this test.
- `cam_a.png`, `cam_b.png`, `cam_c.png`
  - Camera reference frames/screenshots used for marking landmark pixels.

The other raw videos in this folder are kept as source material but are not
currently referenced by `scenario_config.json`.

## Current Scene

Coordinate system:

- Units: meters.
- Origin: bottom-left of the calibrated floorplan rectangle.
- `x` increases to the right.
- `y` increases upward.

Floorplan calibration:

```json
{
  "image_size": [3493, 1874],
  "world_points": [[0, 0], [0, 8.91], [18.27, 8.91], [18.27, 0]],
  "pixel_points": [[136.5, 1739.5], [136.5, 166.5], [3355.5, 166.5], [3357.5, 1737.5]]
}
```

The config also stores the exact `world_to_pixel_homography` and
`pixel_to_world_homography` matrices. The current main visualizer still uses
the existing axis-aligned floorplan overlay helper, so the `image_bounds_px`
entry is compatible with `multi_camera_validate.py`; the stored matrices are
available for a later exact floorplan projection update.

## Cameras

Active cameras:

| Camera | Video | Frame size | Camera position |
| --- | --- | --- | --- |
| `cam_a` | `cam_a.mp4` | `1920x1080` | `(12.91, 0.74)` |
| `cam_b` | `cam_b.mp4` | `1920x1080` | `(9.95, 8.07)` |
| `cam_c` | `cam_c.mp4` | `1080x1920` | `(0.68, 3.19)` |

All videos were read as 30 FPS during inspection.

## Landmark Coordinates

The scene has 11 numbered positioning landmarks:

| ID | Map coordinate |
| --- | --- |
| `1` | `(11.27, 1.76)` |
| `2` | `(13.27, 1.76)` |
| `3` | `(13.27, 2.31)` |
| `4` | `(13.27, 4.36)` |
| `5` | `(13.27, 6.96)` |
| `6` | `(9.67, 1.76)` |
| `7` | `(9.67, 4.06)` |
| `8` | `(9.67, 5.36)` |
| `9` | `(9.67, 6.96)` |
| `10` | `(2.83, 1.76)` |
| `11` | `(2.83, 4.06)` |

Current global map region:

```text
Positioning_Area = 10 -> 2 -> 5 -> 9 -> 7 -> 11
```

This region is used as the coarse map area for homography-localized points.

## Camera Landmark Pixels

`cam_a` visible landmarks:

| ID | Pixel |
| --- | --- |
| `1` | `(632, 984)` |
| `3` | `(1349, 1031)` |
| `4` | `(1352, 596)` |
| `5` | `(1358, 402)` |
| `7` | `(637, 506)` |
| `8` | `(774, 423)` |
| `9` | `(879, 356)` |

`cam_a` projection area landmarks:

```text
1, 3, 5, 9, 7
```

`cam_b` visible landmarks:

| ID | Pixel |
| --- | --- |
| `1` | `(1236, 486)` |
| `2` | `(990, 443)` |
| `3` | `(952, 462)` |
| `4` | `(741, 567)` |
| `5` | `(153, 855)` |
| `6` | `(1581, 549)` |
| `7` | `(1568, 747)` |
| `8` | `(1536, 977)` |

`cam_b` projection area landmarks:

```text
6, 2, 5, 8
```

`cam_c` visible landmarks:

| ID | Pixel |
| --- | --- |
| `11` | `(183, 1514)` |
| `10` | `(1004, 1488)` |
| `7` | `(367, 811)` |
| `6` | `(659, 812)` |
| `1` | `(633, 764)` |
| `4` | `(368, 733)` |
| `2` | `(614, 733)` |

`cam_c` projection area landmarks:

```text
10, 2, 4, 11
```

Projection polygons should not be drawn from the raw landmark input order. The
current config stores `projection_landmark_ids`, and
`single_camera_tag_tracker.py` rebuilds the display polygon by sorting those
landmarks in map-coordinate convex-hull order, then mapping the ordered IDs back
to camera pixels. This prevents self-crossed ROIs such as `1 -> 3 -> 5 -> 7 ->
9` in `cam_a`.

## Validation Results

The new config was checked with:

```powershell
python multi_camera_validate.py --config 0612\scenario_config.json --base-dir 0612 --dry-run
```

Result:

```text
Loaded 3 cameras from 0612\scenario_config.json
cam_a: cam_a.mp4, 1 areas
cam_b: cam_b.mp4, 1 areas
cam_c: cam_c.mp4, 1 areas
```

The landmark checker was run with:

```powershell
python 0612\check_landmark_homography.py --config 0612\scenario_config.json
```

Current all-point homography reprojection results:

| Camera | Mean error | RMSE | Max error | Status |
| --- | ---: | ---: | ---: | --- |
| `cam_a` | `0.0692 m` | `0.0780 m` | `0.1378 m` | OK |
| `cam_b` | `0.0662 m` | `0.0823 m` | `0.1809 m` | OK |
| `cam_c` | `0.1234 m` | `0.1334 m` | `0.2162 m` | OK |

Default checker thresholds:

- all-point single-landmark error: `0.30 m`
- leave-one-out single-landmark error: `0.50 m`

No suspicious landmark exceeded the default thresholds in the current data.

The single-camera tag/person tracker was syntax-checked and smoke-tested on
`cam_a` for three sampled frames:

```powershell
python 0612\single_camera_tag_tracker.py --config 0612\scenario_config.json --base-dir 0612 --camera-id cam_a --yolo-model yolov8n.pt --max-frames 3 --frame-step 30 --tag-fps 2 --output-dir 0612\outputs_single_camera_tag_tracker_test
```

Result:

```text
Processed frames: 3
Wrote tracks: 0612\outputs_single_camera_tag_tracker_test\cam_a_tag_tracks.jsonl
```

Example output row:

```json
{"timestamp": 0.0, "frame_idx": 0, "camera_id": "cam_a", "track_id": 1, "tag_id": null, "bbox": [963.3, 110.3, 1042.4, 322.8], "center": [1002.9, 216.5], "foot": [1002.9, 322.8], "map_point": [10.356, 8.132], "confidence": 0.579}
```

## How To Run

From the project root:

```powershell
python multi_camera_validate.py --config 0612\scenario_config.json --base-dir 0612 --dry-run
```

Check camera landmark consistency:

```powershell
python 0612\check_landmark_homography.py --config 0612\scenario_config.json
```

Run AprilTag-only processing:

```powershell
python multi_camera_validate.py --config 0612\scenario_config.json --base-dir 0612 --tag-fps 2 --quad-decimate 1.0 --use-homography --output-dir 0612\outputs_qd1
```

Run YOLO + AprilTag processing:

```powershell
python multi_camera_validate.py --config 0612\scenario_config.json --base-dir 0612 --yolo-model yolov8n.pt --tag-fps 2 --quad-decimate 1.0 --use-homography --output-dir 0612\outputs_yolo_qd1
```

Run the visual debugger with floorplan overlay:

```powershell
python multi_camera_validate.py --config 0612\scenario_config.json --base-dir 0612 --visualize --show-floorplan --yolo-model yolov8n.pt --tag-fps 2 --quad-decimate 1.0 --output-dir 0612\outputs_visual --debug-output-dir 0612\outputs_debug
```

Run the single-camera tag/person/foot-point debugger:

```powershell
python 0612\single_camera_tag_tracker.py --config 0612\scenario_config.json --base-dir 0612 --camera-id cam_a --yolo-model yolov8n.pt --show
```

Save an annotated video instead of opening a live window:

```powershell
python 0612\single_camera_tag_tracker.py --config 0612\scenario_config.json --base-dir 0612 --camera-id cam_a --yolo-model yolov8n.pt --save-video --output-dir 0612\outputs_single_camera_tag_tracker
```

Run the single-camera homography viewer with the map on the right:

```powershell
python 0612\single_camera_homography_viewer.py --config 0612\scenario_config.json --base-dir 0612 --camera-id cam_a --yolo-model yolov8n.pt
```

From inside the `0612` folder:

```powershell
python .\single_camera_homography_viewer.py --config .\scenario_config.json --base-dir . --camera-id cam_a --yolo-model ..\yolov8n.pt
```

Run the multi-camera fused homography viewer:

```powershell
python .\multi_camera_homography_viewer.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt
```

Run a short non-visual smoke test:

```powershell
python .\multi_camera_homography_viewer.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --max-frames 2 --frame-step 30 --no-show
```

Run the GPU multi-camera viewer on RTX 4090:

```powershell
python .\multi_camera_homography_viewer_gpu.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 960
```

If it is still too slow, lower YOLO input size:

```powershell
python .\multi_camera_homography_viewer_gpu.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 640
```

GPU smoke test:

```powershell
python .\multi_camera_homography_viewer_gpu.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 960 --max-frames 3 --frame-step 30 --tag-fps 2 --no-show
```

Run the faster GPU viewer:

```powershell
python .\multi_camera_homography_viewer_gpu_fast.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 640 --no-camera-overlays
```

If more speed is needed, run YOLO less often and display less often:

```powershell
python .\multi_camera_homography_viewer_gpu_fast.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 640 --yolo-every 3 --display-every 2 --no-camera-overlays
```

If `python` is not available on PATH in this environment, use the bundled Python
shown by Codex workspace dependencies.

On this machine, the system Python 3.13 environment currently has CPU-only
PyTorch, so CUDA runs should use the bundled Codex Python unless CUDA PyTorch is
installed into the system environment:

```powershell
& "C:\Users\Desktop_Yang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\multi_camera_homography_viewer_gpu_fast.py --config .\scenario_config.json --base-dir . --camera-ids cam_a,cam_b,cam_c --yolo-model ..\yolov8n.pt --device 0 --imgsz 640 --no-camera-overlays
```

## Git Snapshot

The intended commit scope for this 0612 snapshot is:

- `README.md`, `scenario_config.json`, and all Python tools in this folder.
- `floorplanB.png`, because it contains the calibrated floorplan used by the
  config.

The videos, raw MOV files, model weights, output folders, `floorplanB1.png`,
and camera reference screenshots are intentionally not part of the snapshot.
After pushing, GitHub will show these committed files under the existing repo's
`0612` folder; this does not create a separate GitHub repository.

## Current Limitations

- The camera `areas` entries are fallback polygons only. The intended primary
  localization method for this scene is image-to-map homography using the
  detected person foot point.
- Door-specific ROIs and semantic room/door/corridor areas have not yet been
  created for this new scene.
- The floorplan exact homography matrices are stored in config, but the current
  floorplan drawing helper in `multi_camera_validate.py` uses its existing
  axis-aligned map bounds logic.
- No formal camera intrinsic calibration has been added; `camera_params` remain
  approximate defaults, matching the existing prototype style.
- The previously rolled-back head projection and lightweight color-histogram
  ReID experiment have not been reintroduced.

## Suggested Next Steps

1. Run the visual debugger and inspect whether YOLO foot points project to the
   correct floorplan positions.
2. If the floorplan overlay needs exact sub-pixel alignment, update
   `multi_camera_validate.py` to use `map.calibration.world_to_pixel_homography`
   for floorplan drawing.
3. Add semantic map regions if the new scene needs more than a single
   `Positioning_Area`.
4. Add door/transition ROIs only after the core homography localization looks
   stable.
5. Save a short annotated visual sample after the projection is visually
   confirmed.
