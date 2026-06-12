# Multi-camera AprilTag validation

This project contains a simple offline validation pipeline for a small
multi-camera scene:

- One room with three doors.
- Corridor coverage from two additional phones.
- AprilTag IDs are used as person IDs.
- Person positions are output once per second after camera-level observations
  are fused.

## Files

- `pos.py` keeps the original single-video AprilTag experiment.
- `multi_camera_validate.py` runs the four-camera offline validation pipeline.
- `scenario_config.json` defines camera videos, approximate camera parameters,
  editable floor-localization polygons, door ROIs, and coarse output positions.
- `requirements.txt` lists the runtime dependencies.

## Prepare videos

Record four 1080p wide-angle videos and place them in this folder:

```text
cam_room_a.mp4
cam_room_b.mp4
cam_corridor_near.mp4
cam_corridor_far.mp4
```

Use a visible or audible sync action at the start of each recording. The first
implementation assumes the videos are already roughly aligned; if they are not,
trim them or add time offsets in a later iteration.

## Install dependencies

```powershell
pip install -r requirements.txt
```

`ultralytics` is optional. If it is not installed or no YOLO model is passed,
the script still runs AprilTag-based observations and fusion. For person-box
tracking, install `ultralytics` and pass a model such as `yolov8n.pt`.

## Run

Validate the config without processing video:

```powershell
python multi_camera_validate.py --dry-run
```

Run AprilTag-only validation:

```powershell
python multi_camera_validate.py --config scenario_config.json
```

Run with YOLO person detection:

```powershell
python multi_camera_validate.py --config scenario_config.json --yolo-model yolov8n.pt
```

Run the four-camera visual debugger:

```powershell
python multi_camera_validate.py --config scenario_config.json --visualize --yolo-model yolov8n.pt --tag-fps 2 --quad-decimate 1.0
```

Run the single-camera `cam_room_a` homography debugger with a live floorplan:

```powershell
python multi_camera_validate.py --config scenario_config.json --visualize --camera-id cam_room_a --show-floorplan --yolo-model yolov8n.pt --tag-fps 2 --quad-decimate 1.0 --output-dir outputs_room_a_homography --debug-output-dir outputs_debug_room_a
```

Start visual or batch processing from a later timestamp by passing seconds:

```powershell
python multi_camera_validate.py --config scenario_config.json --visualize --camera-id cam_room_a --show-floorplan --yolo-model yolov8n.pt --start-sec 120
```

The floorplan map coordinate system uses meters with origin at the bottom-left
corner of the yellow map box: `x` increases to the right and `y` increases
upward. The OpenCV image pixel coordinate still has `y` increasing downward,
so the visualizer flips the map `y` axis only when drawing onto
`floorplan.png`. Homography localization is only accepted when the chosen
image ground point falls inside that camera's projection ROI and the projected
map point falls inside that camera's configured visible map polygon.

Visual debugger controls:

```text
Space  pause/play
N      step one frame while paused
Q/Esc  quit
P      toggle area polygons
D      toggle door ROIs
T      toggle AprilTag search ROIs
F      toggle fused summary
M      toggle floorplan
S      save a 2x2 debug screenshot to outputs_debug/
```

Outputs are written to:

```text
outputs/local_observations.jsonl
outputs/fused_positions.jsonl
outputs/fused_positions.csv
```

## Single-camera person tracking

Use `single_camera_person_tracker.py` to test stable person tracking on
`cam_room_a` without AprilTags, area assignment, or multi-camera fusion. It
uses YOLO person detection with an Ultralytics tracker, defaulting to
BoT-SORT:

```powershell
python single_camera_person_tracker.py --config scenario_config.json --camera-id cam_room_a --model yolov8n.pt --show
```

If `python` is not on PATH, run the same command with the Python executable
used for this project.

Compare the lightweight ByteTrack baseline:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --tracker bytetrack.yaml --show
```

After observing a stable ID, rerun with a target ID to highlight and export
only that person:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --target-track-id 1 --save-video
```

Enable the experimental OSNet ReID gallery to assign longer-lived `person_id`
values when a tracker ID disappears and later reappears:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --reid-backend torchreid-osnet --show
```

Highlight a long-term person ID instead of a raw tracker ID:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --reid-backend torchreid-osnet --target-person-id p1 --save-video
```

Save accepted ReID crops for inspection:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --reid-backend torchreid-osnet --save-crops --output-dir outputs_single_tracker_reid
```

Use GPU for ReID by passing `--reid-device cuda`. First confirm that the Python
environment running the script has CUDA-enabled PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

FastReID and TransReID are exposed as external-repo backends. Clone the official
repo, download a matching config and trained checkpoint, then point the script
at those files:

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --reid-backend fastreid --reid-device cuda --fastreid-repo third_party/fast-reid --fastreid-config third_party/fast-reid/configs/Market1501/bagtricks_R50.yml --fastreid-weights models/fastreid_market1501.pth --show
```

```powershell
python single_camera_person_tracker.py --video cam_room_a.mp4 --model yolov8n.pt --reid-backend transreid --reid-device cuda --transreid-repo third_party/TransReID --transreid-config third_party/TransReID/configs/Market/vit_transreid_stride.yml --transreid-weights models/transreid_market.pth --show
```

The first OSNet run downloads pretrained weights into the ignored
`Ultralytics/torch/` cache. If dependencies are missing, install them into the
same Python environment used to run the script:

```powershell
python -m pip install -r requirements.txt
```

Outputs are written to:

```text
outputs_single_tracker/cam_room_a_tracks.jsonl
outputs_single_tracker/cam_room_a_tracks.mp4
outputs_single_tracker/crops/
```

Visual controls:

```text
Space  pause/play
Q/Esc  quit
S      save screenshot to outputs_single_tracker/screenshots/
```

## Edit polygons

The `areas` polygons in `scenario_config.json` are foot-point localization
polygons for 1920x1080 video. They should cover the floor where a detected
person's bottom-center bbox point can land:

- `Room`
- `Door_1`
- `Door_2`
- `Door_3`
- `Corridor`

Door `areas` should be narrow floor bands around the threshold, not the full
vertical door frame. Keep `door_rois` as separate visual/search regions around
the doorway; do not copy door ROIs directly into `areas`.

The pipeline maps a detected person's bottom-center bbox point into one of
these floor areas and uses the area's configured `position` as the coarse map
coordinate. If a real YOLO track falls outside every polygon, it falls back to
the nearest non-door floor area so door output only occurs on an explicit door
floor band.
