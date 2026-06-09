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
  editable area polygons, door ROIs, and coarse output positions.
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

Visual debugger controls:

```text
Space  pause/play
N      step one frame while paused
Q/Esc  quit
P      toggle area polygons
D      toggle door ROIs
T      toggle AprilTag search ROIs
F      toggle fused summary
S      save a 2x2 debug screenshot to outputs_debug/
```

Outputs are written to:

```text
outputs/local_observations.jsonl
outputs/fused_positions.jsonl
outputs/fused_positions.csv
```

## Edit polygons

The polygons in `scenario_config.json` are placeholders for 1920x1080 video.
Update them for each camera after checking real footage:

- `Room`
- `Door_1`
- `Door_2`
- `Door_3`
- `Corridor_Near`
- `Corridor_Far`
- `Corner_Turn`

The first version maps a detected person's foot point into one of these areas
and uses the area's configured `position` as the coarse map coordinate.
