import cv2
import numpy as np
from pupil_apriltags import Detector

# ==================== 配置 ====================
camera_params = [720, 720, 960, 540]   # [fx, fy, cx, cy]  # 相机参数
tag_size = 0.1                                 # Tag 实际边长（米）

# 创建检测器
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# 读取视频
video_path = "shoulder1080.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 无法打开视频: {video_path}")
    exit()

print(f"视频分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"视频帧率: {cap.get(cv2.CAP_PROP_FPS):.2f} FPS")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ 视频结束")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )

    for det in detections:
        print(f"Frame {frame_count} | Tag ID: {det.tag_id} | "
              f"Distance: {np.linalg.norm(det.pose_t):.3f} m")

        # 画检测框
        for i in range(4):
            p1 = tuple(det.corners[i].astype(int))
            p2 = tuple(det.corners[(i + 1) % 4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 3)

        # ========== 修复后的画坐标轴部分 ==========
        if det.pose_R is not None:
            # 1. 旋转矩阵 → 旋转向量 (rvec)
            R = det.pose_R
            rvec, _ = cv2.Rodrigues(R)          # 关键转换

            tvec = det.pose_t.flatten()

            # 2. 构建相机内参矩阵
            cam_matrix = np.array([
                [camera_params[0], 0, camera_params[2]],
                [0, camera_params[1], camera_params[3]],
                [0, 0, 1]
            ], dtype=np.float64)

            # 3. 畸变系数（当前无标定，用 0）
            dist_coeffs = np.zeros(5, dtype=np.float64)   # 必须是 1D 数组

            # 画坐标轴
            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, tag_size * 0.5)

        # 显示文字信息
        dist = np.linalg.norm(det.pose_t)
        cv2.putText(frame, f"ID:{det.tag_id} D:{dist:.2f}m",
                    (int(det.center[0]), int(det.center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("AprilTag Pose Estimation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)   # 空格暂停

cap.release()
cv2.destroyAllWindows()
