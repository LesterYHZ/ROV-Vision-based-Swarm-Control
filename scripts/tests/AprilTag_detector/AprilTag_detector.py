import cv2
import numpy as np
from pupil_apriltags import Detector

# ====================================================================
# CONFIGURATION
# ====================================================================
# The tag family you are using (e.g., 'tag36h11', 'tag16h5', 'tagCustom48h12')
TAG_FAMILY = "tag36h11"

# The literal side length of the tag square in METERS (excluding the white border)
# Crucial for accurate distance/translation calculations!
TAG_SIZE_METERS = 0.16  # e.g., 0.10 meters = 10 cm

# Path to your saved calibration file
PARAMS_FILE = "camera_params.npz"
# ====================================================================

# 1. Load the calibration parameters
try:
    with np.load(PARAMS_FILE) as data:
        mtx = data['mtx']
        dist = data['dist']
    print("Successfully loaded camera calibration data.")
except FileNotFoundError:
    print(f"Error: Calibration file '{PARAMS_FILE}' not found. Please run calibration first.")
    exit()

# Extract focal length and optical centers needed for the detector
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
camera_params = [fx, fy, cx, cy]

# 2. Initialize the AprilTag Detector
# We set families and specify 1 thread for standard processing (increase if slow)
detector = Detector(families=TAG_FAMILY, nthreads=1)

# 3. Start Video Capture (0 is usually the default web camera)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Match resolution parameters if your calibration was done at a specific size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\nStarting AprilTag ({TAG_FAMILY}) detection loop. Press 'q' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # frame = cv2.flip(frame, 1)
    # Convert to grayscale (AprilTag detector requires 8-bit single channel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Detect tags and extract 3D pose
    # estimate_tag_pose=True calculates the translation and rotation matrices
    results = detector.detect(
        gray, 
        estimate_tag_pose=True, 
        camera_params=camera_params, 
        tag_size=TAG_SIZE_METERS
    )

    # Loop over all detected tags in the current frame
    for r in results:
        # Extract boundary corners (4 points) and center coordinates
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        cX, cY = int(r.center[0]), int(r.center[1])

        # Draw the bounding box around the AprilTag
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

        # Draw the center point of the tag
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # 5. Interpret the Pose Estimation Matrix
        # r.pose_t is the Translation vector [X, Y, Z] relative to the camera lens
        # Units match TAG_SIZE_METERS (meters)
        tx, ty, tz = r.pose_t[0][0], r.pose_t[1][0], r.pose_t[2][0]
        # Rotation angles
        rx, ry, rz = r.pose_R[0][0], r.pose_R[1][1], r.pose_R[2][2]
        
        # Calculate straight-line Euclidean distance to the tag
        distance = np.sqrt(tx**2 + ty**2 + tz**2)

        # Render the ID and structural distance on the video frames
        text = f"ID: {r.tag_id} | Dist: {distance:.2f}m"
        cv2.putText(frame, text, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print coordinate telemetry to console
        print(f"Tag ID: {r.tag_id} | X: {tx:.2f}m, Y: {ty:.2f}m, Z (Depth): {tz:.2f}m")
        print(f"roll: {np.rad2deg(rx):.2f}, pitch: {np.rad2deg(ry):.2f}, yaw: {np.rad2deg(rz):.2f} (degrees)")
        
        # show the coordinate axes for each detected tag
        # Define the 3D points of the axes in the tag's coordinate frame
        axis_length = TAG_SIZE_METERS / 2  # Length of the axes lines
        axis_points = np.float32([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, -axis_length]   # Z-axis (negative for camera coordinate system)
        ]).reshape(-1, 3)
        # Project the 3D axis points to the 2D image plane
        # OpenCV expects a rotation vector here, while pupil_apriltags
        # provides the pose rotation as a 3x3 matrix.
        pose_rvec, _ = cv2.Rodrigues(r.pose_R)
        axis_projections, _ = cv2.projectPoints(
            axis_points, pose_rvec, r.pose_t, mtx, dist
        )
        # Draw the projected axis lines on the image
        for i, pt in enumerate(axis_projections):
            # center axes at the tag's center
            pt = (int(pt[0][0]), int(pt[0][1]))
            if i == 0:
                origin = pt
            else:
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i - 1]  # BGR: Z, Y, X
                cv2.line(frame, origin, pt, color, 2) 

    # Display the output window
    cv2.imshow("AprilTag Detector & Tracker", frame)

    # Break loop cleanly if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Clean up resource allocations
cap.release()
cv2.destroyAllWindows()
