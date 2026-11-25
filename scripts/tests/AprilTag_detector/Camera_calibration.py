import cv2
import numpy as np
import glob

# ====================================================================
# CONFIGURATION
# ====================================================================
# Define the chessboard dimensions (number of INSIDE corners)
# e.g., an 8x6 square chessboard has 7x5 inside corner intersections.
CHESSBOARD_SIZE = (10, 7)

# Size of a square side in metres
SQUARE_SIZE = 0.021  # e.g., 0.021 meters = 21mm

# Path to the directory containing your calibration images (e.g., .jpg or .png)
IMAGE_PATH_PATTERN = "calibration_images/*.jpg"
# ====================================================================

# Stop criteria for refining corner detection accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points in real-world space: (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Grab all calibration images
images = glob.glob(IMAGE_PATH_PATTERN)

if not images:
    print(f"Error: No images found matching pattern '{IMAGE_PATH_PATTERN}'")
    exit()

gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        # Refine the pixel coordinates for sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optional: Draw and display the corners to verify detection
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(100)  # Pause briefly for each image
    else:
        print(f"Warning: Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()

# Run the camera calibration
if len(objpoints) > 0:
    print("\nCalculating calibration... please wait...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # Print results to the console
    print("\n=== CAMERA CALIBRATION RESULTS ===")
    print(f"Reprojection Error: {ret:.4f} pixels (Lower is better, ideally < 0.5)")
    
    print("\n--- Camera Matrix (K) ---")
    print(mtx)
    print(f"  fx = {mtx[0,0]:.2f}")
    print(f"  fy = {mtx[1,1]:.2f}")
    print(f"  cx = {mtx[0,2]:.2f}")
    print(f"  cy = {mtx[1,2]:.2f}")
    
    print("\n--- Distortion Coefficients (D) ---")
    print(f"  k1, k2, p1, p2, k3 = {dist.ravel()}")

    # Save the parameters to a file for your AprilTag script to load
    np.savez("camera_params.npz", mtx=mtx, dist=dist)
    print("\nParameters successfully saved to 'camera_params.npz'")

else:
    print("Error: Calibration failed because no valid chessboards were detected.")
