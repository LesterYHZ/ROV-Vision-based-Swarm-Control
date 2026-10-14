#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from pupil_apriltags import Detector

class RovTagDetector(Node):
    def __init__(self):
        super().__init__('rov_tag_detector')
        self.bridge = CvBridge()
        
        # 1. Initialize the AprilTag detector (specifying the detection family as tag36h11).
        self.detector = Detector(families='tag36h11',
                                 nthreads=1,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

        # 2. [Definition of Camera Matrix (Intrinsic Parameters)]
	    #    These parameters determine the accuracy of back-projecting 2D image pixels to 3D spatial locations. 
	    #    640x480 resolution and a field of view of approximately 80 degrees (1.396 rad).
        self.fx = 381.36  # Focal length X
        self.fy = 381.36  # Focal length Y
        self.cx = 320.0   # Optical center X
        self.cy = 240.0   # Optical center Y
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]
        
        # Define the physical size of the actual AprilTag (in meters). 
        self.tag_size = 0.15 # m

        # 3. Dynamic configuration parameters: Define the cameras of which vehicles you need to monitor simultaneously.
        self.declare_parameter('num_robots', 2)
        num_robots = self.get_parameter('num_robots').get_parameter_value().integer_value

        # 4. Batch create multi-channel subscribers and corresponding pose publishers.
        self.subs = []
        self.pose_pubs = []
        
        for i in range(num_robots):
            robot_name = f"bluerov_{i}"
            image_topic = f"/model/{robot_name}/image_raw"
            pose_topic = f"/model/{robot_name}/detected_tag_pose"
            
            # Use a Python closure to bind the independent `robot_name` for the current iteration of the loop.
            callback = lambda msg, r_name=robot_name: self.image_callback(msg, r_name)
            
            sub = self.create_subscription(Image, image_topic, callback, 10)
            pub = self.create_publisher(PoseStamped, pose_topic, 10)
            
            self.subs.append(sub)
            self.pose_pubs.append((robot_name, pub))
            
            self.get_logger().info(f"Multi-vehicle tracking channel successfully opened: Listening.[{image_topic}]")

    def image_callback(self, msg, robot_name):
        try:
            # Convert ROS 2 image messages to OpenCV grayscale images.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed. ({robot_name}): {str(e)}")
            return

        # Perform AprilTag detection and calculate the 3D pose directly using intrinsic parameters (estimate_tag_pose=True).
        tags = self.detector.detect(gray_image, 
                                    estimate_tag_pose=True, 
                                    camera_params=self.camera_params, 
                                    tag_size=self.tag_size)

        for tag in tags:
            # Extract detected tag information.
            tag_id = tag.tag_id
            
            # tag.pose_t is a 3x1 translation matrix representing the physical 3D distance (X, Y, Z) of the tag relative to the camera.
            t_vec = tag.pose_t.flatten() 
            # tag.pose_R is a 3x3 rotation matrix representing the rotational pose of the tag in 3D space.
            r_mat = tag.pose_R 

            # Convert a 3x3 rotation matrix to a ROS 2-compliant quaternion.
            quat = self.rotation_matrix_to_quaternion(r_mat)

            # Print logs: Output which vehicle detected a tag with a specific ID and the distance to it.
            distance = np.linalg.norm(t_vec)
            self.get_logger().info(
                f"【Detection Successful】Vehicle [{robot_name}] detected target [AprilTag ID: {tag_id}] -> "
                f"Relative distance: {distance:.2f}m | Relative coordinates: X={t_vec[0]:.2f}, Y={t_vec[1]:.2f}, Z={t_vec[2]:.2f}"
            )

            # Encapsulate and publish the ROS 2 standard PoseStamped message.
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = f"{robot_name}/front_camera_optical_frame"
            
            pose_msg.pose.position.x = t_vec[0]
            pose_msg.pose.position.y = t_vec[1]
            pose_msg.pose.position.z = t_vec[2]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            # Locate the corresponding publisher and broadcast the 3D coordinates.
            for r_name, pub in self.pose_pubs:
                if r_name == robot_name:
                    pub.publish(pose_msg)

            # Draw detection bounding boxes and ID labels on the OpenCV frame.
            for j in range(4):
                pt1 = tuple(tag.corners[j].astype(int))
                pt2 = tuple(tag.corners[(j + 1) % 4].astype(int))
                cv2.line(cv_image, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(cv_image, f"TAG ID: {tag_id}", (int(tag.center[0])-40, int(tag.center[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # pop-up windows display the visualized rendered view 
        cv2.imshow(f"Detection View - {robot_name}", cv_image)
        cv2.waitKey(1)

    def rotation_matrix_to_quaternion(self, R): 
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = RovTagDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
