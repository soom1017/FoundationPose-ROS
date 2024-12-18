

import ros,rospy
import os, sys, cv2, glob, copy, time, argparse
# sys.path.append('/root/Figueroa_Lab/catkin_ws/src')
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
import numpy as np
import yaml
import tf2_ros as tf
from geometry_msgs.msg import TransformStamped
from transformations import *
from rospy import Time
# from std_srvs.srv import Empty, EmptyResponse
# from std_msgs.msg import Float64MultiArray, Float32MultiArray
from sensor_msgs.msg import Image as Imgs
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from threading import Event

from estimater import *
from datareader import *
import argparse

class TrackerRos:
  def __init__(self, FoundationPose_object):

    self.color = None
    self.depth = None
    self.cam_K = None
    self.cur_time = None
    self.mask = None
    self.pose = None
    self.est = FoundationPose_object

    self.color_event = Event()
    self.depth_event = Event()
    self.mask_event = Event()
    self.camInfo_event = Event()

    self.sub_depth = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Imgs, self.grab_depth)
    self.sub_color = rospy.Subscriber("/camera/color/image_raw", Imgs, self.grab_color)
    self.sub_mask = rospy.Subscriber("/mask_image", Imgs, self.grab_mask)
    self.sub_camInfo = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.grab_camInfo)

    tf_buffer = tf.Buffer()
    self.tf_listener = tf.TransformListener(tf_buffer)

    self.transform_msg = TransformStamped()
    self.tf_pub = tf.TransformBroadcaster()

  def register(self):
    self.depth = self.depth
    self.pose = self.est.register(K=self.cam_K, rgb=self.color, depth=self.depth, ob_mask=self.mask, iteration=5)
    print(self.pose)


  def reset(self,pose_init):
    self.color = None
    self.depth = None
    self.cur_time = None
    self.pose = self.est.register(K=self.cam_K, rgb=self.color, depth=self.depth, ob_mask=self.mask, iteration=5)


  def grab_camInfo(self, msg):
    self.img_H = msg.height
    self.img_W = msg.width
    self.cam_K = msg.K
    self.camInfo_event.set()
    self.cam_K = np.array(self.cam_K).reshape((3,3))


  def grab_depth(self, msg):
    depth = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # depth = fill_depth(depth/1e3,max_depth=2.0,extrapolate=False)
    depth = depth/1000
    depth[(depth<0.1) | (depth>5)] = 0
    self.depth = depth
    self.depth_event.set()


  def grab_color(self, msg):
    self.cur_time = msg.header.stamp
    color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    self.color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    self.color_event.set()


  def grab_mask(self, msg):
    mask = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.uint8)
    self.mask = mask
    self.mask_event.set()


  def on_track(self):
    print("TRACKING")
    self.depth = self.depth

    self.pose = self.est.track_one(rgb=self.color, depth=self.depth, K=self.cam_K, iteration=2)

    trans = self.pose[:3,3]
    q_wxyz = (Rotation.from_matrix(self.pose[:3, :3])).as_quat()

    print("Object in Cam - \n", self.pose, "\n\n")

    self.transform_msg.header.stamp = self.cur_time
    self.transform_msg.header.frame_id = "camera_color_optical_frame"
    self.transform_msg.child_frame_id = "tracked_object_origin_in_cam"
    
    self.transform_msg.transform.translation.x = trans[0]
    self.transform_msg.transform.translation.y = trans[1]
    self.transform_msg.transform.translation.z = trans[2]

    self.transform_msg.transform.rotation.w = q_wxyz[3]
    self.transform_msg.transform.rotation.x = q_wxyz[0]
    self.transform_msg.transform.rotation.y = q_wxyz[1]
    self.transform_msg.transform.rotation.z = q_wxyz[2]

    self.tf_pub.sendTransform(self.transform_msg)


  def wait_for_data(self):
        # Wait until all required data has been received
        rospy.loginfo("Waiting for all data to be received...")
        self.color_event.wait()
        self.depth_event.wait()
        self.mask_event.wait()
        self.camInfo_event.wait()
        rospy.loginfo("All data received")

  
if __name__=="__main__":
  parser = argparse.ArgumentParser(description="FoundationPose implementation in ROS")
  parser.add_argument('-in', '--input', help='Input mesh file', default="demo_data/leverhandle.obj")
  args = parser.parse_args()

  rospy.init_node('FoundationPose_Node', anonymous=True)
  code_dir = os.path.dirname(os.path.realpath(__file__))

  # mesh_file = "mesh/kinect_driller/textured_mesh.obj" # Hand Drill
  mesh_file = args.input  
  debug = 1
  debug_dir = "outputs"

  mesh = trimesh.load(mesh_file)
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)

  count = 0
  registered = False

  while not rospy.is_shutdown():
  
    if count == 0:
      ros_tracker = TrackerRos(est)
      print("TRACKER INITIALIZED")

      ros_tracker.wait_for_data()

      ros_tracker.register()

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(ros_tracker.pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(ros_tracker.depth, ros_tracker.cam_K)
        valid = ros_tracker.depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], ros_tracker.color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

      registered = True
      print("FIRST FRAME REGISTERED")

    if registered:
      ros_tracker.on_track()
    
    if debug>=1:
      # cv2.imshow('2', depth_to_vis(ros_tracker.depth.reshape(480,640), inverse=False))
      # cv2.waitKey(1)
      center_pose = ros_tracker.pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(ros_tracker.cam_K, img=ros_tracker.color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(ros_tracker.color, ob_in_cam=center_pose, scale=0.1, K=ros_tracker.cam_K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)

    count = count + 1

