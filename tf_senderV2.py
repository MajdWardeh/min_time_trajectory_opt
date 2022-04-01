import time
from math import degrees, pi
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as rot
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
import tf2_ros
import geometry_msgs.msg
from cv_bridge import CvBridge
import cv2

class TF_Sender:
    def __init__(self, pose_list, mc_list=None, pose_update_rate=1000.0):
        rospy.init_node('tf_sender_node', anonymous=True)
        self.pose_list = pose_list
        self.mc_list = mc_list
        self.pose = pose_list[0]
        self.idx = 0
        self.update_count = 0
        self.enable_pose_update = False
        self.done = False

        self.pose_update_rate = pose_update_rate
        self.camera_k = np.array([342.7555236816406, 0.0, 320.0, 0.0, 342.7555236816406, 240.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        self.cv_bridge = CvBridge()

        self.stamp_index_dect = {}

        self.transorm_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_update_timer = rospy.Timer(rospy.Duration(1/self.pose_update_rate), self.poseUpdateTimerCallback)

        if self.mc_list is not None:
            self.image_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.imageCallback)

        print('tf sender node started...')
        time.sleep(3)

        self.done = False
        r = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.done:
            r.sleep()
        # time.sleep(5)
    
    def kill(self):
        self.pose_update_timer.shutdown()
        self.image_subs.unregister()
    
    def poseUpdateTimerCallback(self, msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "uav/imu"
        t.transform.translation.x = self.pose[0]
        t.transform.translation.y = self.pose[1]
        t.transform.translation.z = self.pose[2]
        t.transform.rotation.x = self.pose[3]
        t.transform.rotation.y = self.pose[4]
        t.transform.rotation.z = self.pose[5]
        t.transform.rotation.w = self.pose[6]
        try:
            self.transorm_broadcaster.sendTransform(t)
            if self.idx < len(self.pose_list):
                self.stamp_index_dect[t.header.stamp] = self.idx
            self.updatePose()
        except:
            pass

    def updatePose(self):
        ## wait in the begining
        if not self.enable_pose_update:
            self.update_count += 1
            if self.update_count < self.pose_update_rate/2:
                self.update_count = 0
                self.enable_pose_update = True
                return
                
        if self.idx < len(self.pose_list):
            self.pose = self.pose_list[self.idx]
            self.idx += 1
        else:
            self.done = True

    def imageCallback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        idx = self.stamp_index_dect.get(msg.header.stamp, None)
        if idx is not None:
            mC = self.mc_list[idx]
            p = np.matmul(self.camera_k, mC)
            p = p / p[2]
            img = cv2.circle(img, tuple(p[:2]), 5, (0, 0, 255), -1,)
        if not self.done:    
            cv2.imshow('image', img)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()

        

def test():
    p = [0, -5, 2]
    q = rot.from_euler('xyz', [0, 0, 90], degrees=True).as_quat().tolist()
    initialPose = [p + q] * 1000
    tfSender = TF_Sender(initialPose, pose_update_rate=1000.0)





if __name__=='__main__':
    test()