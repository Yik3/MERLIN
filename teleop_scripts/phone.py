#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import time
import numpy as np
import teledex

class Phone:
    
    def __init__(self, scale=1.0, hand=False):
        self.session = teledex.Session()
        self.scale = scale
        self.initial_ar_pose = None
        self.initial_robot_position = None
        self.initial_robot_rotation = None
        self.fix_rot = R.from_euler('z', 90, degrees=True).as_matrix()
        self.hand = hand
        if self.hand:
            self.position_name = "position_hand"
            self.rotation_name = "rotation_hand"
        else:
            self.position_name = "position"
            self.rotation_name = "rotation"

    def get_pose(self):
        data = self.session.get_latest_data()
        pose = np.eye(4)
        pose[:3, 3] = data[self.position_name]
        pose[:3, :3] = data[self.rotation_name].reshape(3, 3)
        return pose

    def start(self):
        self.session.start()
        while self.session.get_latest_data()["position"] is None:
            time.sleep(0.1)
    
    def reset(self, robot_transform):
        self.initial_ar_pose = self.get_pose()
        self.initial_robot_position = robot_transform[:3, 3]
        self.initial_robot_rotation = R.from_matrix(robot_transform[:3, :3])
    
    def get_target_pose(self):
        current_pose = self.get_pose()

        delta_pose = np.linalg.inv(self.initial_ar_pose) @ current_pose
        delta_phone_position = delta_pose[:3, 3]
        delta_phone_rotation = delta_pose[:3, :3]
        
        delta_phone_rotation_fixed = self.fix_rot.T @ delta_phone_rotation @ self.fix_rot
        delta_phone_position_fixed = self.fix_rot.T @ delta_phone_position
        
        relative_position = delta_phone_position_fixed * self.scale
        target_position = self.initial_robot_position + relative_position
        
        delta_rotation = R.from_matrix(delta_phone_rotation_fixed)
        target_rotation = delta_rotation * self.initial_robot_rotation
        target_quat = target_rotation.as_quat()
        
        target_pose = list(target_position) + [target_quat[3], target_quat[0], target_quat[1], target_quat[2]]
        
        return target_pose, self.get_grasp_state()
    
    def stop(self):
        self.session.stop()

    def get_landmarks(self):
        data = self.session.get_latest_data()
        if "landmarks" in data:
            return data["landmarks"], data["world_landmarks"]
        return None, None

    def get_grasp_state(self):
        if self.hand:
            data = self.session.get_latest_data()
            if "landmarks" not in data:
                return None
            landmarks = data["landmarks"]
            if landmarks is None or len(landmarks) < 21:
                return None
            
            fingers = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [0, 9, 10, 11, 12],
                [0, 13, 14, 15, 16],
                [0, 17, 18, 19, 20]
            ]
            
            curls = []
            palm = landmarks[0]
            
            for indices in fingers:
                tip_to_base_vec = landmarks[indices[-1]] - palm
                tip_to_base = np.sqrt(np.sum(tip_to_base_vec * tip_to_base_vec))
                
                segments = landmarks[indices[1:]] - landmarks[indices[:-1]]
                segment_lengths = np.sqrt(np.sum(segments * segments, axis=1))
                extended_length = np.sum(segment_lengths)
                
                if extended_length > 0:
                    curl = 1 - (tip_to_base / extended_length)
                    curls.append(np.clip(curl, 0, 1))
                else:
                    curls.append(0)
            
            return curls
        else:
            toggle = self.session.get_latest_data().get("toggle", None)
            return toggle if toggle is not None else None