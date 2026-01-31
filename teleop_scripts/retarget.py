import teledex
import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig

class Retarget:
    def __init__(self, hand_type="right", hand_config_path="configs/ro_hand_right_dexpilot.yml"):
        self.OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        self.OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        self.operator2mano = self.OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else self.OPERATOR2MANO_LEFT
        self.detected_hand_type = hand_type
        self.retargeting = RetargetingConfig.load_from_file(hand_config_path).build()
        self.opt = self.retargeting.optimizer


    def get_qpos(self, landmarks, world_landmarks):
        keypoint_2d = landmarks
        keypoint_3d_array = world_landmarks - world_landmarks[0:1]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
        indices = self.retargeting.optimizer.target_link_human_indices


        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = (
            joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        )
        qpos = self.retargeting.retarget(ref_value)

    
        return qpos

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 17]]
        x_vector = points[0] - points[2]
        points = points - points.mean(axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2]
        x = x_vector - np.sum(x_vector * normal) * normal
        x /= np.linalg.norm(x)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        return np.stack([x, normal, z], axis=1)
