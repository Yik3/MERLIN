#!/usr/bin/env python3

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_kinematics_monitor import *
from phone import Phone
from retarget import Retarget


MAX_VALUE = 65535
INV_1_46 = 1.0 / 1.46
INV_1_57 = 1.0 / 1.57


def phone_follow(
    hand='right',
    scale=1.0,
    use_retargeting=True,
    hand_enabled=True,
):
    phone_tracker = Phone(scale=scale, hand=hand_enabled)
    retarget = Retarget(hand_type=hand) if use_retargeting else None

    phone_tracker.start()

    arm_ip = RIGHT_ARM_IP if hand == 'right' else LEFT_ARM_IP
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    if arm.rm_create_robot_arm(ip=arm_ip, port=PORT).id == -1:
        phone_tracker.stop()
        return
    arm.rm_change_work_frame('World')

    arm.rm_set_hand_speed(1)
    arm.rm_set_hand_force(1)
    arm.rm_set_hand_follow_pos([0, 0, 0, 0, 0, 30000], block=2000)

    time.sleep(0.3)

    initial_robot_transform = get_pose(arm)
    if initial_robot_transform is None:
        arm.rm_delete_robot_arm()
        phone_tracker.stop()
        return

    phone_tracker.reset(initial_robot_transform)

    try:
        while True:
            

            if use_retargeting and hand_enabled:
                landmarks, world_landmarks = phone_tracker.get_landmarks()
                if landmarks is None or world_landmarks is None:
                    continue
                qpos = retarget.get_qpos(landmarks, world_landmarks)

                index = np.clip(qpos[0], -1.46, 0.0) * -INV_1_46
                pinky = np.clip(qpos[2], -1.46, 0.0) * -INV_1_46
                middle = np.clip(qpos[4], -1.46, 0.0) * -INV_1_46
                rings = np.clip(qpos[6], -1.46, 0.0) * -INV_1_46
                thumb_1 = np.clip(qpos[8], 0.0, 1.57) * INV_1_57
                thumb_2 = np.clip(qpos[9], -1.46, 0.0) * -INV_1_46

                arm.rm_set_hand_follow_pos(
                    [
                        int(MAX_VALUE * 3 * thumb_2),
                        int(MAX_VALUE * index),
                        int(MAX_VALUE * middle),
                        int(MAX_VALUE * rings),
                        int(MAX_VALUE * pinky),
                        int(MAX_VALUE * thumb_1),
                    ],
                    block=0,
                )


            target_pose, _ = phone_tracker.get_target_pose()
            if target_pose is not None:
                move_to_pose_follow(arm, target_pose)

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    finally:
        rot = R.from_matrix(initial_robot_transform[:3, :3]).as_quat()
        pos = initial_robot_transform[:3, 3]
        init_pose = list(pos) + [rot[3], rot[0], rot[1], rot[2]]

        move_to_pose(arm, init_pose, speed=100)

        arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()
        phone_tracker.stop()


if __name__ == '__main__':
    phone_follow(
        hand='right',
        scale=1.0,
        use_retargeting=1,
        hand_enabled=1,
    )
