#!/usr/bin/env python3

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_kinematics_monitor import get_pose
from phone import Phone
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

LEFT_ARM_IP = "169.254.128.18"
RIGHT_ARM_IP = "169.254.128.19"
PORT = 8080

DT = 0.01
ALPHA = 0.08
MAX_DPOS = 0.001   # 0.5 mm per frame (safe)


def phone_follow(hand="right", scale=1.0):
    phone = Phone(scale=scale, hand=0)
    phone.start()

    arm_ip = RIGHT_ARM_IP if hand == "right" else LEFT_ARM_IP
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    if arm.rm_create_robot_arm(arm_ip, PORT).id == -1:
        phone.stop()
        return

    # ✅ FORCE WORLD FRAME
    arm.rm_change_work_frame("World")
    time.sleep(0.2)

    # ---- Robot initial pose ----
    T0 = get_pose(arm)
    if T0 is None:
        arm.rm_delete_robot_arm()
        phone.stop()
        return

    robot_pos0 = T0[:3, 3].copy()
    world_R = R.from_matrix(T0[:3, :3])

    # Lock orientation forever
    ret, state = arm.rm_get_current_arm_state()
    locked_orientation = state["pose"][3:]

    # ---- Phone reset ----
    phone.reset(T0)
    phone_pose0, _ = phone.get_target_pose()
    phone_pos0 = np.array(phone_pose0[:3], dtype=float)

    current_pos = robot_pos0.copy()

    print("📱 Phone → WORLD position teleop")
    print("Position only | CTRL+C to stop")

    try:
        while True:
            phone_pose, _ = phone.get_target_pose()
            if phone_pose is None:
                time.sleep(DT)
                continue

            phone_pos = np.array(phone_pose[:3], dtype=float)

            # Δ in phone frame
            delta_phone = phone_pos - phone_pos0

            # Rotate into world frame
            delta_world = world_R.apply(delta_phone)

            # Per-frame clamp (RealMan safety)
            delta_world = np.clip(delta_world, -MAX_DPOS, MAX_DPOS)

            # Smooth
            current_pos = current_pos + ALPHA * delta_world

            cmd = list(current_pos) + list(locked_orientation)

            arm.rm_movep_canfd(
                cmd,
                follow=True,
                trajectory_mode=0,  # FULL passthrough
                radio=0
            )

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nStopping")

    finally:
        arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()
        phone.stop()


if __name__ == "__main__":
    phone_follow(
        hand="right",
        scale=1.0
    )
