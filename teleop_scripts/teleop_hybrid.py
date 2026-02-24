#!/usr/bin/env python3
"""
Hybrid teleop: SpaceMouse controls end-effector pose, phone hand tracking
controls fingers. Uses same low-level control as teleop_smooth (rm_movep_canfd
+ rm_set_hand_follow_pos).
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_kinematics_monitor import get_pose, RIGHT_ARM_IP, LEFT_ARM_IP, PORT
from phone import Phone
from spacemouse import Spacemouse
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from retarget import Retarget


MAX_VALUE = 65535
INV_1_46 = 1.0 / 1.46
INV_1_57 = 1.0 / 1.57


def smooth_hybrid_teleop(
    hand='right',
    cycle_time=0.005,
    max_velocity=0.3,
    max_rotation_velocity=1.0,
    translation_gain=0.5,
    rotation_gain=1.0,
    interpolation_alpha=0.2,
    rotation_interpolation=0.15,
    spacemouse_deadzone=0.1,
    use_retargeting=True,
):
    """
    SpaceMouse → end-effector pose; Phone (hand tracking) → fingers.
    Same low-level API as teleop_smooth: rm_movep_canfd + rm_set_hand_follow_pos.
    """
    phone_tracker = Phone(scale=1.0, hand=True)
    retarget = Retarget(hand_type=hand) if use_retargeting else None

    # Start Spacemouse first (non-blocking thread); then wait for phone (blocking).
    sm = Spacemouse(deadzone=spacemouse_deadzone)
    sm.start()
    phone_tracker.start()

    arm_ip = RIGHT_ARM_IP if hand == 'right' else LEFT_ARM_IP
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    if arm.rm_create_robot_arm(ip=arm_ip, port=PORT).id == -1:
        print("Failed to connect to robot arm")
        phone_tracker.stop()
        sm.stop()
        return

    arm.rm_change_work_frame('World')
    time.sleep(0.2)

    arm.rm_set_hand_speed(1)
    arm.rm_set_hand_force(1)
    arm.rm_set_hand_follow_pos([0, 0, 0, 0, 0, 30000], block=2000)
    time.sleep(0.1)

    initial_robot_transform = get_pose(arm)
    if initial_robot_transform is None:
        print("Failed to get initial robot pose")
        arm.rm_delete_robot_arm()
        phone_tracker.stop()
        sm.stop()
        return

    initial_position = initial_robot_transform[:3, 3].copy()
    initial_rotation = R.from_matrix(initial_robot_transform[:3, :3])
    locked_quat = initial_rotation.as_quat()
    locked_orientation = [locked_quat[3], locked_quat[0], locked_quat[1], locked_quat[2]]

    time.sleep(1.0)
    phone_tracker.reset(initial_robot_transform)

    current_position = initial_position.copy()
    current_rotation = initial_rotation
    max_step = max_velocity * cycle_time
    max_rotation_step = max_rotation_velocity * cycle_time

    print("=" * 60)
    print("Hybrid teleop: SpaceMouse = EE pose, Phone = fingers")
    print("=" * 60)
    print(f"Hand: {hand}, Cycle: {cycle_time*1000:.1f}ms")
    print(f"Max velocity: {max_velocity} m/s, Max rotation: {max_rotation_velocity} rad/s")
    print(f"Translation gain: {translation_gain}, Rotation gain: {rotation_gain}")
    print("Press CTRL+C to stop")
    print("=" * 60)

    try:
        loop_count = 0
        start_time = time.time()

        while True:
            loop_start = time.time()

            # ---- End-effector pose from SpaceMouse (velocity integration) ----
            motion = sm.get_motion_state_transformed()
            # motion[:3] = translation (forward, right, up), motion[3:] = rotation (roll, pitch, yaw)
            delta_pos = motion[:3] * translation_gain * cycle_time
            delta_norm = np.linalg.norm(delta_pos)
            if delta_norm > max_step:
                delta_pos = delta_pos * (max_step / delta_norm)
            current_position = current_position + interpolation_alpha * delta_pos

            rotvec = motion[3:] * rotation_gain * max_rotation_velocity * cycle_time
            angle = np.linalg.norm(rotvec)
            if angle > max_rotation_step and angle > 1e-9:
                rotvec = rotvec * (max_rotation_step / angle)
            rotvec = rotvec * rotation_interpolation
            if np.linalg.norm(rotvec) > 1e-9:
                current_rotation = current_rotation * R.from_rotvec(rotvec)

            current_quat = current_rotation.as_quat()
            current_orientation = [current_quat[3], current_quat[0], current_quat[1], current_quat[2]]
            cmd = list(current_position) + list(current_orientation)

            ret = arm.rm_movep_canfd(cmd, follow=1, trajectory_mode=0, radio=0)
            if ret != 0 and loop_count % 50 == 0:
                print(f"Warning: rm_movep_canfd returned {ret}")

            # ---- Fingers from phone hand retargeting ----
            if use_retargeting:
                landmarks, world_landmarks = phone_tracker.get_landmarks()
                if landmarks is not None and world_landmarks is not None:
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

            loop_count += 1
            if loop_count % int(2.0 / cycle_time) == 0:
                elapsed = time.time() - start_time
                print(f"Loop {loop_count}, Avg {loop_count/elapsed:.1f} Hz")

            elapsed = time.time() - loop_start
            sleep_time = cycle_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")

    finally:
        print("Returning to initial position...")
        rot = R.from_matrix(initial_robot_transform[:3, :3]).as_quat()
        pos = initial_robot_transform[:3, 3]
        init_pose = list(pos) + [rot[3], rot[0], rot[1], rot[2]]
        from hand_kinematics_monitor import move_to_pose
        move_to_pose(arm, init_pose, speed=50)
        arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()
        phone_tracker.stop()
        sm.stop()
        print("Teleoperation stopped.")


if __name__ == '__main__':
    smooth_hybrid_teleop(
        hand='right',
        cycle_time=0.05,
        max_velocity=0.5,
        max_rotation_velocity=10.0,
        translation_gain=0.5,
        rotation_gain=1.0,
        interpolation_alpha=0.2,
        rotation_interpolation=0.15,
        spacemouse_deadzone=0.1,
        use_retargeting=True,
    )
