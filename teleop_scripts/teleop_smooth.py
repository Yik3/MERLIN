#!/usr/bin/env python3
"""
Smooth position-only teleoperation using rm_movep_canfd with filtering.
No hand control, no retargeting - just smooth position tracking.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from hand_kinematics_monitor import get_pose, RIGHT_ARM_IP, LEFT_ARM_IP, PORT
from phone import Phone
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from retarget import Retarget


MAX_VALUE = 65535
INV_1_46 = 1.0 / 1.46
INV_1_57 = 1.0 / 1.57


def smooth_position_teleop(
    hand='right',
    scale=1.0,
    filter_strength=500,  # 0-1000, higher = smoother
    cycle_time=0.005,     # 5ms cycle
    max_velocity=0.3,     # m/s, maximum end-effector velocity
    interpolation_alpha=0.2,  # 0-1, lower = smoother but more lag
    enable_rotation=True,  # Enable rotation control
    rotation_interpolation=0.15,  # Rotation interpolation (lower = smoother)
    max_rotation_velocity=1.0,  # rad/s, maximum rotation velocity
    hand_enabled=True,  # Enable hand/finger control
    use_retargeting=True,  # Enable hand retargeting
):
    """
    Smooth position and rotation teleoperation using pose pass-through with filtering.
    
    Args:
        hand: 'right' or 'left'
        scale: Position scaling factor
        filter_strength: Filtering parameter (0-1000), higher values = smoother motion
        cycle_time: Control loop cycle time in seconds
        max_velocity: Maximum end-effector velocity in m/s
        interpolation_alpha: Interpolation factor (0-1), lower = smoother motion
        enable_rotation: Enable rotation control from phone
        rotation_interpolation: Rotation interpolation factor (0-1), lower = smoother
        max_rotation_velocity: Maximum rotation velocity in rad/s
        hand_enabled: Enable hand/finger control
        use_retargeting: Enable hand retargeting from phone hand tracking
    """
    
    # Initialize phone tracker (with hand tracking if enabled)
    phone_tracker = Phone(scale=scale, hand=hand_enabled)
    retarget = Retarget(hand_type=hand) if use_retargeting and hand_enabled else None
    
    phone_tracker.start()

    # Connect to robot arm
    arm_ip = RIGHT_ARM_IP if hand == 'right' else LEFT_ARM_IP
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    if arm.rm_create_robot_arm(ip=arm_ip, port=PORT).id == -1:
        print("Failed to connect to robot arm")
        phone_tracker.stop()
        return
    
    # Set work frame to World
    arm.rm_change_work_frame('World')
    time.sleep(0.2)
    
    # Initialize hand if enabled
    if hand_enabled:
        arm.rm_set_hand_speed(1)
        arm.rm_set_hand_force(1)
        arm.rm_set_hand_follow_pos([0, 0, 0, 0, 0, 30000], block=2000)
        time.sleep(0.1)

    # Get initial robot pose
    initial_robot_transform = get_pose(arm)
    if initial_robot_transform is None:
        print("Failed to get initial robot pose")
        arm.rm_delete_robot_arm()
        phone_tracker.stop()
        return

    # Extract initial position and orientation
    initial_position = initial_robot_transform[:3, 3]
    initial_rotation = R.from_matrix(initial_robot_transform[:3, :3])
    
    # Initialize current orientation
    current_rotation = initial_rotation
    locked_quat = initial_rotation.as_quat()  # [x, y, z, w]
    locked_orientation = [locked_quat[3], locked_quat[0], locked_quat[1], locked_quat[2]]  # [w, x, y, z]
    time.sleep(1.0)
    # Reset phone tracker
    phone_tracker.reset(initial_robot_transform)

    # Initialize interpolation state
    current_position = initial_position.copy()
    max_step = max_velocity * cycle_time  # Maximum position change per cycle
    max_rotation_step = max_rotation_velocity * cycle_time  # Maximum rotation change per cycle (radians)

    print("=" * 60)
    print("Smooth Position Teleoperation")
    print("=" * 60)
    print(f"Hand: {hand}")
    print(f"Scale: {scale}")
    print(f"Filter strength: {filter_strength} (0-1000)")
    print(f"Cycle time: {cycle_time*1000:.1f}ms")
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Max step per cycle: {max_step*1000:.1f}mm")
    print(f"Interpolation alpha: {interpolation_alpha}")
    print(f"Rotation enabled: {enable_rotation}")
    if enable_rotation:
        print(f"Rotation interpolation: {rotation_interpolation}")
        print(f"Max rotation velocity: {max_rotation_velocity:.2f} rad/s ({np.degrees(max_rotation_velocity):.1f} deg/s)")
        print(f"Max rotation step: {np.degrees(max_rotation_step):.2f} degrees")
    print(f"Hand enabled: {hand_enabled}")
    if hand_enabled:
        print(f"Retargeting enabled: {use_retargeting}")
    print(f"Initial position: [{initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f}]")
    print("\nPress CTRL+C to stop")
    print("=" * 60)

    try:
        loop_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.time()
            
            # Get target pose from phone
            target_pose, _ = phone_tracker.get_target_pose()
            
            if target_pose is not None:
                # Handle finger control with retargeting
                if use_retargeting and hand_enabled:
                    landmarks, world_landmarks = phone_tracker.get_landmarks()
                    if landmarks is not None and world_landmarks is not None:
                        qpos = retarget.get_qpos(landmarks, world_landmarks)
                        
                        # Map joint positions to finger commands
                        index = np.clip(qpos[0], -1.46, 0.0) * -INV_1_46
                        pinky = np.clip(qpos[2], -1.46, 0.0) * -INV_1_46
                        middle = np.clip(qpos[4], -1.46, 0.0) * -INV_1_46
                        rings = np.clip(qpos[6], -1.46, 0.0) * -INV_1_46
                        thumb_1 = np.clip(qpos[8], 0.0, 1.57) * INV_1_57
                        thumb_2 = np.clip(qpos[9], -1.46, 0.0) * -INV_1_46
                        
                        # Send finger commands
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
                
                # Extract target position from phone
                target_position = np.array(target_pose[:3])
                
                # Calculate desired delta
                delta = target_position - current_position
                
                # Apply interpolation (exponential smoothing)
                delta_smooth = interpolation_alpha * delta
                
                # Limit velocity (clip delta to max step size)
                delta_norm = np.linalg.norm(delta_smooth)
                if delta_norm > max_step:
                    delta_smooth = delta_smooth * (max_step / delta_norm)
                
                # Update current position
                current_position = current_position + delta_smooth
                
                # Handle rotation
                if enable_rotation:
                    # Extract target orientation from phone (quaternion)
                    target_quat = target_pose[3:]  # [qw, qx, qy, qz]
                    target_rotation = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])  # [x, y, z, w]
                    
                    # Calculate rotation difference (angle in radians)
                    rotation_diff = current_rotation.inv() * target_rotation
                    rotation_angle = rotation_diff.magnitude()  # Angle in radians
                    
                    # Calculate desired interpolation factor based on max rotation velocity
                    # Start with base interpolation
                    desired_angle_change = rotation_interpolation * rotation_angle
                    
                    # Limit to max rotation step
                    if desired_angle_change > max_rotation_step:
                        # Scale down the interpolation factor to respect velocity limit
                        actual_interpolation = max_rotation_step / rotation_angle if rotation_angle > 0 else 0
                    else:
                        actual_interpolation = rotation_interpolation
                    
                    # Spherical linear interpolation (SLERP) for smooth rotation
                    # Use scipy's built-in slerp which properly handles quaternions
                    from scipy.spatial.transform import Slerp
                    key_times = [0, 1]
                    key_rots = R.from_quat([current_rotation.as_quat(), target_rotation.as_quat()])
                    slerp = Slerp(key_times, key_rots)
                    current_rotation = slerp(actual_interpolation)
                    
                    # Get current orientation as quaternion
                    current_quat = current_rotation.as_quat()  # [x, y, z, w]
                    current_orientation = [current_quat[3], current_quat[0], current_quat[1], current_quat[2]]  # [w, x, y, z]
                else:
                    current_orientation = locked_orientation
                
                # Create command: [x, y, z, qw, qx, qy, qz]
                # position + quaternion orientation
                cmd = list(current_position) + list(current_orientation)
                
                # Send pose via pass-through with filtering
                # trajectory_mode=0: Full transparency mode
                # radio=filter_strength: Filter parameter (0-1000)
                # follow=True: High follow mode for better responsiveness
                ret = arm.rm_movep_canfd(
                    cmd,
                    follow=1,
                    trajectory_mode=0,  # Full transparency
                    radio=0
                )
                
                if ret != 0 and loop_count % 50 == 0:
                    print(f"Warning: rm_movep_canfd returned {ret}")
                    
                # Debug: print velocity occasionally
                if loop_count % 100 == 0 and delta_norm > 0.0001:
                    velocity = delta_norm / cycle_time
                    print(f"Velocity: {velocity:.3f} m/s, Delta: {delta_norm*1000:.2f}mm")
            
            loop_count += 1
            
            # Print statistics every 2 seconds
            stats_interval = int(2.0 / cycle_time)  # Every 2 seconds
            if loop_count % stats_interval == 0:
                elapsed = time.time() - start_time
                avg_freq = loop_count / elapsed
                print(f"Loop count: {loop_count}, Avg frequency: {avg_freq:.1f} Hz")
            
            # Maintain cycle time
            elapsed = time.time() - loop_start
            sleep_time = cycle_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_count % 100 == 0:
                print(f"Warning: Cycle overrun by {-sleep_time*1000:.2f}ms")

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")

    finally:
        # Return to initial pose
        print("Returning to initial position...")
        
        rot = R.from_matrix(initial_robot_transform[:3, :3]).as_quat()
        pos = initial_robot_transform[:3, 3]
        init_pose = list(pos) + [rot[3], rot[0], rot[1], rot[2]]
        
        # Use regular movel to return (blocking)
        from hand_kinematics_monitor import move_to_pose
        move_to_pose(arm, init_pose, speed=50)
        
        # Cleanup
        arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()
        phone_tracker.stop()
        
        print("Teleoperation stopped.")


if __name__ == '__main__':
    smooth_position_teleop(
        hand='right',
        scale=1.0,
        filter_strength=0,           # Adjust for smoothness (0-1000)
        cycle_time=0.05,             # 50ms = 20Hz
        max_velocity=0.5,            # m/s - reduce if motion is too jerky
        interpolation_alpha=0.2,     # 0-1 - lower = smoother but more lag
        enable_rotation=1,           # Enable rotation control
        rotation_interpolation=0.1,  # Not used anymore (we go as fast as allowed)
        max_rotation_velocity=10.0,  # rad/s - high value to allow fast rotation
        hand_enabled=1,              # Enable hand/finger control
        use_retargeting=1,           # Enable hand retargeting
    )
