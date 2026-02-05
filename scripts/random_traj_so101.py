# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SO-101 Random Trajectory Collector

Usage:
    python so101_random_trajectory_collector.py \
        --output-dir ./random_data \
        --port /dev/ttyACM0 \
        --frames-per-trajectory 300 \
        --num-trajectories 10 \
        --control-freq 50
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")


# Import from the original collector
from sage.real_so101.so101_lerobot_collector import (
    So101Collector,
    save_sage_format,
    SO101_JOINT_NAMES,
    SAGE_JOINT_NAMES,
    CALIBRATION_PATH,
    PACKED_POSITION,
    interpolate_motion,
)


def log_message(message):
    """Format and print log messages with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[RandomTraj][{current_time}] {message}")


# Maximum velocity limits (rad/s for joints, normalized/s for gripper)
MAX_JOINT_VELOCITY_RAD_S = 1.5  # ~86 deg/s, conservative limit
MAX_GRIPPER_VELOCITY_S = 0.5   # 50%/s for gripper
MIN_MOVE_DURATION = 0.5        # Minimum duration for any move (seconds)
MAX_MOVE_DURATION = 10.0       # Maximum duration for any move (seconds)


class RandomTrajectoryGenerator:
    """
    Random trajectory generator for SO-101 robot arm.
    
    Uses calibration data to determine joint limits and generates
    smooth random trajectories within safe operating range.
    """
    
    # Encoder ticks per radian (STS3215: 4096 ticks = 2π rad)
    TICKS_PER_RADIAN = 4096 / (2 * np.pi)
    
    def __init__(self, calibration_path=None, safety_margin=0.1):
        """
        Initialize trajectory generator.
        
        Args:
            calibration_path: Path to calibration JSON file
            safety_margin: Fraction of range to avoid at limits (0.1 = 10% margin on each side)
        """
        if calibration_path is None:
            calibration_path = CALIBRATION_PATH
        
        if not Path(calibration_path).exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        
        with open(calibration_path) as f:
            self.calib_data = json.load(f)
        
        self.joint_names = SO101_JOINT_NAMES
        self.safety_margin = safety_margin
        
        # Compute joint limits in radians
        self.joint_limits_rad = {}
        self.joint_limits_normalized = {}  # For gripper (0-1)
        
        self._compute_joint_limits()
        
    def _compute_joint_limits(self):
        """Compute joint limits in radians from calibration data."""
        log_message("Computing joint limits from calibration:")
        
        for name in self.joint_names:
            calib = self.calib_data[name]
            range_min = calib["range_min"]
            range_max = calib["range_max"]
            
            # Apply safety margin
            range_span = range_max - range_min
            safe_min = range_min + range_span * self.safety_margin
            safe_max = range_max - range_span * self.safety_margin
            
            if name == "gripper":
                # Gripper uses 0-1 normalized range
                self.joint_limits_normalized[name] = (
                    self.safety_margin,
                    1.0 - self.safety_margin
                )
                log_message(f"  {name}: [{self.safety_margin:.2f}, {1.0-self.safety_margin:.2f}] (normalized)")
            else:
                # Convert encoder to radians
                # Mid point of range = 0 radians
                mid = (range_min + range_max) / 2
                rad_min = (safe_min - mid) / self.TICKS_PER_RADIAN
                rad_max = (safe_max - mid) / self.TICKS_PER_RADIAN
                
                self.joint_limits_rad[name] = (rad_min, rad_max)
                log_message(f"  {name}: [{np.rad2deg(rad_min):.1f}°, {np.rad2deg(rad_max):.1f}°] "
                           f"= [{rad_min:.3f}, {rad_max:.3f}] rad")
    
    def generate_random_waypoints(self, num_waypoints=5):
        """
        Generate random waypoints within joint limits.
        
        Args:
            num_waypoints: Number of waypoints to generate
            
        Returns:
            waypoints: Array of shape (num_waypoints, num_joints) in radians
        """
        waypoints = np.zeros((num_waypoints, len(self.joint_names)))
        
        for j, name in enumerate(self.joint_names):
            if name == "gripper":
                limits = self.joint_limits_normalized[name]
            else:
                limits = self.joint_limits_rad[name]
            
            # Random positions within limits
            waypoints[:, j] = np.random.uniform(limits[0], limits[1], num_waypoints)
        
        return waypoints
    
    def interpolate_waypoints(self, waypoints, frames_per_segment=50, smoothing=True):
        """
        Interpolate between waypoints to create smooth trajectory.
        
        Args:
            waypoints: Array of shape (num_waypoints, num_joints)
            frames_per_segment: Frames between each waypoint pair
            smoothing: If True, use cubic interpolation for smoother motion
            
        Returns:
            trajectory: Array of shape (total_frames, num_joints)
        """
        from scipy.interpolate import interp1d
        
        num_waypoints = len(waypoints)
        num_joints = waypoints.shape[1]
        total_frames = (num_waypoints - 1) * frames_per_segment + 1
        
        # Time points for waypoints
        t_waypoints = np.linspace(0, 1, num_waypoints)
        t_trajectory = np.linspace(0, 1, total_frames)
        
        trajectory = np.zeros((total_frames, num_joints))
        
        for j in range(num_joints):
            if smoothing and num_waypoints >= 4:
                # Cubic interpolation for smooth motion
                f = interp1d(t_waypoints, waypoints[:, j], kind='cubic')
            else:
                # Linear interpolation
                f = interp1d(t_waypoints, waypoints[:, j], kind='linear')
            
            trajectory[:, j] = f(t_trajectory)
        
        return trajectory
    
    def generate_trajectory(self, n_frames, num_waypoints=None, style="smooth"):
        """
        Generate a random trajectory with n frames.
        
        Args:
            n_frames: Total number of frames in trajectory
            num_waypoints: Number of random waypoints (auto-calculated if None)
            style: Trajectory style - "smooth", "random_walk", or "sinusoidal"
            
        Returns:
            trajectory: Array of shape (n_frames, num_joints) in radians
        """
        if style == "smooth":
            return self._generate_smooth_trajectory(n_frames, num_waypoints)
        elif style == "random_walk":
            return self._generate_random_walk(n_frames)
        elif style == "sinusoidal":
            return self._generate_sinusoidal(n_frames)
        else:
            raise ValueError(f"Unknown style: {style}")
    
    def _generate_smooth_trajectory(self, n_frames, num_waypoints=None):
        """Generate smooth trajectory through random waypoints."""
        # Auto-determine waypoints: roughly one every 2 seconds at 50Hz
        if num_waypoints is None:
            num_waypoints = max(3, n_frames // 100 + 1)
        
        frames_per_segment = n_frames // (num_waypoints - 1)
        
        # Generate random waypoints
        waypoints = self.generate_random_waypoints(num_waypoints)
        
        # Interpolate
        trajectory = self.interpolate_waypoints(waypoints, frames_per_segment, smoothing=True)
        
        # Trim or pad to exact n_frames
        if len(trajectory) > n_frames:
            trajectory = trajectory[:n_frames]
        elif len(trajectory) < n_frames:
            # Repeat last frame
            padding = np.tile(trajectory[-1:], (n_frames - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])
        
        return trajectory
    
    def _generate_random_walk(self, n_frames, max_step_rad=0.02):
        """
        Generate random walk trajectory.
        
        Args:
            n_frames: Number of frames
            max_step_rad: Maximum step size in radians per frame
        """
        trajectory = np.zeros((n_frames, len(self.joint_names)))
        
        # Start at random position
        for j, name in enumerate(self.joint_names):
            if name == "gripper":
                limits = self.joint_limits_normalized[name]
            else:
                limits = self.joint_limits_rad[name]
            trajectory[0, j] = np.random.uniform(limits[0], limits[1])
        
        # Random walk
        for i in range(1, n_frames):
            for j, name in enumerate(self.joint_names):
                if name == "gripper":
                    limits = self.joint_limits_normalized[name]
                    step = np.random.uniform(-0.02, 0.02)  # Smaller step for gripper
                else:
                    limits = self.joint_limits_rad[name]
                    step = np.random.uniform(-max_step_rad, max_step_rad)
                
                new_val = trajectory[i-1, j] + step
                trajectory[i, j] = np.clip(new_val, limits[0], limits[1])
        
        return trajectory
    
    def _generate_sinusoidal(self, n_frames, min_period_frames=100, max_period_frames=500):
        """
        Generate sinusoidal trajectory with random frequencies and phases.
        
        Args:
            n_frames: Number of frames
            min_period_frames: Minimum period in frames
            max_period_frames: Maximum period in frames
        """
        trajectory = np.zeros((n_frames, len(self.joint_names)))
        t = np.arange(n_frames)
        
        for j, name in enumerate(self.joint_names):
            if name == "gripper":
                limits = self.joint_limits_normalized[name]
            else:
                limits = self.joint_limits_rad[name]
            
            # Random frequency and phase
            period = np.random.uniform(min_period_frames, max_period_frames)
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Sinusoidal motion within limits
            mid = (limits[0] + limits[1]) / 2
            amplitude = (limits[1] - limits[0]) / 2 * 0.8  # 80% of range
            
            trajectory[:, j] = mid + amplitude * np.sin(2 * np.pi * t / period + phase)
        
        return trajectory
    
    def check_trajectory_velocity(self, trajectory, control_freq, max_vel_rad_s=None):
        """
        Check if trajectory velocities are within safe limits.
        
        Args:
            trajectory: Array of shape (n_frames, num_joints)
            control_freq: Control frequency in Hz
            max_vel_rad_s: Maximum allowed velocity in rad/s (default: MAX_JOINT_VELOCITY_RAD_S)
            
        Returns:
            is_safe: True if all velocities are within limits
            max_velocities: Dict of max velocities per joint
            violations: List of (frame_idx, joint_name, velocity) tuples for violations
        """
        if max_vel_rad_s is None:
            max_vel_rad_s = MAX_JOINT_VELOCITY_RAD_S
        
        dt = 1.0 / control_freq
        velocities = np.diff(trajectory, axis=0) / dt
        
        max_velocities = {}
        violations = []
        is_safe = True
        
        for j, name in enumerate(self.joint_names):
            vel = velocities[:, j]
            max_vel = np.abs(vel).max()
            max_velocities[name] = max_vel
            
            if name == "gripper":
                limit = MAX_GRIPPER_VELOCITY_S
            else:
                limit = max_vel_rad_s
            
            if max_vel > limit:
                is_safe = False
                # Find frames with violations
                violation_frames = np.where(np.abs(vel) > limit)[0]
                for frame in violation_frames[:5]:  # Report first 5
                    violations.append((frame, name, vel[frame]))
        
        return is_safe, max_velocities, violations
    
    def limit_trajectory_velocity(self, trajectory, control_freq, max_vel_rad_s=None):
        """
        Limit trajectory velocities by interpolating more frames where needed.
        
        Args:
            trajectory: Original trajectory
            control_freq: Control frequency
            max_vel_rad_s: Maximum velocity limit
            
        Returns:
            Limited trajectory with safe velocities
        """
        if max_vel_rad_s is None:
            max_vel_rad_s = MAX_JOINT_VELOCITY_RAD_S
        
        dt = 1.0 / control_freq
        result = [trajectory[0]]
        
        for i in range(1, len(trajectory)):
            prev_pos = result[-1]
            next_pos = trajectory[i]
            
            # Calculate required velocity for each joint
            diff = next_pos - prev_pos
            
            # Find the joint that needs the most time
            max_time_needed = dt  # At least one frame
            for j, name in enumerate(self.joint_names):
                if name == "gripper":
                    limit = MAX_GRIPPER_VELOCITY_S
                else:
                    limit = max_vel_rad_s
                
                if limit > 0:
                    time_needed = abs(diff[j]) / limit
                    max_time_needed = max(max_time_needed, time_needed)
            
            # Calculate how many frames we need
            n_interp_frames = max(1, int(np.ceil(max_time_needed / dt)))
            
            if n_interp_frames == 1:
                result.append(next_pos)
            else:
                # Interpolate
                for k in range(1, n_interp_frames + 1):
                    alpha = k / n_interp_frames
                    interp_pos = (1 - alpha) * prev_pos + alpha * next_pos
                    result.append(interp_pos)
        
        return np.array(result)


def compute_move_duration(current_pos, target_pos, max_vel_rad_s=None, joint_names=None):
    """
    Compute the duration needed to move from current to target position
    at a safe constant velocity.
    
    Args:
        current_pos: Current joint positions (numpy array)
        target_pos: Target joint positions (numpy array)
        max_vel_rad_s: Maximum velocity in rad/s
        joint_names: List of joint names (to identify gripper)
        
    Returns:
        duration: Required duration in seconds
        max_distance: Maximum distance to travel (rad or normalized)
        limiting_joint: Name of the joint that limits the speed
    """
    if max_vel_rad_s is None:
        max_vel_rad_s = MAX_JOINT_VELOCITY_RAD_S
    if joint_names is None:
        joint_names = SO101_JOINT_NAMES
    
    max_time = MIN_MOVE_DURATION
    max_distance = 0
    limiting_joint = None
    
    for j, name in enumerate(joint_names):
        distance = abs(target_pos[j] - current_pos[j])
        
        if name == "gripper":
            vel_limit = MAX_GRIPPER_VELOCITY_S
        else:
            vel_limit = max_vel_rad_s
        
        if vel_limit > 0:
            time_needed = distance / vel_limit
            if time_needed > max_time:
                max_time = time_needed
                max_distance = distance
                limiting_joint = name
    
    # Clamp duration
    duration = min(max(max_time, MIN_MOVE_DURATION), MAX_MOVE_DURATION)
    
    return duration, max_distance, limiting_joint


class RandomTrajectoryCollector:
    """
    Collector that generates and executes random trajectories.
    """
    
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000, safety_margin=0.1,
                 max_velocity_rad_s=None):
        """
        Initialize random trajectory collector.
        
        Args:
            port: Serial port for robot
            baudrate: Communication baudrate
            safety_margin: Safety margin for joint limits
            max_velocity_rad_s: Maximum joint velocity limit
        """
        self.collector = So101Collector(port=port, baudrate=baudrate)
        self.generator = RandomTrajectoryGenerator(safety_margin=safety_margin)
        self.max_velocity = max_velocity_rad_s if max_velocity_rad_s else MAX_JOINT_VELOCITY_RAD_S
        
        self.trajectory_count = 0
        
    def collect_random_trajectory(
        self,
        n_frames,
        output_dir,
        control_freq=50,
        slowdown_factor=1,
        style="smooth",
        auto_confirm=False,
    ):
        """
        Generate and collect a random trajectory.
        
        Args:
            n_frames: Number of frames in trajectory
            output_dir: Output directory for data
            control_freq: Control frequency in Hz
            slowdown_factor: Factor to slow down motion
            style: Trajectory generation style
            auto_confirm: If True, skip manual confirmation
            
        Returns:
            success: True if collection completed successfully
        """
        self.trajectory_count += 1
        motion_name = f"random_traj_{self.trajectory_count:04d}_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log_message(f"=== Trajectory {self.trajectory_count} ===")
        log_message(f"Generating {n_frames} frames with style '{style}'...")
        
        # Generate trajectory
        trajectory = self.generator.generate_trajectory(n_frames, style=style)
        
        log_message(f"Generated trajectory shape: {trajectory.shape}")
        
        # Print trajectory statistics
        for j, name in enumerate(self.generator.joint_names):
            vals = trajectory[:, j]
            if name == "gripper":
                log_message(f"  {name}: [{vals.min():.3f}, {vals.max():.3f}] (normalized)")
            else:
                log_message(f"  {name}: [{np.rad2deg(vals.min()):.1f}°, {np.rad2deg(vals.max()):.1f}°]")
        
        try:
            # Enable torque
            self.collector.enable_torque()
            time.sleep(0.3)
            
            # Execute motion and collect data
            if auto_confirm:
                # Skip safety check for automated collection
                log_message("Auto-confirm enabled, skipping safety check...")
                
                # Move to start position with distance-aware velocity control
                self._move_to_start_position(trajectory[0])
                time.sleep(0.3)
                
                # Execute motion
                log_message(f"Starting motion execution: {n_frames} frames at {control_freq}Hz")
                command_times, command_positions = self._execute_motion(
                    trajectory, control_freq, slowdown_factor
                )
            else:
                command_times, command_positions = self.collector.collect_motion(
                    trajectory, control_freq=control_freq, slowdown_factor=slowdown_factor
                )
            
            if len(command_times) == 0:
                log_message("Motion cancelled or failed.")
                return False
            
            # Save data
            save_sage_format(
                output_dir=output_dir,
                motion_name=motion_name,
                joint_names=SAGE_JOINT_NAMES,
                command_times=command_times,
                command_positions=command_positions,
                collected_data=self.collector.collected_data,
            )
            
            log_message(f"Trajectory {self.trajectory_count} collected successfully!")
            return True
            
        except Exception as e:
            log_message(f"Error during collection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _move_to_start_position(self, target_pos):
        """
        Move to trajectory start position with distance-aware velocity control.
        
        Args:
            target_pos: Target start position
        """
        # Read current position
        current_pos, _, _ = self.collector.read_state()
        
        # Compute required duration
        duration, max_dist, limiting_joint = compute_move_duration(
            current_pos, target_pos,
            max_vel_rad_s=self.max_velocity,
            joint_names=self.generator.joint_names
        )
        
        log_message(f"Moving to start position...")
        log_message(f"  Max distance: {np.rad2deg(max_dist):.1f}° ({limiting_joint})")
        log_message(f"  Duration: {duration:.2f}s")
        
        # Execute constant velocity move
        self._move_to_position_constant_velocity(current_pos, target_pos, duration)
        log_message("  Reached start position.")
    
    def _execute_motion(self, motion_seq, control_freq, slowdown_factor):
        """Execute motion without interactive confirmation."""
        # First, check and limit trajectory velocity
        is_safe, max_vels, violations = self.generator.check_trajectory_velocity(
            motion_seq, control_freq, self.max_velocity
        )
        
        if not is_safe:
            log_message("WARNING: Trajectory has velocity violations, limiting...")
            for frame, joint, vel in violations[:3]:
                log_message(f"  Frame {frame}, {joint}: {np.rad2deg(vel):.1f}°/s")
            motion_seq = self.generator.limit_trajectory_velocity(
                motion_seq, control_freq, self.max_velocity
            )
            log_message(f"Trajectory extended from {len(motion_seq)} to {len(motion_seq)} frames")
        
        # Log max velocities
        log_message("Max velocities in trajectory:")
        for name, vel in max_vels.items():
            if name == "gripper":
                log_message(f"  {name}: {vel:.2f}/s")
            else:
                log_message(f"  {name}: {np.rad2deg(vel):.1f}°/s")
        
        n_frames = motion_seq.shape[0]
        loop_dt = 1.0 / control_freq * slowdown_factor
        
        command_times = []
        command_positions = []
        
        # Clear previous data
        self.collector.collected_data = {
            "time": [],
            "positions": [],
            "velocities": [],
            "currents": [],
        }
        
        start_time = time.monotonic()
        self.collector.start_monotonic = start_time
        
        for i in range(n_frames):
            loop_start = time.monotonic()
            t = loop_start - start_time
            
            # Send command
            target_pos = motion_seq[i]
            self.collector.write_positions(target_pos)
            
            # Record command
            command_times.append(t)
            command_positions.append(target_pos.copy())
            
            # Read and store state
            positions, velocities, currents = self.collector.read_state()
            self.collector.collected_data["time"].append(t)
            self.collector.collected_data["positions"].append(positions)
            self.collector.collected_data["velocities"].append(velocities)
            self.collector.collected_data["currents"].append(currents)
            
            # Timing control
            loop_elapsed = time.monotonic() - loop_start
            sleep_time = max(0, loop_dt - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Progress logging
            if i % 100 == 0:
                log_message(f"  Frame {i}/{n_frames}")
        
        log_message(f"Motion completed: {n_frames} frames in {time.monotonic() - start_time:.2f}s")
        
        # Return to packed position with distance-aware velocity control
        self._return_to_packed_position()
        
        return command_times, command_positions
    
    def _return_to_packed_position(self):
        """
        Return to packed position with distance-aware constant velocity control.
        Computes duration based on the maximum joint distance to ensure uniform speed.
        """
        # Read current position
        current_pos, _, _ = self.collector.read_state()
        target_pos = PACKED_POSITION
        
        # Compute required duration based on distance and max velocity
        duration, max_dist, limiting_joint = compute_move_duration(
            current_pos, target_pos, 
            max_vel_rad_s=self.max_velocity,
            joint_names=self.generator.joint_names
        )
        
        # Log movement info
        log_message(f"Returning to packed position...")
        log_message(f"  Max distance: {np.rad2deg(max_dist):.1f}° ({limiting_joint})")
        log_message(f"  Duration: {duration:.2f}s (velocity limit: {np.rad2deg(self.max_velocity):.1f}°/s)")
        
        # Execute smooth constant-velocity move
        self._move_to_position_constant_velocity(current_pos, target_pos, duration)
        
        log_message("Returned to packed position.")
    
    def _move_to_position_constant_velocity(self, start_pos, target_pos, duration, control_freq=50):
        """
        Move from start to target position with constant velocity interpolation.
        
        Args:
            start_pos: Starting positions
            target_pos: Target positions
            duration: Total duration for the move
            control_freq: Control frequency for interpolation
        """
        num_steps = max(1, int(duration * control_freq))
        dt = duration / num_steps
        
        start_time = time.monotonic()
        
        for step in range(num_steps + 1):
            loop_start = time.monotonic()
            
            # Linear interpolation (constant velocity)
            alpha = step / num_steps
            interpolated = (1 - alpha) * start_pos + alpha * target_pos
            
            # Send command
            self.collector.write_positions(interpolated)
            
            # Timing control
            loop_elapsed = time.monotonic() - loop_start
            sleep_time = max(0, dt - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Ensure final position
        self.collector.write_positions(target_pos)
        
        actual_duration = time.monotonic() - start_time
        log_message(f"  Move completed in {actual_duration:.2f}s")
    
    def run_continuous_collection(
        self,
        n_frames,
        output_dir,
        num_trajectories=None,
        control_freq=50,
        slowdown_factor=1,
        style="smooth",
        pause_between=2.0,
        auto_confirm=False,
    ):
        """
        Continuously collect random trajectories.
        
        Args:
            n_frames: Frames per trajectory
            output_dir: Output directory
            num_trajectories: Number of trajectories to collect (None = infinite)
            control_freq: Control frequency
            slowdown_factor: Slowdown factor
            style: Trajectory style
            pause_between: Pause between trajectories in seconds
            auto_confirm: Skip manual confirmation
        """
        log_message("=" * 60)
        log_message("RANDOM TRAJECTORY COLLECTION")
        log_message("=" * 60)
        log_message(f"Frames per trajectory: {n_frames}")
        log_message(f"Number of trajectories: {num_trajectories if num_trajectories else 'infinite'}")
        log_message(f"Control frequency: {control_freq} Hz")
        log_message(f"Style: {style}")
        log_message(f"Output directory: {output_dir}")
        log_message("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        collected = 0
        
        try:
            while True:
                # Check if we've collected enough
                if num_trajectories is not None and collected >= num_trajectories:
                    log_message(f"Collected {collected} trajectories. Done!")
                    break
                
                # Collect trajectory
                success = self.collect_random_trajectory(
                    n_frames=n_frames,
                    output_dir=output_dir,
                    control_freq=control_freq,
                    slowdown_factor=slowdown_factor,
                    style=style,
                    auto_confirm=auto_confirm,
                )
                
                if success:
                    collected += 1
                    log_message(f"Progress: {collected}/{num_trajectories if num_trajectories else '∞'}")
                
                # Pause between trajectories
                if pause_between > 0:
                    log_message(f"Pausing for {pause_between}s before next trajectory...")
                    time.sleep(pause_between)
                    
        except KeyboardInterrupt:
            log_message("\nCollection interrupted by user.")
        
        log_message(f"Total trajectories collected: {collected}")
    
    def close(self):
        """Close the collector."""
        self.collector.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SO-101 Random Trajectory Collector")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for collected data")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--frames-per-trajectory", "-n", type=int, default=300, 
                        help="Number of frames per trajectory")
    parser.add_argument("--num-trajectories", type=int, default=None,
                        help="Number of trajectories to collect (default: infinite)")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency Hz")
    parser.add_argument("--slowdown", type=int, default=1, help="Slowdown factor")
    parser.add_argument("--style", type=str, default="smooth", 
                        choices=["smooth", "random_walk", "sinusoidal"],
                        help="Trajectory generation style")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Pause between trajectories in seconds")
    parser.add_argument("--safety-margin", type=float, default=0.1,
                        help="Safety margin for joint limits (0.1 = 10% on each side)")
    parser.add_argument("--max-velocity", type=float, default=1.5,
                        help="Maximum joint velocity in rad/s (default: 1.5 = ~86 deg/s)")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Skip manual confirmation (use with caution!)")
    
    args = parser.parse_args()
    
    log_message(f"Max velocity limit: {args.max_velocity:.2f} rad/s ({np.rad2deg(args.max_velocity):.1f}°/s)")
    
    collector = RandomTrajectoryCollector(
        port=args.port,
        safety_margin=args.safety_margin,
        max_velocity_rad_s=args.max_velocity,
    )
    
    try:
        collector.run_continuous_collection(
            n_frames=args.frames_per_trajectory,
            output_dir=args.output_dir,
            num_trajectories=args.num_trajectories,
            control_freq=args.control_freq,
            slowdown_factor=args.slowdown,
            style=args.style,
            pause_between=args.pause,
            auto_confirm=args.auto_confirm,
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
