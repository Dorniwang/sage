# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
JointMotionBenchmark with GapONet sim-to-real gap correction.

Extends simulation.py by adding GapONet delta correction to the target
joint positions before sending them to the simulated robot.
"""

import numpy as np
import torch

from .simulation import JointMotionBenchmark, log_message


class GapONetCorrector:
    """
    Lightweight GapONet JIT inference wrapper.
    Manages model history internally.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_dofs: int = 6,
        num_sensor_positions: int = 12,
        sensor_dim: int = 6,
        model_history_length: int = 4,
        model_history_dim: int = 18,
        joint_delta_scale: list = None,
    ):
        self.device = torch.device(device)
        self.num_dofs = num_dofs
        self.branch_dim = num_sensor_positions * sensor_dim
        self.model_history_length = model_history_length
        self.model_history_dim = model_history_dim

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        log_message(f"[GapONet] Loaded JIT model from {model_path}")

        self.model_history = torch.zeros(
            1, model_history_length, model_history_dim, device=self.device
        )

        # Per-joint delta scaling
        if joint_delta_scale is not None:
            self._delta_scale = torch.tensor(
                joint_delta_scale, device=self.device
            ).unsqueeze(0)
        else:
            self._delta_scale = torch.ones(1, num_dofs, device=self.device)

    def reset(self):
        """Reset model history (call at start of each motion)."""
        self.model_history.zero_()

    @torch.no_grad()
    def step(self, joint_pos, joint_vel, target_action, last_cmd=None):
        """
        Compute corrected action.

        Args:
            joint_pos:     [num_dofs] numpy array, current joint positions (rad)
            joint_vel:     [num_dofs] numpy array, current joint velocities (rad/s)
            target_action: [num_dofs] numpy array, target command from motion file (rad)
            last_cmd:      [num_dofs] numpy array, last command sent. If None, uses target_action.

        Returns:
            corrected_action: [num_dofs] numpy array, corrected command (rad)
        """
        jp = torch.tensor(joint_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        jv = torch.tensor(joint_vel, dtype=torch.float32, device=self.device).unsqueeze(0)
        ta = torch.tensor(target_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        if last_cmd is None:
            jt = ta
        else:
            jt = torch.tensor(last_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)

        # model_obs = [joint_pos, model_history.flatten()]
        model_obs = torch.cat([jp, self.model_history.flatten(1, 2)], dim=1)

        # Update history
        self.model_history = self.model_history.roll(1, dims=1)
        self.model_history[:, 0, :] = torch.cat([jp, jv, jt], dim=1)

        # branch_input (zeros, overwritten by sensor model inside JIT)
        branch_input = torch.zeros(1, self.branch_dim, device=self.device)

        # trunk_input = target_action
        trunk_input = ta.clone()

        # Forward
        delta = self.model(model_obs, branch_input, trunk_input)
        # print(delta)

        # Correct with per-joint scaling
        corrected = ta + delta * self._delta_scale
        return corrected.squeeze(0).cpu().numpy()


class JointMotionBenchmarkGapONet(JointMotionBenchmark):
    """
    JointMotionBenchmark with GapONet correction.

    Overrides run_benchmark() to apply GapONet delta correction
    to target joint positions before sending to the simulated robot.
    """

    def __init__(self, args):
        # GapONet-specific args
        self.gaponet_model_path = args.gaponet_model
        self.gaponet_device = getattr(args, 'gaponet_device', 'cpu')
        self.gaponet_enabled = self.gaponet_model_path is not None

        # Parse joint_delta_scale from args
        self.joint_delta_scale = None
        if hasattr(args, 'joint_delta_scale') and args.joint_delta_scale is not None:
            self.joint_delta_scale = [float(x) for x in args.joint_delta_scale.split(',')]

        # Initialize parent (sets up simulation)
        super().__init__(args)

        # Initialize GapONet after parent init
        self.gaponet = None
        if self.gaponet_enabled:
            self.gaponet = GapONetCorrector(
                model_path=self.gaponet_model_path,
                device=self.gaponet_device,
                joint_delta_scale=self.joint_delta_scale,
            )

    def run_benchmark(self):
        """Run benchmark with GapONet correction applied to target positions."""
        if not self.gaponet_enabled:
            log_message("GapONet not enabled, running original benchmark")
            return super().run_benchmark()

        # Load motion data
        log_message(f"Loading motion data from {self.motion_file}...")
        joint_angles, joint_names = self._load_motion()

        log_message(f"Physics dt: {self.physics_dt}, Rendering dt: {self.physics_dt * self.divisor}")
        log_message(f"GapONet correction: ENABLED")

        # Reset world and GapONet
        self.world.reset()
        self._config_controller()
        self.gaponet.reset()

        # ==================== Buffer phase (same as parent) ====================
        BUFFER_TIME = 5.0
        buffer_control_steps = int(BUFFER_TIME / self.control_dt)

        initial_joint_positions = self.robot.get_joint_positions(joint_indices=self.joint_indices)[0]
        motion_start_positions = np.array([joint_angles[j][0] for j in range(len(self.joint_names))])

        log_message(f"Starting initialization phase with {BUFFER_TIME}s buffer...")

        for buffer_counter in range(buffer_control_steps * self.divisor):
            control_step = int(buffer_counter / self.divisor)

            if buffer_counter % self.divisor == 0:
                alpha = control_step / buffer_control_steps
                interpolated_positions = (1 - alpha) * initial_joint_positions + alpha * motion_start_positions

                target_pos = np.zeros((1, self.robot.num_dof), dtype=np.float32)
                for j, idx in enumerate(self.joint_indices):
                    target_pos[0, idx] = interpolated_positions[j]
                self.robot.set_joint_position_targets(target_pos)

            self.world.step(True)
        buffer_end_time = self.world.current_time

        log_message(f"Buffer completed. Starting main motion with GapONet correction...")

        if self.record_video:
            self._init_video_writer()

        # ==================== Main motion with GapONet ====================
        last_cmd = None

        for counter in range(len(joint_angles[0]) * self.divisor):
            index = int(counter / self.divisor)
            adjusted_time = self.world.current_time - buffer_end_time

            if index >= len(joint_angles[0]):
                break

            # Apply GapONet correction at control frequency
            if counter % self.divisor == 0:
                # Get original command from motion file
                command_positions = np.array(
                    [joint_angles[k][index] for k in range(len(self.joint_names))],
                    dtype=np.float32,
                )

                # Get current robot state
                actual_positions = self.robot.get_joint_positions(joint_indices=self.joint_indices)[0]
                actual_velocities = self.robot.get_joint_velocities(joint_indices=self.joint_indices)[0]

                # GapONet: correct the command
                corrected_positions = self.gaponet.step(
                    joint_pos=actual_positions,
                    joint_vel=actual_velocities,
                    target_action=command_positions,
                    last_cmd=last_cmd,
                )
                last_cmd = corrected_positions.copy()

                # Set corrected positions as target
                target_pos = np.zeros((1, self.robot.num_dof), dtype=np.float32)
                for j, idx in enumerate(self.joint_indices):
                    target_pos[0, idx] = corrected_positions[j]
                self.robot.set_joint_position_targets(target_pos)

            # Log state (log original command AND actual state for comparison)
            command_positions = np.array([joint_angles[k][index] for k in range(len(self.joint_names))])
            actual_positions = self.robot.get_joint_positions(joint_indices=self.joint_indices)[0]
            actual_velocities = self.robot.get_joint_velocities(joint_indices=self.joint_indices)[0]
            actual_efforts = self.robot.get_measured_joint_efforts(joint_indices=self.joint_indices)[0]

            self._log_state(
                time=adjusted_time,
                command_positions=command_positions,
                actual_positions=actual_positions,
                actual_velocities=actual_velocities,
                actual_efforts=actual_efforts,
            )

            if self.record_video:
                from .simulation import capture_viewport_to_buffer
                capture_viewport_to_buffer(self.viewport_api, self._capture_video_fn)

            self.world.step(True)

        if self.record_video:
            self.video_writer.release()

        log_message(
            f"Motion completed with GapONet correction in {counter+1} physics steps. "
            f"Results saved to {self.sim_output_folder}"
        )
