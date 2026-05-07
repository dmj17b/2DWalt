import numpy as np
import mujoco


class StairBuilder:
    def __init__(self,
                 size = [20.0, 5.0, 5.0, 0.1],  # [x_half_size, y_half_size, z_max, z_bottom]
                 nrow = 2,
                 ncol = 4096,):
        self.size = size
        self.nrow = nrow
        self.ncol = ncol
        self.z_max = size[2]

        self.hf_resolution = (2.0*size[0]) / (ncol-1)  # Horizontal resolution of hf
        self.hf_data = np.zeros((nrow, ncol), dtype=np.float32) # Initialize hfdata

        self.current_x = 0.0
        self.current_z_norm = 0.0

    def set_starting_height(self, normalized_height):
        """Manually override the Z cursor (useful before descending stairs)."""
        if normalized_height > 1.0 or normalized_height < 0.0:
            raise ValueError("Height exceeds MuJoCo bounds [0, 1].")
        self.current_z_norm = normalized_height

    def add_flat(self, length):
        """Advances the X cursor while maintaining the current height."""
        start_idx = int(self.current_x / self.hf_resolution)
        end_idx = int((self.current_x + length) / self.hf_resolution)
        end_idx = min(end_idx, self.ncol)
        
        self.hf_data[:, start_idx:end_idx] = self.current_z_norm
        self.current_x += length

    def add_stairs(self, rise, run, num_steps, direction):
        """
        direction: 1.0 for ascending, -1.0 for descending.
        Only requires local stair geometry parameters.
        """
        step_height_delta = rise / self.z_max
        
        for _ in range(num_steps):
            step_start_x = self.current_x
            step_end_x = self.current_x + run
            
            start_idx = int(step_start_x / self.hf_resolution)
            end_idx = int(step_end_x / self.hf_resolution)
            end_idx = min(end_idx, self.ncol)
            
            self.hf_data[:, start_idx:end_idx] = self.current_z_norm
            
            # Advance the spatial cursors for the next iteration
            self.current_x += run
            self.current_z_norm += direction * step_height_delta

    def finalize(self):
        """Applies the calibration spike and returns the flattened array."""
        self.hf_data[-1, -1] = 1.0  # Forces max(H) to 1.0
        self.hf_data[-1, -2] = 0.0  # Forces min(H) to 0.0
        return self.hf_data.flatten()