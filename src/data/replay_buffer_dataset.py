import torch
import pickle
from torch.utils.data import Dataset
import os
import h5py


class ReplayBufferDataset(Dataset):
    """
    Dataset for loading replay buffer data for training sequential decision-making modules.
    
    Args:
        buffer_path (str): Path to the replay buffer pickle file
        state_shape (tuple): Expected shape of state tensors (C, H, W)
        mode (str): Training mode - 'color_only', 'selection_color', or 'end_to_end'
        num_samples (int, optional): Number of samples to use (for testing). If None, uses all data.
    """
    
    def __init__(self, buffer_path, state_shape, mode='color_only', num_samples=None):
        self.buffer_path = buffer_path
        self.state_shape = state_shape
        self.mode = mode
        
        # Load or generate buffer data
        if os.path.exists(buffer_path):
            print(f"Loading replay buffer from {buffer_path}")
            if buffer_path.endswith('.hdf5'):
                self.buffer = self._load_hdf5_buffer(buffer_path)
            elif buffer_path.endswith('.pkl'):
                with open(buffer_path, 'rb') as f:
                    self.buffer = pickle.load(f)
            elif buffer_path.endswith('.pt'):
                import time
                start_time = time.time()
                buffer_dict = torch.load(buffer_path, map_location='cpu')
                
                # Convert dictionary format to list of transitions
                if isinstance(buffer_dict, dict) and 'state' in buffer_dict:
                    # This is a dictionary format with arrays for each field
                    num_transitions = len(buffer_dict['state'])
                    self.buffer = []
                    
                    for i in range(num_transitions):
                        transition = {
                            'state': buffer_dict['state'][i],
                            'action': buffer_dict['action'][i],
                            'next_state': buffer_dict['next_state'][i],
                            'target_state': buffer_dict['target_state'][i],
                            'reward': buffer_dict['reward'][i],
                            'done': buffer_dict['done'][i],
                            'transition_type': buffer_dict['transition_type'][i],
                            'shape_h': buffer_dict['shape_h'][i],
                            'shape_w': buffer_dict['shape_w'][i],
                            'num_colors_grid': buffer_dict['num_colors_grid'][i],
                            'most_present_color': buffer_dict['most_present_color'][i],
                            'least_present_color': buffer_dict['least_present_color'][i],
                            'num_colors_grid_target': buffer_dict['num_colors_grid_target'][i],
                            'most_present_color_target': buffer_dict['most_present_color_target'][i],
                            'least_present_color_target': buffer_dict['least_present_color_target'][i],
                            'shape_h_target': buffer_dict['shape_h_target'][i],
                            'shape_w_target': buffer_dict['shape_w_target'][i],
                            'shape_h_next': buffer_dict['shape_h_next'][i],
                            'shape_w_next': buffer_dict['shape_w_next'][i],
                            'num_colors_grid_next': buffer_dict['num_colors_grid_next'][i],
                            'most_present_color_next': buffer_dict['most_present_color_next'][i],
                            'least_present_color_next': buffer_dict['least_present_color_next'][i],
                            'step_distance_to_target': buffer_dict.get('step_distance_to_target', [0])[i]
                        }
                        self.buffer.append(transition)
                else:
                    # This is already in list format
                    self.buffer = buffer_dict
                
                end_time = time.time()
                print(f"Loaded {len(self.buffer)} transitions in {end_time - start_time:.2f} seconds")
            elif buffer_path.endswith('h5'):
                import time
                start_time = time.time()
                self.buffer = self._load_hdf5_buffer(buffer_path)
                end_time = time.time()
                print(f"Loaded {len(self.buffer)} transitions in {end_time - start_time:.2f} seconds")
            else:
                raise ValueError(f"Unsupported buffer file format: {buffer_path}. Please use .hdf5, .pkl, or .pt")
        else:
            print(f"ERROR: Buffer file {buffer_path} not found. Please provide a valid replay buffer file.")
            raise FileNotFoundError(f"Buffer file {buffer_path} not found.")
        
        # Limit samples if specified (for testing)
        if num_samples is not None:
            self.buffer = self.buffer[:num_samples]
        
        print(f"Dataset initialized with {len(self.buffer)} samples in {mode} mode")
    

    
    def _load_hdf5_buffer(self, buffer_path):
        """Load replay buffer data from an HDF5 file."""
        buffer = []
        with h5py.File(buffer_path, 'r') as f:
            # Assuming all datasets have the same length
            num_transitions = f['state'].shape[0]
            
            for i in range(num_transitions):
                transition = {
                    'state': f['state'][i],
                    'action': f['action'][i],
                    'next_state': f['next_state'][i],
                    'target_state': f['target_state'][i],
                    'reward': f['reward'][i],
                    'done': f['done'][i],
                    'transition_type': f['transition_type'][i],
                    'shape_h': f['shape_h'][i],
                    'shape_w': f['shape_w'][i],
                    'num_colors_grid': f['num_colors_grid'][i],
                    'most_present_color': f['most_present_color'][i],
                    'least_present_color': f['least_present_color'][i],
                    'num_colors_grid_target': f.get('num_colors_grid_target', f['num_colors_grid'])[i],
                    'most_present_color_target': f.get('most_present_color_target', f['most_present_color'])[i],
                    'least_present_color_target': f.get('least_present_color_target', f['least_present_color'])[i],
                    'shape_h_target': f.get('shape_h_target', f['shape_h'])[i],
                    'shape_w_target': f.get('shape_w_target', f['shape_w'])[i],
                    'shape_h_next': f.get('shape_h_next', f['shape_h'])[i],
                    'shape_w_next': f.get('shape_w_next', f['shape_w'])[i],
                    'num_colors_grid_next': f.get('num_colors_grid_next', f['num_colors_grid'])[i],
                    'most_present_color_next': f.get('most_present_color_next', f['most_present_color'])[i],
                    'least_present_color_next': f.get('least_present_color_next', f['least_present_color'])[i],
                    'step_distance_to_target': f.get('step_distance_to_target', [0])[i]
                }
                
                buffer.append(transition)
        return buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def _to_tensor(self, data, dtype):
        """Convert data to tensor, handling both tensor and non-tensor inputs."""
        if torch.is_tensor(data):
            return data.clone().detach().to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        transition = self.buffer[idx]
        
        # Extract state grids and convert to tensors
        state = self._to_tensor(transition['state'], torch.long)
        target_state = self._to_tensor(transition['target_state'], torch.long)
        next_state = self._to_tensor(transition['next_state'], torch.long)
        
        # Extract single action
        action = self._to_tensor(transition['action'], torch.long)
        
        # Extract grid metadata
        shape_h = self._to_tensor(transition['shape_h'], torch.long)
        shape_h_target = self._to_tensor(transition.get('shape_h_target', transition['shape_h']), torch.long)
        shape_h_next = self._to_tensor(transition.get('shape_h_next', transition['shape_h']), torch.long)
        shape_w = self._to_tensor(transition['shape_w'], torch.long)
        shape_w_target = self._to_tensor(transition.get('shape_w_target', transition['shape_w']), torch.long)
        shape_w_next = self._to_tensor(transition.get('shape_w_next', transition['shape_w']), torch.long)
        num_colors_grid = self._to_tensor(transition['num_colors_grid'], torch.long)
        num_colors_grid_target = self._to_tensor(transition.get('num_colors_grid_target', transition['num_colors_grid']), torch.long)
        num_colors_grid_next = self._to_tensor(transition.get('num_colors_grid_next', transition['num_colors_grid']), torch.long)
        most_present_color = self._to_tensor(transition['most_present_color'], torch.long)
        most_present_color_target = self._to_tensor(transition.get('most_present_color_target', transition['most_present_color']), torch.long)
        most_present_color_next = self._to_tensor(transition.get('most_present_color_next', transition['most_present_color']), torch.long)
        least_present_color = self._to_tensor(transition['least_present_color'], torch.long)
        least_present_color_target = self._to_tensor(transition.get('least_present_color_target', transition['least_present_color']), torch.long)
        least_present_color_next = self._to_tensor(transition.get('least_present_color_next', transition['least_present_color']), torch.long)
        
        # Extract step distance to target
        step_distance_to_target = self._to_tensor(transition.get('step_distance_to_target', 0), torch.long)
        
        # Prepare sample with new structure
        sample = {
            # Grid data
            'state': state,
            'target_state': target_state,
            'next_state': next_state,
            
            # Actions
            'action': action,
            
            # Other data
            'reward': self._to_tensor(transition['reward'], torch.float32),
            'done': self._to_tensor(float(transition['done']), torch.float32),
            
            # Grid metadata
            'shape_h': shape_h,
            'shape_w': shape_w,
            'num_colors_grid': num_colors_grid,
            'most_present_color': most_present_color,
            'least_present_color': least_present_color,
            
            # Target variants
            'shape_h_target': shape_h_target,
            'shape_w_target': shape_w_target,
            'num_colors_grid_target': num_colors_grid_target,
            'most_present_color_target': most_present_color_target,
            'least_present_color_target': least_present_color_target,
            
            # Next variants
            'shape_h_next': shape_h_next,
            'shape_w_next': shape_w_next,
            'num_colors_grid_next': num_colors_grid_next,
            'most_present_color_next': most_present_color_next,
            'least_present_color_next': least_present_color_next,
            
            # Episode info
            'transition_type': transition.get('transition_type', 'unknown'),
            'step_distance_to_target': step_distance_to_target
        }
        
        return sample 