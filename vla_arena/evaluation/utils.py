import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import cv2
import logging
import colorlog
import json
import os
import yaml
from typing import Any
import tensorflow as tf
def normalize(v):
    return v / np.linalg.norm(v)

def compute_rotation_quaternion(camera_pos, target_pos, forward_axis=[1, 0, 0]):
    """
        Compute the ratation quaternion from camera position to target position
    """
    target_direction = np.array(target_pos) - np.array(camera_pos)
    target_direction = normalize(target_direction)

    base_forward = normalize(np.array(forward_axis))

    if np.allclose(target_direction, base_forward):
        return R.from_quat([0, 0, 0, 1])
    elif np.allclose(target_direction, -base_forward):
        orthogonal_axis = np.array([base_forward[1], -base_forward[0], 0])
        orthogonal_axis = normalize(orthogonal_axis)
        return R.from_rotvec(np.pi * orthogonal_axis).as_quat()
    else:
        axis = np.cross(base_forward, target_direction)
        axis = normalize(axis)
        angle = np.arccos(np.clip(np.dot(base_forward, target_direction), -1.0, 1.0))
        return R.from_rotvec(angle * axis).as_quat()

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return (qw, qx, qy, qz)

def quaternion_to_euler(quat, is_degree=False):
    # (w, x, y, z) -> (x, y, z, w)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def matrix_to_quaternion(matrix):
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    r = R.from_matrix(matrix)
    quaternion = r.as_quat()
    # (x, y, z, w) -> (w, x, y, z)
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return quaternion

def quaternion_to_matrix(quat):
    # (w, x, y, z) -> (x, y, z, w)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    matrix = r.as_matrix()
    return matrix

def move_long_quaternion(position, quaternion, distance):
    """
        Move along the quaternion direction
    """
    roation = R.from_quat(quaternion)
    direction = roation.as_rotvec()
    direction = direction / np.linalg.norm(direction)
    new_position = position + direction * distance
    return new_position

def distance(p1, p2):
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1)
    if not isinstance(p2, np.ndarray):
        p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def farthest_first_sampling(points, k):
    sampled_points = [points[np.random.randint(len(points))]]
    
    for _ in range(1, k):
        min_distances = [min(distance(p, sp) for sp in sampled_points) for p in points]
        
        # choose the point with max minimal distance
        farthest_point = points[np.argmax(min_distances)]
        sampled_points.append(farthest_point)
    
    return sampled_points

def grid_sample(workspace, grid_size, n_samples, farthest_sample=True):
    """
    workspace: [min_x, max_x, min_y, max_y, min_z, max_z]
    grid_size: [n_row, n_col]
    
    """
    min_x, max_x, min_y, max_y, _, _ = workspace
    n_row, n_col = grid_size
    x_step = (max_x - min_x) / n_col
    y_step = (max_y - min_y) / n_row
    
    grid_points = []
    for i in range(n_row):
        for j in range(n_col):
            center_x = min_x + (j + 0.5) * x_step
            center_y = min_y + (i + 0.5) * y_step
            grid_points.append((center_x, center_y))
    if farthest_sample:
        sampled_points = farthest_first_sampling(grid_points, n_samples)
    else:
        sampled_points = random.sample(grid_points, n_samples)
    
    return sampled_points

def point_to_line_distance(anchor, axis, point):
    """
    compute the distance from a point to a line
    
    param:
    - anchor: the anchor point of rotation axis (3D vector) [x, y, z]
    - axis: the direction vector of rotation axis [vx, vy, vz]
    - point: (3D vector) [x, y, z]
    
    return:
    - the distance
    """
    A = np.array(anchor)  
    V = np.array(axis)    
    Q = np.array(point)

    AQ = Q - A

    cross_product = np.cross(AQ, V)

    distance = np.linalg.norm(cross_product)

    return distance

def rotate_point_around_axis(point, anchor, axis, angle):
    """
    compute the point after rotation around the axis with Rodrigues' rotation formula

    params:
    - point: (3D vector) [x, y, z]
    - anchor:(3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    P = np.array(point)
    A = np.array(anchor)
    V = np.array(axis) / np.linalg.norm(axis)  

    PA = P - A

    part1 = np.cos(angle) * PA
    part2 = np.sin(angle) * np.cross(V, PA)
    part3 = (1 - np.cos(angle)) * V * np.dot(V, PA)

    P_prime = A + part1 + part2 + part3

    return P_prime

def slide_point_along_axis(point, axis, distance):
    """
    compute the point after sliding along the axis 

    params:
    - point: (3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    point = np.array(point)
    axis = np.array(axis)
    
    xaxis_normalized = axis / np.linalg.norm(axis)
    
    new_point = point + distance * xaxis_normalized
    
    return new_point

def quaternion_from_axis_angle(axis, angle):
    """
    param:
     - angle: radian
    """
    half_angle = angle / 2
    w = np.cos(half_angle)
    sin_half_angle = np.sin(half_angle)
    
    v = np.array(axis) / np.linalg.norm(axis)
    
    x = v[0] * sin_half_angle
    y = v[1] * sin_half_angle
    z = v[2] * sin_half_angle
    
    return np.array([w, x, y, z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def flatten_list(ls):
    new_list = []
    for item in ls:
        if isinstance(item, list):
            new_list.extend(item)
        elif isinstance(item, str):
            new_list.append(item)
    return new_list

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_point_by_quaternion(point, quat):
    p = np.array([0] + list(point))
    q_conj = quaternion_conjugate(quat)
    p_prime = quaternion_multiply(quaternion_multiply(quat, p), q_conj)
    
    return p_prime[1:]

def expand_mask(masks, kernel_size=3, iterations=1):
    """
    Expands a batch of binary masks (0 and 1 values) using morphological dilation.
    
    Parameters:
    - masks: np.ndarray, shape (n, h, w), batch of binary masks (0 and 1 values).
    - kernel_size: int, size of the kernel for dilation, default is 3x3.
    - iterations: int, number of times to apply dilation, default is 1.

    Returns:
    - expanded_masks: np.ndarray, shape (n, h, w), batch of masks with dilated edges.
    """
    if len(masks.shape) == 2: #  convert (h, w) to (1, h, w) for unified operation
        masks = masks.reshape(1, masks.shape[0], masks.shape[1])
    # Define the dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Create an empty array to store the expanded masks
    expanded_masks = np.zeros_like(masks, dtype=np.uint8)
    # Loop through each mask in the batch
    for i in range(masks.shape[0]):
        # Invert the mask: 0 -> 1, 1 -> 0
        inverted_mask = 1 - masks[i]
        # Convert the inverted mask to uint8 (required for OpenCV functions)
        mask_uint8 = (inverted_mask * 255).astype(np.uint8)
        # Apply morphological dilation
        expanded_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        # Convert back to binary (0 and 1), then invert again: 1 -> 0, 0 -> 1
        expanded_masks[i] = 1 - (expanded_mask > 0).astype(np.uint8)
    return expanded_masks


def find_key_by_value(dictionary, target_value):
    """
    Given a dictionary and the corresponding value, find the key that contains the target value
    """
    for key, value in dictionary.items():
        if isinstance(value, list) and target_value in value:
            return key
        elif not isinstance(value, list) and value == target_value:
            return key
    return target_value

def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
        
    logger.addHandler(console_handler)
    return logger

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action

def load_initial_states(cfg, task_suite, task_id, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        print(f"Using initial states from {cfg.initial_states_path}")
        return initial_states, all_initial_states
    else:
        print("Using default initial states")
        return initial_states, None
    
def read_eval_cfgs(model_family: str, eval_cfgs_path: str = None):
    if eval_cfgs_path is not None:
        yaml_path = os.path.join(eval_cfgs_path)
    else:
        current_file_path = os.path.abspath(__file__)
        parent_path = os.path.dirname(os.path.dirname(current_file_path))
        yaml_path = os.path.join(parent_path, 'configs', 'evaluation',  f'{model_family}.yaml')
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc

    return configs

def read_task_suite_cfgs(task_suite_name: str):
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, 'configs', 'task_suite',  f'{task_suite_name}.yaml')
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc
    return configs  

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img