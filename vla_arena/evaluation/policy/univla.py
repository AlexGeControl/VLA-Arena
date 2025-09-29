import numpy as np
import torch
import math
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.policy.modeling_univla import *
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action
from vla_arena.evaluation.utils import read_eval_cfgs, resize_image

def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite transform_utils.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if den > 0.0:
        quat[0] /= den
        quat[1] /= den
        quat[2] /= den
    return quat[:3]


@PolicyRegistry.register("univla")
class UniVLA(Policy):
    def __init__(self, 
                 model_ckpt,
                 action_decoder_path,
                 device="cuda",
                 unnorm_key=None,
                 eval_cfgs_path='../../configs/UniVLA.yaml',
                 **kwargs):
        super().__init__(**kwargs)
        self.eval_cfgs = read_eval_cfgs(self.name, eval_cfgs_path)
        self.model = OpenVLAForActionPrediction.from_pretrained(model_ckpt,
            attn_implementation=self.eval_cfgs.get('attn_implementation', 'eager'),
            torch_dtype=torch.bfloat16,
            load_in_8bit=self.eval_cfgs['load_in_8bit'],
            load_in_4bit=self.eval_cfgs['load_in_4bit'],
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()
        self.image_processor = PrismaticImageProcessor.from_pretrained(model_ckpt)
        self.action_decoder = ActionDecoder().to(device)
        self.action_decoder.load_state_dict(torch.load(action_decoder_path, map_location=device))
        self.action_decoder.eval().to(device)
        self.device = device
        self.policy_type = "continuous"
        self.name = "univla"
        self.instruction = None
        self.prev_hist_action = []
        self.unnorm_key = unnorm_key if unnorm_key is not None else self.eval_cfgs['unnorm_key']
        self.center_crop = self.eval_cfgs['center_crop']
        self.image_resize_size = self.eval_cfgs['image_resize_size']
    
    def _build_prompt(self, hist_action=None):
        if hist_action is not None:
            prompt = f"In: What action should the robot take to {self.instruction.lower()}? History action {hist_action}\nOut:"
        else:
            prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        return prompt
    
    def _process_observation(self, obs, **kwargs):
        full_image = obs['agentview_image']
        full_image = resize_image(full_image, (self.image_resize_size, self.image_resize_size))
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if self.center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            full_image = tf.convert_to_tensor(np.array(full_image))
            orig_dtype = full_image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            full_image = tf.image.convert_image_dtype(full_image, tf.float32)

            # Crop and then resize back to original size
            full_image = crop_and_resize(full_image, crop_scale, batch_size)

            # Convert back to original data type
            full_image = tf.clip_by_value(full_image, 0, 1)
            full_image = tf.image.convert_image_dtype(full_image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            full_image = Image.fromarray(full_image.numpy())
            full_image = full_image.convert("RGB")
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
        return full_image, state

    def _process_action(self, action):
        action = normalize_gripper_action(action)
        action = invert_gripper_action(action)
        return action
    
    def predict(self, obs, **kwargs):
        
        # prepare inputs
        image, state = self._process_observation(obs, **kwargs)
        prompt = self._build_prompt(self.prev_hist_action[-1] if len(self.prev_hist_action) > 0 else None)
        inputs = self.image_processor(prompt, image).to(self.model.device, dtype=torch.bfloat16)
        
        # get latent action
        latent_action, visual_embed, generated_ids = self.model.predict_latent_action(**inputs, unnorm_key=self.unnorm_key, do_sample=True, temperature=0.75, top_p = 0.9)
        
        # record history latent action
        hist_action = ''
        for latent_action_ids in generated_ids[0]:
            hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
        prev_hist_action.append(hist_action)        
        
        # get action norm stats
        action_norm_stats = self.model.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        
        # get action from latent action
        action = self.action_decoder(latent_action, visual_embed, mask, action_low, action_high)
        
        # process action to align with the action format
        action = self._process_action(action)
        return action