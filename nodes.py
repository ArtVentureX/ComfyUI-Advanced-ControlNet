import sys
import os

import torch

import numpy as np
from PIL import Image, ImageOps

import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .control import load_controlnet, ControlNetWeightsType, T2IAdapterWeightsType,\
    LatentKeyframe, LatentKeyframeGroup, TimestepKeyframe, TimestepKeyframeGroup


def get_properly_arranged_t2i_weights(initial_weights: list[float]):
    new_weights = []
    new_weights.extend([initial_weights[0]]*3)
    new_weights.extend([initial_weights[1]]*3)
    new_weights.extend([initial_weights[2]]*3)
    new_weights.extend([initial_weights[3]]*3)
    return new_weights


class ScaledSoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, base_multiplier, flip_weights):
        weights = [(base_multiplier ** float(12 - i)) for i in range(13)]
        if flip_weights:
            weights.reverse()
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_net_weights=weights)))


class SoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.09941396206337118, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.12050177219802567, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.14606275417942507, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 0.17704576264172736, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 0.214600924414215, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 0.26012233262329093, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 0.3152997971191405, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 0.3821815722656249, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 0.4632503906249999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 0.561515625, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 0.6806249999999999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        if flip_weights:
            weights.reverse()
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_net_weights=weights)))


class CustomControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        if flip_weights:
            weights.reverse()
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_net_weights=weights)))


class SoftT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        if flip_weights:
            weights.reverse()
        weights = get_properly_arranged_t2i_weights(weights)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(t2i_adapter_weights=weights)))


class CustomT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        if flip_weights:
            weights.reverse()
        weights = get_properly_arranged_t2i_weights(weights)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(t2i_adapter_weights=weights)))


class TimestepKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "prev_timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "adv-controlnet/keyframes"

    def load_keyframe(self,
                      start_percent: float,
                      control_net_weights: ControlNetWeightsType=None,
                      t2i_adapter_weights: T2IAdapterWeightsType=None,
                      latent_keyframe: LatentKeyframeGroup=None,
                      prev_timestep_keyframe: TimestepKeyframeGroup=None):
        if not prev_timestep_keyframe:
            prev_timestep_keyframe = TimestepKeyframeGroup()
        keyframe = TimestepKeyframe(start_percent, control_net_weights, t2i_adapter_weights, latent_keyframe)
        prev_timestep_keyframe.add(keyframe)
        return (prev_timestep_keyframe,)
    

class LatentKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.00001}, ),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "adv-controlnet/keyframes"

    def load_keyframe(self,
                      batch_index: int,
                      strength: float,
                      prev_latent_keyframe: LatentKeyframeGroup=None):
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        keyframe = LatentKeyframe(batch_index, strength)
        prev_latent_keyframe.add(keyframe)
        return (prev_latent_keyframe,)


class LatentKeyframeTimingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index_from": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "batch_index_to": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "strength_from": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.00001}, ),
                "strength_to": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.00001}, ),
                "timming": (["linear", "ease-in", "ease-out", "ease-in-out"], ),
                "flip_weights": ([False, True], ),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "adv-controlnet/keyframes"

    def load_keyframe(self,
                        batch_index_from: int,
                        strength_from: float,
                        batch_index_to: int,
                        strength_to: float,
                        timming: str,
                        flip_weights: bool,
                        prev_latent_keyframe: LatentKeyframeGroup=None):

        if (batch_index_from > batch_index_to):
            raise ValueError("batch_index_from must be less than or equal to batch_index_to.")

        if (batch_index_from < 0 and batch_index_to >= 0):
            raise ValueError("batch_index_from and batch_index_to must be either both positive or both negative.")

        if (strength_to < strength_from):
            raise ValueError("strength_to must be greater than or equal to strength_from.")

        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()

        steps = batch_index_to - batch_index_from + 1
        diff = strength_to - strength_from
        if timming == "linear":
            weights = np.linspace(strength_from, strength_to, steps)
        elif timming == "ease-in":
            index = np.linspace(0, 1, steps)
            weights = diff * np.power(index, 2) + strength_from
        elif timming == "ease-out":
            index = np.linspace(0, 1, steps)
            weights = diff * (1 - np.power(1 - index, 2)) + strength_from
        elif timming == "ease-in-out":
            index = np.linspace(0, 1, steps)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

        if flip_weights:
            weights = np.flip(weights)

        for i in range(steps):
            keyframe = LatentKeyframe(batch_index_from + i, float(weights[i]))
            print("keyframe", batch_index_from + i, ":", weights[i])
            prev_latent_keyframe.add(keyframe)

        return (prev_latent_keyframe,)


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "adv-controlnet/loaders"

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroup=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe)
        return (controlnet,)
    

class DiffControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), )
            },
            "optional": {
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "adv-controlnet/loaders"

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroup, model):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe, model)
        return (controlnet,)


class ControlNetApplyAdvanced_AdvControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "mask_opt": ("MASK", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "adv-controlnet/loaders/conditioning"

    def apply_controlnet(self, positive, negative, control_net, image, mask_opt, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class ControlNetApplyPartialBatch: # NOT USED: was used for a different test, has useful index parsing code though
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})    
            },
            "optional": {
                "latent_image": ("LATENT", ),
                "latent_indeces": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "adv-controlnet/conditioning"

    def validate_index(self, index: int, latent_count: int, is_range: bool = False) -> int:
        # if part of range, do nothing
        if is_range:
            return index
        # otherwise, validate index
        # validate not out of range
        if index > latent_count-1:
            raise IndexError(f"Index '{index}' out of range for the total {latent_count} latents.")
        # if negative, validate not out of range
        if index < 0:
            conv_index = latent_count+index
            if conv_index < 0:
                raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for the total {latent_count} latents.")
            index = conv_index
        return index

    def convert_to_index_int(self, raw_index: str, is_range: bool = False) -> int:
        try:
            return self.validate_index(int(raw_index), is_range=is_range)
        except ValueError as e:
            raise ValueError(f"index '{raw_index}' must be an integer.", e)

    def convert_to_indeces(self, latent_indeces: str, latent_count: int) -> set[int]:
        if not latent_indeces:
            return set()
        all_indeces = [i for i in range(0, latent_count)]
        chosen_indeces = set()
        # parse string - allow positive ints, negative ints, and ranges separated by ':'
        groups = latent_indeces.split(",")
        groups = [g.strip() for g in groups]
        for g in groups:
            # parse range of indeces (e.g. 2:16)
            if ':' in g:
                index_range = g.split(":", 1)
                index_range = [r.strip() for r in index_range]
                start_index = self.convert_to_index_int(index_range[0], is_range=True)
                end_index = self.convert_to_index_int(index_range[1], is_range=True)
                for i in all_indeces[start_index, end_index]:
                    chosen_indeces.add(i)
            # parse individual indeces
            else:
                chosen_indeces.add(self.convert_to_index_int(g))
        return chosen_indeces

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, latent_image=None, latent_indeces: str=None):
        if strength == 0:
            return (positive, negative)

        latent_count = 1
        if latent_image:
            latent_count = latent_image['samples'].size()[0]
        indeces_to_apply = self.convert_to_indeces(latent_indeces, latent_count)

        control_hint = image.movedim(-1,1)
        cnets = {}

        evaluating_positive = True
        out = []
        for conditioning in [positive, negative]:
            c = []
            if evaluating_positive and latent_count > 1:
                # should copy positive conditioning to match latent_count
                if len(conditioning) < latent_count:
                    pass
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
                evaluating_positive = False
            out.append(c)
        return (out[0], out[1])


class LoadImagesFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "adv-controlnet/image"

    def load_images(self, directory: str, image_load_cap: int = 0):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1
        
        if len(images) == 0:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        return (torch.cat(images, dim=0), torch.cat(masks, dim=0), image_count)




# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": TimestepKeyframeNode,
    "LatentKeyframe": LatentKeyframeNode,
    "LatentKeyframeTiming": LatentKeyframeTimingNode,
    # Conditioning
    # "ControlNetApplyPartialBatch": ControlNetApplyPartialBatch,
    # Loaders
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Weights
    "ScaledSoftControlNetWeights": ScaledSoftControlNetWeights,
    "SoftControlNetWeights": SoftControlNetWeights,
    "CustomControlNetWeights": CustomControlNetWeights,
    "SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    # Image
    "LoadImagesFromDirectory": LoadImagesFromDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe",
    "LatentKeyframe": "Latent Keyframe",
    "LatentKeyframeTiming": "Latent Keyframe Timing",
    # Conditioning
    # "ControlNetApplyPartialBatch": "Apply ControlNet (Partial Batch)",
    # Loaders
    "ControlNetLoaderAdvanced": "Load ControlNet Model (Advanced)",
    "DiffControlNetLoaderAdvanced": "Load ControlNet Model (diff Advanced)",
    # Weights
    "ScaledSoftControlNetWeights": "Scaled Soft ControlNet Weights",
    "SoftControlNetWeights": "Soft ControlNet Weights",
    "CustomControlNetWeights": "Custom ControlNet Weights",
    "SoftT2IAdapterWeights": "Soft T2IAdapter Weights",
    "CustomT2IAdapterWeights": "Custom T2IAdapter Weights",
    # Image
    "LoadImagesFromDirectory": "Load Images"
}
