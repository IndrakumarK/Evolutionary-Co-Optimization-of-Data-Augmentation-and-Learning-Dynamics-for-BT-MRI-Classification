import torchvision.transforms.functional as TF
import torch


class AugmentationPipeline:

    def __init__(self):
        self.params = {}

    def update_params(self, param_dict):
        self.params = param_dict

    def __call__(self, images):

        brightness = self.params["brightness"]
        contrast = self.params["contrast"]
        rotation = self.params["rotation"]
        noise = self.params["noise"]

        images = TF.adjust_brightness(images, 1 + brightness)
        images = TF.adjust_contrast(images, 1 + contrast)
        images = TF.rotate(images, rotation)

        if noise > 0:
            images = images + noise * torch.randn_like(images)

        return images