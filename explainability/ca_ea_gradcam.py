import torch
import torch.nn.functional as F


class CA_EA_GradCAM:
    """
    Convergence-Aware Evolutionary Attention GradCAM

    Combines:
    - CNN GradCAM localization
    - Transformer attention context
    - Evolutionary convergence confidence
    """

    def __init__(self, model, target_layer):
        """
        model: trained CNN-Transformer model
        target_layer: CNN feature layer used for GradCAM
        """

        self.model = model
        self.target_layer = target_layer

        self.feature_maps = None
        self.gradients = None

        # hooks
        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def compute_gradcam(self, input_tensor, class_idx=None):

        """
        Compute CNN GradCAM heatmap
        """

        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[:, class_idx]

        score.backward(retain_graph=True)

        gradients = self.gradients
        feature_maps = self.feature_maps

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * feature_maps, dim=1)

        cam = F.relu(cam)

        cam = cam / (torch.max(cam) + 1e-8)

        return cam

    def compute_entropy(self, heatmap):
        """
        Spatial entropy of GradCAM map
        """

        p = heatmap / (torch.sum(heatmap) + 1e-8)

        entropy = -torch.sum(p * torch.log(p + 1e-8))

        return entropy

    def fuse_maps(self, cnn_map, transformer_map, psi):
        """
        Fuse CNN and transformer explanations
        """

        entropy = self.compute_entropy(cnn_map)

        gate = torch.sigmoid(entropy)

        fused = psi * (
            gate * cnn_map +
            (1 - gate) * transformer_map
        )

        fused = fused / (torch.max(fused) + 1e-8)

        return fused

    def generate(self, input_tensor, transformer_attention, psi):
        """
        Full CA-EA-GradCAM generation pipeline
        """

        cnn_map = self.compute_gradcam(input_tensor)

        # resize transformer map to CNN resolution
        transformer_map = F.interpolate(
            transformer_attention.unsqueeze(1),
            size=cnn_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        heatmap = self.fuse_maps(cnn_map, transformer_map, psi)

        return heatmap