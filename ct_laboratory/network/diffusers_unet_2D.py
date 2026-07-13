import torch
from diffusers import UNet2DModel

class DiffusersUnet2D(torch.nn.Module):

    def __init__(self,  
                 image_size=512,
                 unet_in_channels=64, 
                 unet_base_channels=64,
                 unet_out_channels=1):

        super().__init__()

        # Create a UNet2DModel for noise prediction given x_t and t
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            layers_per_block=2,
            norm_num_groups=1,
            block_out_channels=(unet_base_channels, unet_base_channels, 2*unet_base_channels, 2*unet_base_channels, 4*unet_base_channels, 4*unet_base_channels),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x_t, t):

        x_0_pred = self.unet(x_t, t.squeeze())[0]

        return x_0_pred

