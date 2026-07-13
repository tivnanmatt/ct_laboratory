import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

class DiffusersUnet2D_Size28(torch.nn.Module):
    """
    Diffusers UNet2D wrapper for 28x28 images.
    
    This network circular pads 28x28 input to 32x32 for better U-Net performance,
    then crops back to 28x28 output.
    """

    def __init__(self, 
                 in_channels=3,
                 out_channels=3,
                 layers_per_block=2,
                 block_out_channels=(32, 64, 64),
                 down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                 up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D")):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = 28
        self.padded_size = 32
        
        # Create the base UNet2D model for 32x32 images
        self.unet = UNet2DModel(
            sample_size=self.padded_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            norm_num_groups=1,
        )

    def forward(self, x, timestep=None):
        """
        Forward pass with circular padding and cropping.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 28, 28)
            timestep: Optional timestep tensor for diffusion models. If None, uses a default timestep.
            
        Returns:
            Output tensor of shape (batch_size, channels, 28, 28)
        """
        batch_size, channels, height, width = x.shape
        
        # Verify input size
        if height != self.input_size or width != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}x{self.input_size}, got {height}x{width}")
        
        # Circular pad to 32x32
        pad_h = (self.padded_size - self.input_size) // 2
        pad_w = (self.padded_size - self.input_size) // 2
        
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        
        # Pass through UNet with timestep
        if timestep is None:
            # Use a default timestep (middle of the diffusion process)
            timestep = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        x_output = self.unet(x_padded, timestep=timestep).sample
        
        # Crop back to 28x28
        x_cropped = x_output[:, :, pad_h:pad_h+self.input_size, pad_w:pad_w+self.input_size]
        
        return x_cropped 