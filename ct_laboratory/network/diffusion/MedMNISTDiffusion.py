import torch
import torch.nn as nn
from .. import SimpleCNN, LambdaLayer, DenseNet
from diffusers import UNet2DModel


class MedMNISTDiffusion(nn.Module):
    def __init__(self, 
                image_shape, 
                y_dim=None,
                x_t_embedding_channels=16, 
                t_embedding_channels=8, 
                y_embedding_channels=8,
                x_t_encoder_channel_list=[16, 32, 16],
                y_encoder_channel_list=[16, 32, 16],
                unet_layers_per_block=2,
                unet_norm_num_groups=1,
                unet_block_out_channels=(32, 64),
                unet_down_block_types=("AttnDownBlock2D", "AttnDownBlock2D"),
                unet_up_block_types=("AttnUpBlock2D", "AttnUpBlock2D"),
                preconditioner_type="EDM",
                c_skip=None,
                c_out=None,
                c_in=None,
                c_noise=None):
        
        # call the super constructor
        super(MedMNISTDiffusion, self).__init__()

        # if y_dim is None, there is no conditional input
        if y_dim is None:
            y_embedding_channels = 0
            y_encoder_channel_list = []

        # if preconditioner_type is not None, c_skip, c_out, c_in, and c_noise must be provided
        if preconditioner_type is not None:
            if c_skip is not None or c_out is not None or c_in is not None or c_noise is not None:
                raise ValueError("If preconditioner_type is provided, c_skip, c_out, c_in, and c_noise must be None.")
        
        # if preconditioner_type is provided, it must be one of ["SGM", "EDM", "denoising"]
        if c_skip is not None or c_out is not None or c_in is not None or c_noise is not None:
            if preconditioner_type is None:
                raise ValueError("If preconditioner_type is not provided, c_skip, c_out, c_in, and c_noise must all be provided.")

        # if preconditioner_type is not None, it must be one of ["SGM", "EDM", "denoising"]
        if preconditioner_type is not None:
            if preconditioner_type not in ["SGM", "EDM", "denoising"]:
                raise ValueError("preconditioner_type must be one of ['SGM', 'EDM', 'denoising']")

            def expand_t_to_x_shape(t, x_shape):
                assert isinstance(t, torch.Tensor)
                assert t.shape[0] == x_shape[0], f"t.shape[0] = {t.shape[0]}, x_shape[0] = {x_shape[0]}"
                t_shape = [t.shape[0]] + [1]*len(x_shape[1:])
                t = t.reshape(t_shape)
                return t
            
            if preconditioner_type == "denoising":
                c_skip = lambda t,x_shape: 0.0
                c_out = lambda t,x_shape: 1.0
                c_in = lambda t,x_shape: 1.0
                c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))
            elif preconditioner_type == "SGM":
                c_skip = lambda t,x_shape: 1.0
                c_out = lambda t,x_shape: torch.sqrt(expand_t_to_x_shape(t, x_shape))
                c_in = lambda t,x_shape: 1.0
                c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))
            elif preconditioner_type == "EDM":
                var_data = 0.25
                c_skip = lambda t,x_shape: var_data/(var_data + expand_t_to_x_shape(t,x_shape))
                c_out = lambda t,x_shape: torch.sqrt(expand_t_to_x_shape(t, x_shape)) * torch.sqrt(var_data/(var_data + expand_t_to_x_shape(t,x_shape)))
                c_in = lambda t,x_shape: 1.0 / torch.sqrt(var_data + expand_t_to_x_shape(t,x_shape))
                c_noise = lambda t,x_shape: 0.5*torch.log(expand_t_to_x_shape(t,x_shape))
        
        self.c_skip = c_skip
        self.c_out = c_out
        self.c_in = c_in
        self.c_noise = c_noise

        self.image_shape = image_shape
        self.y_dim = y_dim
        self.x_t_embedding_channels = x_t_embedding_channels
        self.t_embedding_channels = t_embedding_channels
        self.y_embedding_channels = y_embedding_channels

        self.x_t_encoder = SimpleCNN(input_channels=image_shape[0],
                                    output_channels=x_t_embedding_channels,
                                    hidden_channels_list=x_t_encoder_channel_list,
                                    activation=torch.nn.SiLU(),
                                    dim=2)

        self.t_encoder = torch.nn.Sequential(
            LambdaLayer(lambda x: x.view(-1, 1)),
            DenseNet((1,),
                     (t_embedding_channels,), 
                     hidden_channels_list=[16, 32, 16], 
                     activation=torch.nn.SiLU()),
            LambdaLayer(lambda x: x.view(-1, t_embedding_channels, 1, 1)),
            LambdaLayer(lambda x: x.repeat(1, 1, *image_shape[1:]))
        )

        class X_Out_Predictor(nn.Module):
            def __init__(self):
                super(X_Out_Predictor, self).__init__()

                self.model = self.unet = UNet2DModel(
                    sample_size=None,
                    in_channels=x_t_embedding_channels + t_embedding_channels + y_embedding_channels,
                    out_channels=image_shape[0],
                    layers_per_block=unet_layers_per_block,
                    norm_num_groups=unet_norm_num_groups,
                    block_out_channels=unet_block_out_channels,
                    down_block_types=unet_down_block_types,
                    up_block_types=unet_up_block_types)
                
            def forward(self, x_t_and_y_and_t_embedding, t):
                return self.model(x_t_and_y_and_t_embedding, t.squeeze())[0]
            
        self.x_out_predictor = X_Out_Predictor()

        if y_dim is not None:
            self.y_encoder = torch.nn.Sequential(
                LambdaLayer(lambda x: x.to(torch.float32)),
                LambdaLayer(lambda x: x.view(-1, y_dim)),
                DenseNet((y_dim,),
                        (y_embedding_channels,),
                        hidden_channels_list=y_encoder_channel_list,
                        activation=torch.nn.SiLU()),
                LambdaLayer(lambda x: x.view(-1, y_embedding_channels, 1, 1)),
                LambdaLayer(lambda x: x.repeat(1, 1, *image_shape[1:]))
            )

    def forward(self, x_t, t, y):

        # x_t should be a tensor
        assert isinstance(x_t, torch.Tensor)

        # t should be a tensor
        assert isinstance(t, torch.Tensor)

        # y should be a tensor or None
        assert isinstance(y, torch.Tensor)

        # apply the input preconditioner
        x_t_precond = self.c_in(t, x_t.shape)*x_t

        # apply the noise preconditioner
        t_precond = self.c_noise(t, t.shape)
    
        # apply the x_t_encoder to get the x_t_embedding
        x_t_embedding = self.x_t_encoder(x_t_precond)

        # apply the t_encoder to get the t_embedding
        t_embedding = self.t_encoder(t_precond)

        # x_t_embedding should be a tensor
        assert isinstance(x_t_embedding, torch.Tensor)

        # t_embedding should be a tensor
        assert isinstance(t_embedding, torch.Tensor)

        # apply the y_encoder to get the y_embedding
        y_embedding = self.y_encoder(y)

        # y_embedding should be a tensor
        assert isinstance(y_embedding, torch.Tensor)

        # apply the x_out_predictor to get the x_out
        x_out = self.x_out_predictor(torch.cat([x_t_embedding, t_embedding, y_embedding], dim=1),t)

        # x_out should be a tensor
        assert isinstance(x_out, torch.Tensor)

        # apply the output and skip preconditioners to get the x_0_pred
        x_0_pred = self.c_skip(t, x_t.shape)*x_t + self.c_out(t, x_out.shape)*x_out

        # return it :)
        return x_0_pred