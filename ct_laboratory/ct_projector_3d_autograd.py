import torch

# For Torch-based forward/back:
from .ct_projector_3d_torch import (
    forward_project_3d_torch,
    back_project_3d_torch
)

# For CUDA-based forward/back:
from .ct_projector_3d_cuda import (
    forward_project_3d_cuda,
    back_project_3d_cuda
)


class CTProjector3DFunction(torch.autograd.Function):
    """
    Autograd Function for 3D forward/back projection with precomputed intersections.
    Supports either 'torch' or 'cuda' backend.
    """
    @staticmethod
    def forward(ctx, volume, t_main, t_start, t_scale, M, b, src, dst, backend='cuda'):
        """
        volume: [n_x,n_y,n_z] or [B,n_x,n_y,n_z]
        t_main: tvals tensor (float) OR tvals_uint16 (if compressed)
        t_start: start values (if compressed) else None
        t_scale: scale values (if compressed) else None
        """
        # Helper to construct arg
        if t_start is not None and t_scale is not None:
            tvals_arg = (t_main, t_start, t_scale)
            ctx.is_compressed = True
        else:
            tvals_arg = t_main
            ctx.is_compressed = False
            
        ctx.backend = backend
        ctx.volume_shape = volume.shape
        
        # We need to save tensors for backward.
        if ctx.is_compressed:
            ctx.save_for_backward(t_main, t_start, t_scale, M, b, src, dst)
        else:
            # Handle None safely for save_for_backward if t_main is None (on-the-fly)
            if t_main is not None:
                ctx.save_for_backward(t_main, M, b, src, dst)
            else:
                ctx.save_for_backward(M, b, src, dst)

        if backend == 'torch':
            sinogram = forward_project_3d_torch(volume, tvals_arg, M, b, src, dst)
        else:
            # Fallback for CUDA
            if ctx.is_compressed:
                # Compressed CUDA path (chunked to save memory)
                from .ct_projector_3d_cuda import forward_project_3d_compressed_cuda
                tvals_u16, tvals_s, tvals_sc = tvals_arg
                n_ray = tvals_u16.shape[0]
                
                # Determine basic shapes
                if volume.ndim == 3:
                    batch_size = 1
                else:
                    batch_size = volume.shape[0]
                
                sinogram = torch.zeros((batch_size, n_ray) if batch_size > 1 else (n_ray,), 
                                     device=volume.device, dtype=volume.dtype)
                
                import math
                chunk_size = 65536 # Reasonable chunk size for GPU
                n_chunks = math.ceil(n_ray / chunk_size)
                
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = min(n_ray, (i + 1) * chunk_size)
                    
                    src_chunk = src[start:end]
                    dst_chunk = dst[start:end]
                    
                    sino_chunk = forward_project_3d_compressed_cuda(
                        volume, 
                        tvals_u16[start:end], 
                        tvals_s[start:end], 
                        tvals_sc[start:end],
                        M, b, src_chunk, dst_chunk
                    )
                    
                    if batch_size > 1:
                        sinogram[:, start:end] = sino_chunk
                    else:
                        sinogram[start:end] = sino_chunk.squeeze(0) if sino_chunk.ndim==2 else sino_chunk
                        
            else:
                sinogram = forward_project_3d_cuda(volume, tvals_arg, M, b, src, dst)

        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        saved = ctx.saved_tensors
        if ctx.is_compressed:
            t_main, t_start, t_scale, M, b, src, dst = saved
            tvals_arg = (t_main, t_start, t_scale)
        else:
            if len(saved) == 5:
                t_main, M, b, src, dst = saved
                tvals_arg = t_main
            else:
                M, b, src, dst = saved
                tvals_arg = None
            t_start = None
            t_scale = None
            
        backend = ctx.backend
        volume_shape = ctx.volume_shape

        if len(volume_shape) == 3:
            n_x, n_y, n_z = volume_shape
        else:
            _, n_x, n_y, n_z = volume_shape

        if backend == 'torch':
            grad_volume = back_project_3d_torch(
                grad_output, tvals_arg, M, b, src, dst, n_x, n_y, n_z
            )
        else:
             if ctx.is_compressed:
                # Compressed CUDA path (chunked to save memory)
                from .ct_projector_3d_cuda import back_project_3d_compressed_cuda
                tvals_u16, tvals_s, tvals_sc = tvals_arg
                n_ray = tvals_u16.shape[0]
                
                grad_volume = torch.zeros(volume_shape, device=grad_output.device, dtype=grad_output.dtype)
                
                import math
                chunk_size = 65536 # Chunk size for backprojection
                n_chunks = math.ceil(n_ray / chunk_size)
                
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = min(n_ray, (i + 1) * chunk_size)
                    
                    src_chunk = src[start:end]
                    dst_chunk = dst[start:end]
                    
                    if grad_output.ndim == 2:
                        grad_chunk = grad_output[:, start:end].contiguous()
                    else:
                        grad_chunk = grad_output[start:end].contiguous()

                    grad_vol_chunk = back_project_3d_compressed_cuda(
                        grad_chunk, 
                        tvals_u16[start:end], 
                        tvals_s[start:end], 
                        tvals_sc[start:end],
                        M, b, src_chunk, dst_chunk, n_x, n_y, n_z
                    )
                    
                    grad_volume += grad_vol_chunk
             else:
                grad_volume = back_project_3d_cuda(
                    grad_output, tvals_arg, M, b, src, dst, n_x, n_y, n_z
                )

        return grad_volume, None, None, None, None, None, None, None, None

    



class CTBackProjector3DFunction(torch.autograd.Function):
    def __init__(self, base_forward_projector):
        super().__init__()
        self.base_forward_projector = base_forward_projector

    @staticmethod
    def forward(self, sinogram):
        self.base_forward_projector.backward(sinogram)

    @staticmethod
    def backward(self, grad_volume):
        self.base_forward_projector.forward(grad_volume)