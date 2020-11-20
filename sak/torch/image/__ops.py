import torch
import string

def batch_xcorr(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    all_letters = string.ascii_lowercase
    einsum_format = '{},{}->{}'.format(all_letters[:x.ndim+1],all_letters[:x.ndim+1],all_letters[1:3])
    dot_xy = torch.einsum(einsum_format,x[None,],y[:,None])
    dot_x  = torch.einsum(einsum_format,x[None,],x[:,None])
    dot_y  = torch.einsum(einsum_format,y[None,],y[:,None])

    return dot_xy/torch.sqrt(dot_x*dot_y)