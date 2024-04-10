import argparse
import torch

def matmult_gpu(input_data, weights):
    """
    Perform matrix multiplication of two tensors on GPU.
    
    Args:
    input_data (torch.Tensor): Input tensor.
    weights (torch.Tensor): Weight tensor.
    
    Returns:
    torch.Tensor: Result of matrix multiplication.
    """
    # Creating tensors on GPU
    input_data = input_data.to('cuda')
    weights = weights.to('cuda')
    
    # Optimized matrix multiplication using torch.matmul
    output = torch.matmul(input_data, weights)
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform matrix multiplication of two tensors.')
    parser.add_argument('--x_shape', nargs=2, type=int, default=[1000, 500], metavar=('N', 'M'), help='Shape of input data matrix')
    parser.add_argument('--w_shape', nargs=2, type=int, default=[500, 500], metavar=('J', 'K'), help='Shape of weight matrix')
    args = parser.parse_args()

    input_data = torch.randn(*args.x_shape)
    weights = torch.randn(*args.w_shape)

    output = matmult_gpu(input_data, weights)    
    print(f'Shape of input data matrix: {args.x_shape}, weight matrix: {args.w_shape}, result matrix:{output.shape}')
    print(output)
