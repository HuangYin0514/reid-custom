import torch

if __name__ == "__main__":
    x = torch.randn(1, 64, 3, 3)
    x = torch.Tensor([
        [[1, 2, 3], [4, 5, 60], [7, 8, 9], [10, 11, 12]],
        [[8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19]]
    ])
    # x = torch.Tensor([[1, 3, 2],
    #                   [1, 2, 3],
    #                   [1, 200, 100],
    #                   [1, 200, 100]])
    v, max_index = x.max(dim=1, keepdim=True)
    onehot_index = torch.zeros_like(x).scatter_(1, max_index, 1)
    print(onehot_index.shape)
    print(onehot_index)
    print(x.shape)
    print(max_index)
    print(v)
