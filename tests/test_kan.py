import torch
import torch.nn as nn
from tqdm import tqdm

from models.component.efficient_kan import KAN

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
def test_mul():
    kan = KAN([2, 2, 1], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=1e-1, max_iter=20, max_eval=20, history_size=100, line_search_fn="strong_wolfe")
    with tqdm(range(100)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2).to(device=device)
                y = kan(x, update_grid=(i % 20 == 0))

                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = kan.regularization_loss(1, 0)
                (loss + 1e-5 * reg_loss).backward()
                return loss + reg_loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())
    for layer in kan.layers:
        print(layer.spline_weight)

if __name__ == "__main__":
    test_mul()