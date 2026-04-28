import copy
import torch
from typing import List


class BERTWhitening:

    def __init__(self, reduced_dim: int, num=0, mean=None, cov=None, kernel=None):
        self.reduced_dim = reduced_dim
        self.num = num
        self.mean = mean
        self.cov = cov
        self.kernel = kernel

    def __copy__(self):
        new_instance = BERTWhitening(
            self.reduced_dim, self.num, self.mean, self.cov, self.kernel
        )
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = BERTWhitening(
            copy.deepcopy(self.reduced_dim, memo),
            copy.deepcopy(self.num, memo),
            copy.deepcopy(self.mean, memo),
            copy.deepcopy(self.cov, memo),
            copy.deepcopy(self.kernel, memo)
        )
        return new_instance

    def copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)

    def incremental_fit(self, x: torch.Tensor | List[torch.Tensor]) -> "BERTWhitening":
        # initialize or update
        self._init_mean_cov(x) if self.mean is None or self.cov is None else self._update_mean_cov(x)

        return self

    def compute_kernel(self):
        """
        Compute kernel matrix.
        """
        # perform SVD
        u, s, vh = torch.svd(self.cov)

        # compute W
        W = u * (1.0 / torch.sqrt(s)).unsqueeze(0)  # u @ torch.diag(1.0 / torch.sqrt(s))
        W = torch.linalg.inv(W.T)  # (dim, dim)

        # compute kernel
        self.kernel = W[:, :self.reduced_dim]  # (dim, reduced_dim)

    def transform_norm(self, x: torch.Tensor | List[torch.Tensor]) -> torch.Tensor | List[torch.Tensor]:
        # error handling
        if self.kernel is None or self.mean is None:
            print('ERROR: BERT Whitening is not trained!')
            return None

        # transformation and normalization
        if isinstance(x, list):
            return [self._trans_norm(x_i) for x_i in x]  # [(seq_len, reduced_dim)]
        elif len(x.shape) == 3:
            shape = x.shape
            return self._trans_norm(x.reshape(-1, shape[-1])).reshape(shape)  # (bs * seq_len, reduced_dim)
        else:
            return self._trans_norm(x)  # (seq_len, reduced_dim)

    def _init_mean_cov(self, x: torch.Tensor | List[torch.Tensor]):
        """
        Initialize mean and covariance matrix.
        """
        # flatten
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
            x = x.reshape(-1, x.shape[-1])  # (bs * seq_len, dim)
        elif len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])  # (bs * seq_len, dim)

        # compute mean and covariance matrix
        self.mean = x.mean(dim=0, keepdim=True)  # (1, dim)
        self.cov = torch.cov(x.T)  # (dim, dim)

        # update number of samples
        self.num = x.shape[0]

    def _update_mean_cov(self, x: torch.Tensor | List[torch.Tensor]):
        """
        Update mean and covariance matrix.
        """
        # flatten
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
            x = x.reshape(-1, x.shape[-1])  # (bs * seq_len, dim)
        elif len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])  # (bs * seq_len, dim)

        # compute new number of samples
        num_new = self.num + x.shape[0]

        # compute new mean
        batch_sum = x.sum(dim=0)  # (1, dim)
        self.mean = (self.num * self.mean + batch_sum) / num_new  # (1, dim)

        # compute new covariance matrix
        x_centered = x - self.mean  # (seq_len, dim) | (bs * seq_len, dim)
        self.cov = (self.num * self.cov + torch.einsum('si,sj->ij', x_centered, x_centered)) / num_new  # (dim, dim)

        # update number of samples
        self.num = num_new

    def _trans_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        The last transformation: y = (x - mean) @ kernel
        """
        x_reduced = (x - self.mean) @ self.kernel  # (seq_len, reduced_dim) | (bs * seq_len, reduced_dim)

        return x_reduced / torch.linalg.norm(x_reduced, dim=-1, keepdim=True)

