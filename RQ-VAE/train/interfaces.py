import abc

from torch import nn


class Stage1Model(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_codes(self, *args, **kwargs):
        """Generate the code from the input."""
        pass

    @abc.abstractmethod
    def decode_code(self, *args, **kwargs):
        """Generate the decoded image from the given code."""
        pass

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute the losses necessary for training.

        return {
            'loss_total': ...,
            'loss_recon': ...,
            'loss_latent': ...,
            'codes': ...,
            ...
        }
        """
        pass