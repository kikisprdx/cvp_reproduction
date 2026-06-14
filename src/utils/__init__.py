import torch
import torch.nn.functional as F
import torchvision.transforms as T

# TODO: Should be in its own utils folder
# the augmentations specified by the paper
ssl_transform = T.Compose(
    [
        T.RandomResizedCrop(32, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=[-90, 90]),
    ]
)


def contrastive_loss(features, batch_size, n_views=3, temperature=0.1):
    """Implementation of Equation 1 from the paper."""
    # normalize features for cosine similarity
    features = F.normalize(features, dim=1)

    # calculate cosine similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature

    # mask out self-similarity -> don't want to compare a view to itself
    mask = torch.eye(batch_size * n_views, dtype=torch.bool, device=features.device)
    sim_matrix.masked_fill_(mask, -9e15)

    # labels: y_i,j is a 0-1 vector indicating positive pairs
    # views from the same original image share the same label
    labels = torch.arange(batch_size, device=features.device).repeat(n_views)

    # boolean mask of positive pairs (same label, but not the exact same view)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask

    # compute log probabilities
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    # calculate the mean log-probability of positive pairs
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

    # negative expected value
    loss = -pos_log_prob.mean()
    return loss
