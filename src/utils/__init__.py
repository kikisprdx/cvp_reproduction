import torch
import torch.nn.functional as F
import torchvision.transforms as T

# The augmentations specified by the paper
ssl_transform = T.Compose(
    [
        T.RandomResizedCrop(32, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=[-90, 90]),
    ]
)


def contrastive_loss(features, batch_size, n_views=3, temperature=0.1):
    """Contrastive loss over n_views augmented views per sample."""
    # Normalize features for cosine similarity
    features = F.normalize(features, dim=1)

    # Calculate cosine similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Mask out self-similarity -> don't want to compare a view to itself
    mask = torch.eye(batch_size * n_views, dtype=torch.bool, device=features.device)
    sim_matrix.masked_fill_(mask, float('-inf'))

    # Labels: y_i,j is a 0-1 vector indicating positive pairs
    # Views from the same original image share the same label
    labels = torch.arange(batch_size, device=features.device).repeat(n_views)

    # Boolean mask of positive pairs (same label, but not the exact same view)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask

    # Compute log probabilities
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    # Calculate the mean log-probability of positive pairs
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

    # Negative expected value
    loss = -pos_log_prob.mean()
    return loss
