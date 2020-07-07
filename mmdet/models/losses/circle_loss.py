# coding:utf-8

import torch
import torch.distributed as dist

class CircleLoss(torch.nn.Module):
    """
    Circle loss for pairwise labels only. Support for class-level labels will be added
    in the future.

    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 256
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """

    def __init__(
            self,
            m=0.4,
            gamma=80,
            triplets_per_anchor='all',
            normalize_embeddings=True,
            **kwargs
    ):
        super(CircleLoss, self).__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.m = m
        self.gamma = gamma
        self.triplets_per_anchor = triplets_per_anchor
        self.soft_plus = torch.nn.Softplus(beta=1)


    def get_all_triplets_indices(self, labels, batch_size, ref_labels=None):
        if ref_labels is None:
            ref_labels = labels
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1
        if ref_labels is labels:
            matches -= torch.eye(matches.size(0)).byte().to(labels.device)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        a_idx = triplets.nonzero()[:, 0].flatten()
        p_idx = triplets.nonzero()[:, 1].flatten()
        n_idx = triplets.nonzero()[:, 2].flatten()

        filter_mask = ((a_idx % 4 == 0) & (a_idx < batch_size)) | (a_idx > batch_size)
        return a_idx[filter_mask], p_idx[filter_mask], n_idx[filter_mask]

    def forward(self, embeddings, labels, batch_size, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss (float)
        """
        indices_tuple = self.get_all_triplets_indices(labels, batch_size)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        self.num_triplets = len(anchor_idx)
        if self.num_triplets == 0:
            self.num_unique_anchors = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]

        # compute cosine similarities
        # since embeddings are normalized, we only need to compute dot product
        sp = torch.sum(anchors * positives, dim=1)
        sn = torch.sum(anchors * negatives, dim=1)

        # compute some constants
        loss = 0.
        op = 1 + self.m
        on = -self.m
        delta_p = 1 - self.m
        delta_n = self.m

        # find unique anchor index
        # for each unique anchor index, we have (sp1, sp2, ..., spK) (sn1, sn2, ..., snL)
        unique_anchor_idx = torch.unique(anchor_idx)
        self.num_unique_anchors = len(unique_anchor_idx)

        for anchor in unique_anchor_idx:
            mask = anchor_idx == anchor
            sp_for_this_anchor = sp[mask]
            sn_for_this_anchor = sn[mask]
            alpha_p = torch.clamp(op - sp_for_this_anchor.detach(), min=0.)
            alpha_n = torch.clamp(sn_for_this_anchor.detach() - on, min=0.)

            logit_p = -self.gamma * alpha_p * (sp_for_this_anchor - delta_p)
            logit_n = self.gamma * alpha_n * (sn_for_this_anchor - delta_n)

            loss_for_this_anchor = self.soft_plus(
                torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            loss += loss_for_this_anchor

        loss /= len(unique_anchor_idx)
        if loss == 0:
            loss = torch.sum(embeddings * 0)
        return loss