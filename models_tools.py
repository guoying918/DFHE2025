import torch
from tools.utils import *
import torch

def MD_distance_0(support_proto, support_feature, support_labels, query_features):

    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)
    class_means = support_proto
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    return sample_logits

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations = {}
    class_precision_matrices = {}
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        
        flat_class_mask_indices = torch.reshape(class_mask_indices, (-1,)) # 45
        assert torch.all(flat_class_mask_indices >= 0), "Error: class_mask_indices contains negative values"
        assert torch.all(flat_class_mask_indices < context_features.size(0)), "Error: class_mask_indices contains out-of-range values"

        class_features = torch.index_select(context_features, 0, flat_class_mask_indices.cuda())
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        # class_precision_matrices[c.item()] = torch.inverse(
        #     torch.eye(class_features.size(1), class_features.size(1)).cuda())
        class_precision_matrices[c] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)).cuda())
    return class_representations,class_precision_matrices

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()