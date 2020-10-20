import torch
import numpy as np

EPS = 1e-8

class TrajStorage(object):
    def __init__(self, rollouts, aug_fn=None):
      trajs = []
      num_processes = rollouts.obs.shape[1]
      for env_index in range(num_processes):
        env_masks = rollouts.masks[:, env_index]
        env_obs = rollouts.obs[:, env_index]
        env_actions = rollouts.pi[:, env_index]
        env_recurrent_states = rollouts.recurrent_hidden_states[:, env_index]

        masks_indices = 1 - env_masks[:, 0]
        indices = masks_indices.nonzero()[:, 0].tolist()
        if len(indices) < 2:
          continue
        prev_index = indices[0]
        for index in indices[1:]:
          obs = env_obs[prev_index: index]
          actions = env_actions[prev_index: index]
          traj_masks = env_masks[prev_index: index]
          recurrent_states = env_recurrent_states[prev_index: index]
          prev_index = index
          trajs.append(((obs, recurrent_states, traj_masks), actions))

      self.trajs = trajs
      self.num_trajs = len(trajs)
      self.aug_fn = aug_fn

    def sample_traj_pair(self):
        idx1, idx2 = np.random.randint(0, self.num_trajs, 2)
        traj1, traj2 = self.trajs[idx1], self.trajs[idx2]
        if self.aug_fn is not None:
          obs_aug1 = self.aug_fn.do_augmentation(traj1[0][0])
          obs_aug2 = self.aug_fn.do_augmentation(traj2[0][0])
          traj_inputs1 = (obs_aug1, traj1[0][1], traj1[0][2])
          traj_inputs2 = (obs_aug2, traj2[0][1], traj2[0][2])
          traj1, traj2 = (traj_inputs1, traj1[1]), (traj_inputs2, traj2[1])
        return traj1, traj2


def metric_fixed_point(cost_matrix, gamma=0.99, eps=1e-7):
  """DP for calculating PSM (approximately).

  Args:
    cost_matrix: DIST matrix where entries at index (i, j) is DIST(x_i, y_j)
    gamma: Metric discount factor.
    eps: Threshold for stopping the fixed point iteration.
  """
  d = torch.zeros_like(cost_matrix)
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if torch.sum(torch.abs(d - d_new)) < eps:
      break
    else:
      d = d_new[:]
  return d


def _calculate_action_cost_matrix(actions_1, actions_2):
  diff = torch.unsqueeze(actions_1, dim=1) - torch.unsqueeze(actions_2, dim=0)
  tv_distance = 0.5 * torch.sum(torch.abs(diff), dim=2)
  return tv_distance


def contrastive_loss(similarity_matrix,
                     metric_values,
                     temperature=1.0,
                     beta=1.0):
  """Contrative Loss with embedding similarity ."""
  metric_shape = metric_values.shape
  ## z_\theta(X): embedding_1 = nn_model.representation(X)
  ## z_\theta(Y): embedding_2 = nn_model.representation(Y)
  ## similarity_matrix = cosine_similarity(embedding_1, embedding_2)
  ## metric_values = PSM(X, Y)
  soft_similarity_matrix = similarity_matrix / temperature

  col_indices = torch.argmin(metric_values, dim=1)
  pos_indices1 = (torch.arange(start=0, end=metric_shape[0],
                              dtype=torch.int64), col_indices)

  metric_values /= beta
  similarity_measure = torch.exp(-metric_values)
  pos_weights1 = -metric_values[pos_indices1]
  pos_logits1 = soft_similarity_matrix[pos_indices1] + pos_weights1
  negative_weights = torch.log((1.0 - similarity_measure) + 1e-8)
  negative_weights[pos_indices1] +=  pos_weights1
  neg_logits1 = soft_similarity_matrix + negative_weights

  neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
  return torch.mean(neg_logits1 - pos_logits1) # Equation 4

def representation_alignment_loss(nn_model,
                                  traj_tuple,
                                  coupling_temperature=0.1,
                                  gamma = 0.1,
                                  temperature=1.0):
  (inputs1, ac1), (inputs2, ac2) = traj_tuple
  metric_vals = compute_metric(ac1, ac2, gamma=gamma)

  representation_1 = nn_model.representation(*inputs1)
  representation_2 = nn_model.representation(*inputs2)
  similarity_matrix = _cosine_similarity(representation_1, representation_2)

  alignment_loss = contrastive_loss(
      similarity_matrix,
      metric_vals,
      temperature=temperature,
      beta=coupling_temperature)

  return alignment_loss

def compute_metric(actions1, actions2, gamma):
  action_cost = _calculate_action_cost_matrix(actions1, actions2)
  return metric_fixed_point(action_cost, gamma=gamma)

def _cosine_similarity(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = torch.unsqueeze(x, dim=1), torch.unsqueeze(y, dim=0)
  similarity_matrix = torch.sum(x_expanded * y_expanded, dim=-1)
  similarity_matrix /= (
      torch.norm(x_expanded, dim=-1) * torch.norm(y_expanded, dim=-1) + EPS)
  return similarity_matrix