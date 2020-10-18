import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
from ucb_rl2_meta.algo import contrastive_helpers as chs

class RAD():
    """
    RAD object.
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_id=None,
                 aug_func=None,
                 env_name=None,
                 pse_gamma=0.1,
                 pse_coef=0.1,
                 pse_temperature=0.1):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.aug_id = aug_id
        self.aug_func = aug_func

        self.env_name = env_name

        self.pse_gamma = pse_gamma
        self.pse_coef = pse_coef
        self.pse_temperature = pse_temperature

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        pse_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            if self.pse_coef > 0:
                trajs_sampler = chs.TrajStorage(rollouts, aug_fn=self.aug_func)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                # RAD simply applies augmentation
                obs_batch = self.aug_func.do_augmentation(obs_batch)
                obs_batch_id = self.aug_id(obs_batch)

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()

                # Compute PSE loss (naive attempt)
                if self.pse_coef > 0:
                    aug_traj_tuple =  trajs_sampler.sample_traj_pair()
                    pse_loss = chs.representation_alignment_loss(
                        self.actor_critic, aug_traj_tuple,
                        temperature=self.pse_temperature,
                        gamma=self.pse_gamma)
                else:
                    pse_loss = torch.zeros(1)

                # Update actor-critic using both PPO and Augmented Loss
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef +
                    pse_loss * self.pse_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                pse_loss_epoch += pse_loss.item()

                if self.aug_func:
                    self.aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        pse_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, pse_loss_epoch
