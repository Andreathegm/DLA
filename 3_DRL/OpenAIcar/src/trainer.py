import os
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from tqdm import tqdm

class PPOTrainer:
    def __init__(self, agent, optimizer, config, envs, buffer, device, writer, run_dir):
        self.agent = agent
        self.optimizer = optimizer
        self.config = config
        self.envs = envs
        self.buffer = buffer
        self.device = device
        self.writer = writer
        self.global_step = 0
        
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.recent_returns = deque(maxlen=20)
        self.best_mean_return = -float('inf')

    def save_checkpoint(self, is_best=False, update=None):
        state = {
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mean_return': self.best_mean_return,
        }
        
        if is_best:
            filepath = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, filepath)
        
        if update is not None:
            filepath = os.path.join(self.checkpoint_dir, f"model_update_{update}.pt")
            torch.save(state, filepath)

    def optimize_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        b_inds = np.arange(self.config.batch_size)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break
                
        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def train(self):
        start_time = time.time()
        next_obs,info = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.envs.num_envs).to(self.device)
        num_updates = self.config.total_timesteps // self.config.batch_size

        pbar = tqdm(range(1, num_updates + 1), desc="PPO Training")
        for update in pbar:
            if self.config.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.config.num_steps):
                self.global_step += self.envs.num_envs

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                
                next_obs_numpy, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                ## maybe put a flag here.
                #reward = np.clip(reward,max=1.0)
                done = np.logical_or(terminated,truncated)
                
                self.buffer.add(
                    next_obs, action, logprob, torch.tensor(reward).to(self.device).view(-1), 
                    next_done, value.flatten()
                )

                next_obs = torch.Tensor(next_obs_numpy).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                if "episode" in info and "_episode" in info:
                    mask = info["_episode"]
                    returns = info["episode"]["r"]
                    lengths = info["episode"]["l"]

                    for i in range(len(mask)):
                        if mask[i]: 
                            ep_return = returns[i].item()
                            ep_length = lengths[i].item()

                            self.recent_returns.append(ep_return)
                            
                            self.writer.add_scalar("charts/episodic_return", ep_return, self.global_step)
                            self.writer.add_scalar("charts/episodic_length", ep_length, self.global_step)

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
            
            returns, advantages = self.buffer.compute_returns_and_advantages(
                next_value, next_done, self.config.gamma, self.config.gae_lambda
            )

            flat_data = self.buffer.get_flattened_data(returns, advantages)
            pg_loss, v_loss, ent_loss, old_kl, approx_kl, clipfracs = self.optimize_policy(*flat_data)

            if len(self.recent_returns) > 0:
                mean_return = np.mean(self.recent_returns)
                self.writer.add_scalar("charts/mean_return_20_eps", mean_return, self.global_step)

                pbar.set_postfix({"Mean (last 20)": f"{mean_return:.3f}", "Best": f"{self.best_mean_return:.3f}"})
                
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save_checkpoint(is_best=True)

            if update % self.config.checkpoint_freq == 0:
                self.save_checkpoint(is_best=False, update=update)

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", ent_loss.item(), self.global_step)
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)
