"""
ULTRA-OPTIMIZED PPO with Safety Constraints
Fixes the over-transformation issue by:
1. Adding transformation bounds
2. Early stopping on divergence
3. Conservative action space
4. Better reward shaping
5. Penalizing excessive transformations
"""

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
from collections import deque
import time

from model import ImprovedA3CNet

# Data paths
DATA_PATH = Path(r"C:\Users\Admin\Desktop\RL Image Registration\data\train")
SAR_MIS_DIR = DATA_PATH / "sar_misaligned"
SAR_GT_DIR  = DATA_PATH / "sar_preprocessed"
OPT_DIR     = DATA_PATH / "optical_fixed"

sar_mis_paths = sorted(SAR_MIS_DIR.glob("*.png"))
sar_gt_paths  = sorted(SAR_GT_DIR.glob("*.png"))
opt_paths     = sorted(OPT_DIR.glob("*.png"))

print(f"📊 Training data: {len(sar_mis_paths)} triplets")


def read_gray(path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {path}")
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


class SafeRegistrationEnv:
    """Environment with safety constraints to prevent over-transformation"""
    def __init__(self, sar_mis_paths, sar_gt_paths, opt_paths,
                 size=256, max_steps=20):
        
        self.sar_mis_paths = sar_mis_paths
        self.sar_gt_paths  = sar_gt_paths
        self.opt_paths     = opt_paths
        self.size = size
        self.max_steps = max_steps
        
        # CRITICAL: Transformation bounds to prevent going out of bounds
        self.max_translation = 30  # Maximum 30 pixels in any direction
        self.max_rotation = 15     # Maximum 15 degrees
        
        # Conservative step sizes (smaller = safer)
        self.trans_step = 1.5  # pixels per step
        self.rot_step = 0.75   # degrees per step

    def similarity(self, fixed, moving):
        """NMI calculation"""
        try:
            h = cv2.calcHist([fixed, moving], [0, 1], None, [32, 32], [0, 1, 0, 1])
            pxy = h + 1e-10
            pxy /= pxy.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            px_py = px[:, None] * py[None, :] + 1e-10
            mi = np.sum(pxy * np.log(pxy / px_py))
            return np.clip(mi / 3.5, 0, 1)
        except:
            return 0.0

    def apply_transform(self, img, tx, ty, angle):
        """Apply transformation with border handling"""
        M = cv2.getRotationMatrix2D((self.size//2, self.size//2), angle, 1.0)
        M[0,2] += tx
        M[1,2] += ty
        return cv2.warpAffine(img, M, (self.size, self.size), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)

    def reset(self):
        idx = random.randint(0, len(self.sar_mis_paths)-1)
        
        self.fixed   = read_gray(self.opt_paths[idx])
        self.source  = read_gray(self.sar_mis_paths[idx])
        self.gt      = read_gray(self.sar_gt_paths[idx])
        
        self.tx, self.ty, self.angle = 0.0, 0.0, 0.0
        self.moving = self.source.copy()
        self.prev   = self.moving.copy()
        self.steps  = 0
        
        self.initial_mse = np.mean((self.source - self.gt) ** 2)
        self.initial_mi = self.similarity(self.fixed, self.source)
        self.best_mse = self.initial_mse
        self.best_mi = self.initial_mi
        
        self.mse_history = deque([self.initial_mse], maxlen=5)
        self.no_improvement_steps = 0
        self.total_movement = 0.0  # Track total transformation magnitude
        
        return self._state()

    def _state(self):
        """State with normalization"""
        fixed_norm = (self.fixed - self.fixed.mean()) / (self.fixed.std() + 1e-8)
        moving_norm = (self.moving - self.moving.mean()) / (self.moving.std() + 1e-8)
        s = np.stack([fixed_norm, moving_norm], axis=0)
        return torch.tensor(s, dtype=torch.float32)
    
    def _is_within_bounds(self, tx, ty, angle):
        """Check if transformation is within safe bounds"""
        if abs(tx) > self.max_translation or abs(ty) > self.max_translation:
            return False
        if abs(angle) > self.max_rotation:
            return False
        return True

    def step(self, action):
        """Step with safety constraints and improved reward"""
        
        # Action mapping (conservative steps)
        actions = {
            0: (0, -self.trans_step, 0),      # up
            1: (0, self.trans_step, 0),       # down
            2: (-self.trans_step, 0, 0),      # left
            3: (self.trans_step, 0, 0),       # right
            4: (0, 0, -self.rot_step),        # rotate CCW
            5: (0, 0, self.rot_step)          # rotate CW
        }
        
        dx, dy, da = actions[action]
        
        # Proposed new transformation
        new_tx = self.tx + dx
        new_ty = self.ty + dy
        new_angle = self.angle + da
        
        # CRITICAL: Check bounds before applying
        if not self._is_within_bounds(new_tx, new_ty, new_angle):
            # Hit boundary - large penalty and don't move
            return self._state(), -10.0, True
        
        # Apply transformation
        self.tx = new_tx
        self.ty = new_ty
        self.angle = new_angle
        
        self.moving = self.apply_transform(self.source, self.tx, self.ty, self.angle)
        
        # Calculate metrics
        current_mse = np.mean((self.moving - self.gt) ** 2)
        current_mi = self.similarity(self.fixed, self.moving)
        prev_mse = self.mse_history[-1]
        
        # Track total movement
        movement = np.sqrt(dx**2 + dy**2) + abs(da) * 0.5
        self.total_movement += movement
        
        # === IMPROVED REWARD CALCULATION ===
        reward = 0.0
        
        # 1. MSE improvement (primary signal) - SCALED DOWN
        mse_delta = prev_mse - current_mse
        reward += 200.0 * mse_delta  # Amplify good moves
        
        # 2. Absolute MSE bonus (encourage low MSE)
        if current_mse < 0.01:
            reward += 5.0
        elif current_mse < 0.02:
            reward += 2.0
        
        # 3. MI improvement
        mi_delta = current_mi - self.best_mi
        if mi_delta > 0:
            reward += 10.0 * mi_delta
            self.best_mi = current_mi
        
        # 4. Progress tracking
        improvement_ratio = (self.initial_mse - current_mse) / (self.initial_mse + 1e-8)
        
        if current_mse < self.best_mse:
            self.best_mse = current_mse
            self.no_improvement_steps = 0
            
            # Milestone bonuses
            if improvement_ratio > 0.8:
                reward += 10.0
            elif improvement_ratio > 0.6:
                reward += 6.0
            elif improvement_ratio > 0.4:
                reward += 3.0
            elif improvement_ratio > 0.2:
                reward += 1.0
        else:
            self.no_improvement_steps += 1
        
        # 5. CRITICAL: Penalize making things worse
        if mse_delta < -0.001:  # Making it worse
            penalty = min(abs(mse_delta) * 500, 10.0)
            reward -= penalty
        
        # 6. Penalize excessive movement (prevent going too far)
        if self.total_movement > 40:
            reward -= 0.5
        
        # 7. Penalize stagnation
        if self.no_improvement_steps >= 5:
            reward -= 2.0
        
        # 8. Small step penalty for efficiency
        reward -= 0.02
        
        # Track history
        self.mse_history.append(current_mse)
        self.prev = self.moving.copy()
        self.steps += 1
        
        # === TERMINATION CONDITIONS ===
        done = False
        
        # Success conditions
        if current_mse < 0.003:
            done = True
            reward += 50.0
            print(f"  ✅ Success! MSE: {current_mse:.6f}")
        elif current_mse < 0.005:
            done = True
            reward += 30.0
        
        # Max steps
        if self.steps >= self.max_steps:
            done = True
            
            # Final evaluation
            if improvement_ratio > 0.7:
                reward += 25.0
            elif improvement_ratio > 0.5:
                reward += 15.0
            elif improvement_ratio > 0.3:
                reward += 8.0
            elif improvement_ratio < -0.1:  # Made it worse
                reward -= 20.0
        
        # CRITICAL: Early stopping if diverging badly
        if current_mse > self.initial_mse * 1.5 and self.steps > 5:
            done = True
            reward -= 25.0
            print(f"  ❌ Diverged! Initial: {self.initial_mse:.6f}, Current: {current_mse:.6f}")
        
        # Severe stagnation
        if self.no_improvement_steps >= 10:
            done = True
            reward -= 10.0
        
        return self._state(), reward, done


class RolloutBuffer:
    """Efficient rollout buffer"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def store(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def get(self):
        return {
            'states': torch.stack(self.states),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'logprobs': torch.tensor(self.logprobs, dtype=torch.float32),
            'rewards': self.rewards,
            'dones': self.dones,
            'values': self.values
        }


class SafePPO:
    """PPO with conservative updates"""
    def __init__(self, policy_net, lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.03,
                 epochs=4, minibatch_size=64, max_grad_norm=0.5):
        
        self.policy = policy_net
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        
        self.buffer = RolloutBuffer()
    
    def select_action(self, state, h, c):
        """Conservative action selection with exploration"""
        with torch.no_grad():
            logits, value, (h_new, c_new) = self.policy(state.unsqueeze(0), h, c)
            
            # Conservative clipping
            logits = torch.clamp(logits, -10, 10)
            probs = torch.softmax(logits, dim=-1)
            
            # Ensure minimum exploration
            probs = probs + 1e-6
            probs = probs / probs.sum()
            
            dist = Categorical(probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            
        return action.item(), logprob.item(), value.item(), h_new, c_new
    
    def compute_gae(self, rewards, values, dones):
        """GAE computation"""
        advantages = []
        returns = []
        gae = 0
        
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """Conservative PPO update"""
        if len(self.buffer.states) < 2:
            return None
        
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        old_logprobs = data['logprobs']
        rewards = data['rewards']
        dones = data['dones']
        values = data['values']
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -5, 5)  # Conservative clipping
        
        # Statistics
        total_p_loss = 0
        total_v_loss = 0
        total_ent = 0
        update_count = 0
        
        # Multiple epochs
        for epoch in range(self.epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(indices), self.minibatch_size):
                end = min(start + self.minibatch_size, len(indices))
                batch_idx = indices[start:end]
                
                if len(batch_idx) < 2:
                    continue
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass
                h = torch.zeros(1, 256, 32, 32)
                c = torch.zeros(1, 256, 32, 32)
                
                logits_list = []
                values_list = []
                
                for i in range(len(batch_states)):
                    logits, value, (h, c) = self.policy(
                        batch_states[i:i+1],
                        h.detach(),
                        c.detach()
                    )
                    logits_list.append(logits)
                    values_list.append(value)
                
                batch_logits = torch.cat(logits_list, dim=0)
                batch_values = torch.cat(values_list, dim=0).squeeze(-1)
                
                # Safety checks
                if torch.isnan(batch_logits).any() or torch.isinf(batch_logits).any():
                    continue
                
                batch_logits = torch.clamp(batch_logits, -10, 10)
                probs = torch.softmax(batch_logits, dim=-1)
                probs = torch.clamp(probs, 1e-8, 1.0)
                dist = Categorical(probs)
                
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                ratio = torch.clamp(ratio, 0.1, 10.0)  # Conservative ratio clipping
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (batch_values - batch_returns).pow(2).mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_ent += entropy.item()
                update_count += 1
        
        self.scheduler.step()
        self.buffer.clear()
        
        if update_count > 0:
            return {
                'policy_loss': total_p_loss / update_count,
                'value_loss': total_v_loss / update_count,
                'entropy': total_ent / update_count,
                'lr': self.scheduler.get_last_lr()[0]
            }
        
        return None


def train_safe_ppo(num_episodes=10000, rollout_length=1024):
    """Training with safety constraints"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    Path("models").mkdir(exist_ok=True)
    
    # Model
    policy = ImprovedA3CNet(actions=6).to(device)
    
    # Environment with safety
    env = SafeRegistrationEnv(sar_mis_paths, sar_gt_paths, opt_paths, max_steps=20)
    
    # Agent with conservative settings
    agent = SafePPO(
        policy, 
        lr=1e-4,  # Lower learning rate
        clip_epsilon=0.2,
        epochs=4,
        minibatch_size=64,
        entropy_coef=0.03  # Higher entropy for more exploration
    )
    
    # Tracking
    episode_rewards = deque(maxlen=100)
    episode_improvements = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_final_mse = deque(maxlen=100)
    
    best_improvement = float('-inf')
    timestep = 0
    
    print(f"\n{'='*80}")
    print(f"🛡️ SAFE PPO TRAINING WITH CONSTRAINTS")
    print(f"{'='*80}")
    print(f"Model: ImprovedA3CNet")
    print(f"Max translation: ±30 pixels")
    print(f"Max rotation: ±15 degrees")
    print(f"Step size: 1.5 pixels, 0.75 degrees")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        state = state.to(device)
        h = torch.zeros(1, 256, 32, 32).to(device)
        c = torch.zeros(1, 256, 32, 32).to(device)
        
        episode_reward = 0
        episode_length = 0
        
        for t in range(env.max_steps):
            action, logprob, value, h, c = agent.select_action(state, h, c)
            
            next_state, reward, done = env.step(action)
            next_state = next_state.to(device)
            
            agent.buffer.store(state, action, logprob, reward, done, value)
            
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            state = next_state
            
            # Update
            if timestep % rollout_length == 0:
                metrics = agent.update()
                
                if metrics and episode % 100 == 0:
                    print(f"  📊 P-Loss: {metrics['policy_loss']:.4f} | "
                          f"V-Loss: {metrics['value_loss']:.4f} | "
                          f"Ent: {metrics['entropy']:.4f} | "
                          f"LR: {metrics['lr']:.2e}")
            
            if done:
                break
        
        # Metrics
        improvement_pct = (env.initial_mse - env.best_mse) / (env.initial_mse + 1e-8) * 100
        
        episode_rewards.append(episode_reward)
        episode_improvements.append(improvement_pct)
        episode_lengths.append(episode_length)
        episode_final_mse.append(env.best_mse)
        
        # Save best
        if improvement_pct > best_improvement:
            best_improvement = improvement_pct
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode,
                'best_improvement': best_improvement
            }, 'models/registration_ppo_safe_best.pth')
        
        # Logging
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_improvement = np.mean(episode_improvements)
            avg_length = np.mean(episode_lengths)
            avg_final_mse = np.mean(episode_final_mse)
            elapsed = time.time() - start_time
            
            print(f"[Ep {episode:5d}] "
                  f"R: {episode_reward:+8.2f} | "
                  f"AvgR: {avg_reward:+8.2f} | "
                  f"MSE↓: {improvement_pct:+6.1f}% | "
                  f"Avg↓: {avg_improvement:+6.1f}% | "
                  f"FinalMSE: {avg_final_mse:.5f} | "
                  f"Len: {avg_length:.1f} | "
                  f"Time: {elapsed/60:.1f}m")
            
            if avg_improvement > 30:
                print(f"  🌟 EXCELLENT: {avg_improvement:.1f}%!")
            elif avg_improvement > 15:
                print(f"  ✅ GOOD: {avg_improvement:.1f}%")
            elif avg_improvement > 5:
                print(f"  ⚡ Moderate: {avg_improvement:.1f}%")
            elif avg_improvement < -5:
                print(f"  ⚠️ WARNING: Negative improvement {avg_improvement:.1f}%")
        
        # Checkpoint
        if episode % 500 == 0 and episode > 0:
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode,
            }, f'models/registration_ppo_safe_ep{episode}.pth')
    
    # Final save
    torch.save(policy.state_dict(), 'models/registration_ppo_safe_final.pth')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 SAFE PPO TRAINING COMPLETE!")
    print("="*80)
    print(f"Best improvement: {best_improvement:.2f}%")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_safe_ppo(num_episodes=10000, rollout_length=1024)