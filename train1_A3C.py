"""
FIXED A3C Training Script
Resolves gradient computation errors
"""

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from pathlib import Path
from collections import deque

from model import A3CNet

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


class ImprovedRegistrationEnv:
    """Improved environment with balanced rewards"""
    def __init__(self, sar_mis_paths, sar_gt_paths, opt_paths, size=256, max_steps=20):
        self.sar_mis_paths = sar_mis_paths
        self.sar_gt_paths  = sar_gt_paths
        self.opt_paths     = opt_paths
        self.size = size
        self.max_steps = max_steps

    def similarity(self, fixed, moving):
        """Normalized Mutual Information"""
        try:
            h, _, _ = np.histogram2d(fixed.ravel(), moving.ravel(), bins=32)
            pxy = h + 1e-10
            pxy /= pxy.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            px_py = px[:, None] * py[None, :]
            mi = np.sum(pxy * np.log(pxy / (px_py + 1e-10)))
            normalized_mi = np.clip(mi / 3.0, 0, 1)
            return normalized_mi
        except:
            return 0.0

    def apply_transform(self, img, tx, ty, angle):
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
        
        self.mse_history = [self.initial_mse]
        self.action_history = []
        
        return self._state()

    def _state(self):
        s = np.stack([self.fixed, self.moving], axis=0)
        return torch.tensor(s, dtype=torch.float32)

    def step(self, action):
        """Balanced reward system"""
        actions = {
            0: (0, -2, 0),      1: (0, 2, 0),
            2: (-2, 0, 0),      3: (2, 0, 0),
            4: (0, 0, -1.0),    5: (0, 0, 1.0)
        }
        
        dx, dy, da = actions[action]
        self.tx += dx
        self.ty += dy
        self.angle += da
        
        self.moving = self.apply_transform(self.source, self.tx, self.ty, self.angle)
        
        current_mse = np.mean((self.moving - self.gt) ** 2)
        current_mi = self.similarity(self.fixed, self.moving)
        
        prev_mse = np.mean((self.prev - self.gt) ** 2)
        prev_mi = self.similarity(self.fixed, self.prev)
        
        # Balanced reward
        reward = 0.0
        
        mse_improvement = (prev_mse - current_mse)
        reward += 50.0 * mse_improvement
        
        mi_improvement = current_mi - prev_mi
        reward += 5.0 * mi_improvement
        
        if current_mse < self.best_mse:
            improvement_ratio = (self.best_mse - current_mse) / (self.initial_mse + 1e-8)
            reward += 2.0 * improvement_ratio
            self.best_mse = current_mse
        
        if current_mi > self.best_mi:
            reward += 0.5
            self.best_mi = current_mi
        
        self.mse_history.append(current_mse)
        if len(self.mse_history) > 3:
            recent_mse = self.mse_history[-3:]
            if np.var(recent_mse) > 0.0001:
                reward -= 0.5
        
        if mse_improvement < -0.001:
            reward -= 1.0
        
        reward -= 0.005
        
        self.action_history.append(action)
        if len(self.action_history) >= 5:
            recent_actions = self.action_history[-5:]
            unique_actions = len(set(recent_actions))
            if unique_actions >= 4:
                reward += 0.2
            elif unique_actions == 1:
                reward -= 0.3
        
        self.prev = self.moving.copy()
        self.steps += 1
        
        done = False
        
        if self.steps >= self.max_steps:
            done = True
        
        if current_mse < 0.005:
            done = True
            reward += 20.0
        
        if current_mse > self.initial_mse * 1.5 and self.steps > 5:
            done = True
            reward -= 10.0
        
        if done:
            final_improvement = (self.initial_mse - current_mse) / (self.initial_mse + 1e-8)
            
            if final_improvement > 0.7:
                reward += 15.0
            elif final_improvement > 0.5:
                reward += 10.0
            elif final_improvement > 0.3:
                reward += 5.0
            elif final_improvement > 0.1:
                reward += 2.0
            elif final_improvement < -0.1:
                reward -= 10.0
        
        return self._state(), reward, done


def worker(rank, global_net, optimizer, counter, lock, stats, 
           gamma=0.99, episodes=2500, gae_lambda=0.95, save_interval=500):
    """Worker with FIXED gradient computation"""
    
    torch.manual_seed(rank + 100)
    np.random.seed(rank + 100)
    
    env = ImprovedRegistrationEnv(sar_mis_paths, sar_gt_paths, opt_paths)
    
    local_net = A3CNet()
    local_net.load_state_dict(global_net.state_dict())
    
    episode_rewards = deque(maxlen=100)
    episode_improvements = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    for ep in range(episodes):
        state = env.reset()
        h = torch.zeros(1, 128, 32, 32)
        c = torch.zeros_like(h)
        
        logp, vals, rews, ents = [], [], [], []
        
        # Episode rollout - CRITICAL: No torch.no_grad() here!
        for t in range(env.max_steps):
            # Forward pass WITH gradients
            logits, val, (h_new, c_new) = local_net(state.unsqueeze(0), h, c)
            
            # Exploration noise
            if ep < 500:
                noise = torch.randn_like(logits) * 0.1
                logits = logits + noise
            
            prob = torch.softmax(logits, -1)
            dist = Categorical(prob)
            act = dist.sample()
            
            next_state, r, done = env.step(act.item())
            
            # Store for loss computation
            logp.append(dist.log_prob(act))
            vals.append(val)
            rews.append(r)
            ents.append(dist.entropy())
            
            # Detach hidden states for next step
            h, c = h_new.detach(), c_new.detach()
            state = next_state
            
            if done:
                break
        
        # Skip if episode too short
        if len(rews) < 2:
            continue
        
        # Compute GAE
        returns = []
        advantages = []
        gae = 0
        
        vals_np = torch.cat(vals).detach().squeeze().numpy()
        
        for step in reversed(range(len(rews))):
            next_val = 0 if step == len(rews) - 1 else vals_np[step + 1]
            delta = rews[step] + gamma * next_val - vals_np[step]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + vals_np[step])
        
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        logp_tensor = torch.stack(logp)
        vals_tensor = torch.cat(vals)
        ents_tensor = torch.stack(ents)
        
        # Verify gradients
        if not logp_tensor.requires_grad:
            print(f"[Worker {rank}] Warning: No gradients in episode {ep}, skipping")
            continue
        
        policy_loss = -(logp_tensor * advantages.detach()).mean()
        value_loss = 0.5 * (vals_tensor - returns).pow(2).mean()
        entropy_loss = -0.01 * ents_tensor.mean()
        
        loss = policy_loss + value_loss + entropy_loss
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Worker {rank}] Warning: Invalid loss at episode {ep}, skipping")
            continue
        
        # Update global network with error handling
        optimizer.zero_grad()
        
        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), 0.5)
            
            # Sync gradients
            for global_param, local_param in zip(global_net.parameters(), 
                                                local_net.parameters()):
                if local_param.grad is not None:
                    if global_param.grad is None:
                        global_param._grad = local_param.grad.clone()
                    else:
                        global_param._grad.add_(local_param.grad)
            
            optimizer.step()
            local_net.load_state_dict(global_net.state_dict())
            
        except RuntimeError as e:
            print(f"[Worker {rank}] Backward error at episode {ep}: {e}")
            local_net.load_state_dict(global_net.state_dict())
            continue
        
        # Track metrics
        total_reward = sum(rews)
        improvement_pct = (env.initial_mse - env.best_mse) / (env.initial_mse + 1e-8) * 100
        
        episode_rewards.append(total_reward)
        episode_improvements.append(improvement_pct)
        episode_lengths.append(len(rews))
        
        with lock:
            counter.value += 1
            stats['rewards'].append(total_reward)
            stats['improvements'].append(improvement_pct)
            
            if improvement_pct > stats['best_improvement']:
                stats['best_improvement'] = improvement_pct
                if rank == 0:
                    torch.save(global_net.state_dict(), 
                             'models/registration_model_a3c_best.pth')
            
            # Checkpoint saving
            if rank == 0 and (ep + 1) % save_interval == 0:
                checkpoint_path = f'models/registration_model_a3c_ep{counter.value}.pth'
                torch.save({
                    'episode': counter.value,
                    'model_state_dict': global_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_improvement': stats['best_improvement'],
                    'avg_improvement': np.mean(episode_improvements) if episode_improvements else 0,
                    'avg_reward': np.mean(episode_rewards) if episode_rewards else 0
                }, checkpoint_path)
                print(f"\n  💾 Checkpoint: {checkpoint_path}")
                print(f"     Episodes: {counter.value}, Best: {stats['best_improvement']:.2f}%\n")
        
        # Logging
        if rank == 0 and ep % 20 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_improvement = np.mean(episode_improvements) if episode_improvements else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            
            print(f"[W{rank}] Ep {ep:4d} | "
                  f"R: {total_reward:+7.2f} | "
                  f"AvgR: {avg_reward:+7.2f} | "
                  f"MSE↓: {improvement_pct:+6.1f}% | "
                  f"AvgMSE↓: {avg_improvement:+6.1f}% | "
                  f"Len: {avg_length:.1f} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Total: {counter.value}")
            
            if avg_improvement > 10:
                print(f"  ✅ POSITIVE: {avg_improvement:.1f}%")
            elif avg_improvement > 0:
                print(f"  ⚡ Slight positive: {avg_improvement:.1f}%")
            else:
                print(f"  ⚠️  Negative: {avg_improvement:.1f}%")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    Path("models").mkdir(exist_ok=True)
    
    global_net = A3CNet()
    global_net.share_memory()
    
    optimizer = optim.Adam(global_net.parameters(), lr=5e-4, eps=1e-5)
    
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    manager = mp.Manager()
    stats = manager.dict()
    stats['rewards'] = manager.list()
    stats['improvements'] = manager.list()
    stats['best_improvement'] = -float('inf')
    
    num_workers = 4
    episodes_per_worker = 500
    
    print(f"\n{'='*70}")
    print(f"🚀 FIXED A3C TRAINING")
    print(f"{'='*70}")
    print(f"Workers: {num_workers}")
    print(f"Episodes per worker: {episodes_per_worker}")
    print(f"Total episodes: {num_workers * episodes_per_worker}")
    print(f"Learning rate: 5e-4")
    print(f"Checkpoint interval: 500 episodes")
    print(f"{'='*70}")
    print(f"Key Fixes:")
    print(f"  ✅ Removed torch.no_grad() during rollout")
    print(f"  ✅ Added gradient verification")
    print(f"  ✅ Error handling for backward pass")
    print(f"  ✅ Safety checks for NaN/Inf")
    print(f"{'='*70}\n")
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(i, global_net, optimizer, counter, lock, stats),
            kwargs={'save_interval': 500}
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Total episodes: {counter.value}")
    print(f"Best improvement: {stats['best_improvement']:.2f}%")
    
    # Final save
    models_dir = Path("models")
    torch.save(global_net.state_dict(), models_dir / 'registration_model_a3c_final.pth')
    print(f"✓ Final model saved!")
    
    checkpoint_files = sorted(models_dir.glob('registration_model_a3c_ep*.pth'))
    print(f"\n📁 Saved {len(checkpoint_files)} checkpoints")
    print("="*70 + "\n")