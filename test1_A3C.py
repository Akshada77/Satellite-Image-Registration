"""
FIXED test_A3C.py
Corrected to use A3CNet (basic) to match training
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import correct model architecture - FIXED
from model import A3CNet  # Use basic A3CNet, not Improved


class RegistrationEnv:
    """Simplified environment for testing"""
    def __init__(self, sar_mis_paths, sar_gt_paths, opt_paths,
                 size=256, max_steps=20):
        self.sar_mis_paths = sar_mis_paths
        self.sar_gt_paths  = sar_gt_paths
        self.opt_paths     = opt_paths
        self.size = size
        self.max_steps = max_steps

    def apply_transform(self, img, tx, ty, angle):
        M = cv2.getRotationMatrix2D(
            (self.size//2, self.size//2), angle, 1.0
        )
        M[0,2] += tx
        M[1,2] += ty
        return cv2.warpAffine(img, M, (self.size, self.size))

    def _state(self):
        s = np.stack([self.fixed, self.moving], axis=0)
        return torch.tensor(s, dtype=torch.float32)


def read_gray(path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {path}")
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def load_model_safe(model_path, model):
    """Safe model loading for PyTorch 2.6+"""
    try:
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint (episode: {checkpoint.get('episode', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded model weights")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_single_image(model, env, idx, save_path='test_result_a3c.png'):
    """Test the model on a single image and visualize results"""
    
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Testing A3C on image index: {idx}")
    print(f"Image: {env.sar_mis_paths[idx].name}")
    print(f"{'='*60}\n")
    
    # Load images
    env.fixed  = read_gray(env.opt_paths[idx])
    env.source = read_gray(env.sar_mis_paths[idx])
    env.gt     = read_gray(env.sar_gt_paths[idx])
    
    # Initialize
    env.moving = env.source.copy()
    env.tx = env.ty = env.angle = 0
    
    # Calculate initial metrics
    initial_mse = np.mean((env.source - env.gt) ** 2)
    
    # Run agent - FIXED: Use 128 for A3CNet
    state = env._state()
    h = torch.zeros(1, 128, 32, 32)  # 128 for A3CNet
    c = torch.zeros_like(h)
    
    trajectory = []
    actions_taken = []
    
    with torch.no_grad():
        for step in range(env.max_steps):
            logits, value, (h, c) = model(state.unsqueeze(0), h, c)
            action = torch.argmax(logits, -1).item()
            actions_taken.append(action)
            
            action_map = {
                0:(0,-2,0), 1:(0,2,0),
                2:(-2,0,0), 3:(2,0,0),
                4:(0,0,-1.0), 5:(0,0,1.0)
            }
            dx, dy, da = action_map[action]
            env.tx += dx
            env.ty += dy
            env.angle += da
            
            env.moving = env.apply_transform(env.source, env.tx, env.ty, env.angle)
            trajectory.append(env.moving.copy())
            
            state = env._state()
    
    # Calculate final metrics
    final_mse = np.mean((trajectory[-1] - env.gt) ** 2)
    improvement = (initial_mse - final_mse) / initial_mse * 100
    
    # Print statistics
    print(f"Initial MSE:  {initial_mse:.6f}")
    print(f"Final MSE:    {final_mse:.6f}")
    print(f"Improvement:  {improvement:.2f}%")
    print(f"\nFinal Transform:")
    print(f"  Translation: ({env.tx:+.1f}, {env.ty:+.1f}) pixels")
    print(f"  Rotation:    {env.angle:+.2f} degrees")
    
    # Action analysis
    action_names = ['Up', 'Down', 'Left', 'Right', 'Rotate CCW', 'Rotate CW']
    action_counts = np.bincount(actions_taken, minlength=6)
    print(f"\nAction distribution:")
    for name, count in zip(action_names, action_counts):
        print(f"  {name:12s}: {count:2d} ({count/len(actions_taken)*100:.1f}%)")
    
    # Create overlay images
    def create_overlay(img1, img2):
        overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.float32)
        overlay[:,:,0] = img1  # Red
        overlay[:,:,1] = img2  # Green
        return overlay
    
    overlay_optical_gt = create_overlay(env.fixed, env.gt)
    overlay_optical_registered = create_overlay(env.fixed, trajectory[-1])
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(env.fixed, cmap='gray')
    axes[0,0].set_title("Optical Image\n(Fixed Reference)", 
                       fontweight='bold', fontsize=13)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(env.gt, cmap='gray')
    axes[0,1].set_title("SAR Ground Truth\n(Target Alignment)", 
                       fontweight='bold', fontsize=13)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(env.source, cmap='gray')
    axes[0,2].set_title(f"SAR Misaligned\nInitial MSE: {initial_mse:.6f}", 
                       fontweight='bold', fontsize=13)
    axes[0,2].axis('off')
    
    axes[1,0].imshow(trajectory[-1], cmap='gray')
    axes[1,0].set_title(f"Registered Image (A3C)\nFinal MSE: {final_mse:.6f}\nImprovement: {improvement:+.1f}%", 
                       fontweight='bold', fontsize=13,
                       color='green' if improvement > 0 else 'red')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(overlay_optical_gt)
    axes[1,1].set_title("Overlay: Optical + SAR GT\n(Red=Optical, Green=SAR GT, Yellow=Aligned)", 
                       fontweight='bold', fontsize=13)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(overlay_optical_registered)
    axes[1,2].set_title("Overlay: Optical + Registered\n(Red=Optical, Green=Registered, Yellow=Aligned)", 
                       fontweight='bold', fontsize=13)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.show()
    
    return {
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'improvement': improvement,
        'actions': actions_taken,
        'final_transform': (env.tx, env.ty, env.angle)
    }


def test_all_images(model, env, results_dir):
    """Test on ALL test images"""
    
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    num_tests = len(env.sar_mis_paths)
    
    print(f"\n{'='*60}")
    print(f"Testing A3C model on {num_tests} test images...")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx in range(num_tests):
        print(f"Processing {idx+1}/{num_tests}...", end=' ')
        
        model.eval()
        env.fixed  = read_gray(env.opt_paths[idx])
        env.source = read_gray(env.sar_mis_paths[idx])
        env.gt     = read_gray(env.sar_gt_paths[idx])
        
        env.moving = env.source.copy()
        env.tx = env.ty = env.angle = 0
        
        initial_mse = np.mean((env.source - env.gt) ** 2)
        
        state = env._state()
        h = torch.zeros(1, 128, 32, 32)  # 128 for A3CNet
        c = torch.zeros_like(h)
        
        actions_taken = []
        
        with torch.no_grad():
            for step in range(env.max_steps):
                logits, value, (h, c) = model(state.unsqueeze(0), h, c)
                action = torch.argmax(logits, -1).item()
                actions_taken.append(action)
                
                action_map = {
                    0:(0,-2,0), 1:(0,2,0),
                    2:(-2,0,0), 3:(2,0,0),
                    4:(0,0,-1.0), 5:(0,0,1.0)
                }
                dx, dy, da = action_map[action]
                env.tx += dx
                env.ty += dy
                env.angle += da
                
                env.moving = env.apply_transform(env.source, env.tx, env.ty, env.angle)
                state = env._state()
        
        final_mse = np.mean((env.moving - env.gt) ** 2)
        improvement = (initial_mse - final_mse) / initial_mse * 100
        
        results.append({
            'initial_mse': initial_mse,
            'final_mse': final_mse,
            'improvement': improvement,
            'actions': actions_taken,
            'final_transform': (env.tx, env.ty, env.angle)
        })
        
        print("✓")
    
    # Aggregate statistics
    improvements = [r['improvement'] for r in results]
    initial_mses = [r['initial_mse'] for r in results]
    final_mses = [r['final_mse'] for r in results]
    
    print(f"\n{'='*60}")
    print(f"A3C MODEL - TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"\nMSE Improvement:")
    print(f"  Mean:   {np.mean(improvements):+.2f}%")
    print(f"  Median: {np.median(improvements):+.2f}%")
    print(f"  Std:    {np.std(improvements):.2f}%")
    print(f"  Min:    {np.min(improvements):+.2f}%")
    print(f"  Max:    {np.max(improvements):+.2f}%")
    
    success_5 = sum(1 for imp in improvements if imp > 5)
    success_10 = sum(1 for imp in improvements if imp > 10)
    success_30 = sum(1 for imp in improvements if imp > 30)
    print(f"\nSuccess Rate:")
    print(f"  >5%:  {success_5}/{num_tests} ({success_5/num_tests*100:.1f}%)")
    print(f"  >10%: {success_10}/{num_tests} ({success_10/num_tests*100:.1f}%)")
    print(f"  >30%: {success_30}/{num_tests} ({success_30/num_tests*100:.1f}%)")
    
    # Save results to JSON
    results_json = {
        'algorithm': 'A3C',
        'num_tests': num_tests,
        'mean_improvement': float(np.mean(improvements)),
        'median_improvement': float(np.median(improvements)),
        'std_improvement': float(np.std(improvements)),
        'min_improvement': float(np.min(improvements)),
        'max_improvement': float(np.max(improvements)),
        'success_rate_5pct': float(success_5/num_tests),
        'success_rate_10pct': float(success_10/num_tests),
        'success_rate_30pct': float(success_30/num_tests),
        'improvements': improvements,
        'initial_mses': initial_mses,
        'final_mses': final_mses
    }
    
    with open(results_dir / 'a3c_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_dir / 'a3c_results.json'}")
    
    # Plot statistics
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(improvements, bins=30, edgecolor='black', alpha=0.7, color='red')
    plt.axvline(np.mean(improvements), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(improvements):.1f}%')
    plt.xlabel('Improvement (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('A3C: Distribution of Improvements', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(initial_mses, final_mses, alpha=0.6, color='red', s=20)
    plt.plot([min(initial_mses), max(initial_mses)], 
             [min(initial_mses), max(initial_mses)], 
             'k--', linewidth=2, label='No improvement')
    plt.xlabel('Initial MSE', fontsize=12)
    plt.ylabel('Final MSE', fontsize=12)
    plt.title('A3C: Initial vs Final MSE', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    categories = ['>5%', '>10%', '>30%']
    rates = [success_5/num_tests*100, success_10/num_tests*100, success_30/num_tests*100]
    bars = plt.bar(categories, rates, color='red', alpha=0.7, edgecolor='black')
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('A3C: Success Rates', fontweight='bold', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'a3c_statistics.png', dpi=150, bbox_inches='tight')
    print(f"✓ Statistics plot saved to: {results_dir / 'a3c_statistics.png'}")
    plt.show()
    
    return results


if __name__ == "__main__":
    
    # Paths
    DATA_PATH = Path(r"C:\Users\Admin\Desktop\RL Image Registration\data\test")
    MODEL_PATH = Path(r"C:\Users\Admin\Desktop\RL Image Registration\experiments\models\registration_model_a3c_best.pth")
    RESULTS_DIR = Path(r"C:\Users\Admin\Desktop\RL Image Registration\results\A3C_results")
    
    SAR_MIS_DIR = DATA_PATH / "sar_misaligned"
    SAR_GT_DIR  = DATA_PATH / "sar_preprocessed"
    OPT_DIR     = DATA_PATH / "optical_fixed"
    
    # Load TEST data
    print("="*60)
    print("A3C MODEL TESTING")
    print("="*60)
    print("\nLoading test data...")
    
    sar_mis_paths = sorted(SAR_MIS_DIR.glob("*.png"))
    sar_gt_paths  = sorted(SAR_GT_DIR.glob("*.png"))
    opt_paths     = sorted(OPT_DIR.glob("*.png"))
    
    assert len(sar_mis_paths) == len(sar_gt_paths) == len(opt_paths), \
           "Mismatch in test images"
    
    print(f"✓ Found {len(sar_mis_paths)} test image triplets")
    
    # Load model
    print(f"\nLoading A3C model from: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print(f"Please run train_A3C.py first!")
        exit(1)
    
    # FIXED: Use basic A3CNet
    model = A3CNet()
    
    if not load_model_safe(MODEL_PATH, model):
        print("\n❌ Failed to load model. Please check the model file.")
        exit(1)
    
    model.eval()
    print("✓ A3C model loaded successfully!")
    
    # Create environment
    env = RegistrationEnv(sar_mis_paths, sar_gt_paths, opt_paths)
    
    # Run tests
    print("\n" + "="*60)
    print("Choose test mode:")
    print("="*60)
    print("1. Test single image (by index)")
    print("2. Test ALL test images (recommended)")
    print("="*60)
    
    choice = input("\nEnter choice (1/2) [default=2]: ").strip() or "2"
    
    if choice == "1":
        idx = int(input(f"Enter index (0-{len(sar_mis_paths)-1}): "))
        test_single_image(model, env, idx, RESULTS_DIR / 'single_test_a3c.png')
    else:
        test_all_images(model, env, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("A3C Testing Complete!")
    print("="*60)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("\nNext step: Run test_PPO.py, then compare_algorithms.py!")
    print("="*60 + "\n")