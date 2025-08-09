import argparse
import time
import types
import matplotlib.pyplot as plt
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run
import os
import gc

# Disable parallelism in tokenizers to avoid potential deadlocks in distributed training
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set MLX GPU memory growth to help with memory management
os.environ["MLX_GPU_MEMORY_GROWTH"] = "1"

# This is how we define the "world" of our distributed training. MLX needs to know that we're using MPI, and it can figure out the rest
world = mx.distributed.init()
size = world.size()

def all_reduce_grads(grads):
    # I added this check so that we can easily run this script as a single process. Size is always 1 if we only have one slot, or aren't using MPI
    if size == 1:
        return grads
    
    # Sum across all ranks, then divide with retry logic
    try:
        # Clear cache before distributed operation to avoid memory conflicts
        mx.clear_cache() 
        reduced_grads = tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)
        return reduced_grads
    except Exception as e:
        print(f"Rank {world.rank()}: Error in gradient reduction: {e}")
        # Fallback to local gradients if distributed reduction fails
        return grads

# We need this to extend the TrainingCallback class in order to add our custom gradient averaging function
class MetricsCallback(TrainingCallback):

    def __init__(self):
        # Initialize lists for loss tracking
        self.train_losses = []
        self.val_losses = []
        self.validation_failures = 0
        self.max_validation_failures = 2

        # This runs after backwards pass but before optimizer step
    def on_after_backward(self, model, grads, step):
        new_grads = all_reduce_grads(grads)
        return new_grads

    # This runs when the trainer reports training loss
    def on_train_loss_report(self, info):
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}")

    # This runs when the trainer reports validation loss
    def on_val_loss_report(self, info):
        try:
            iteration = info.get("iteration")
            val_loss = info.get("val_loss")
            if iteration is not None and val_loss is not None:
                self.val_losses.append((iteration, val_loss))
                print(f"[Valid] Rank {world.rank() if size > 1 else 0} Iteration {iteration}: Loss = {val_loss:.4f}")
                # Clear cache after validation to prevent memory buildup
                mx.clear_cache()
                self.validation_failures = 0  # Reset failure count on success
        except Exception as e:
            self.validation_failures += 1
            print(f"Rank {world.rank() if size > 1 else 0}: Validation error #{self.validation_failures}: {e}")
            mx.clear_cache()
            if self.validation_failures >= self.max_validation_failures:
                print(f"Rank {world.rank() if size > 1 else 0}: Too many validation failures, skipping future validations")
                # Disable future validations by setting a very high step interval
                return False

def plot_metrics(metrics_callback, save_path=None):
    if not metrics_callback.train_losses and not metrics_callback.val_losses:
        print("No metrics to plot.")
        return

    plt.figure(figsize=(8, 5))
    if metrics_callback.train_losses:
        train_its, train_vals = zip(*metrics_callback.train_losses)
        plt.plot(train_its, train_vals, '-o', label='Train Loss')
    if metrics_callback.val_losses:
        val_its, val_vals = zip(*metrics_callback.val_losses)
        plt.plot(val_its, val_vals, '-o', label='Validation Loss')

    plt.title("Training and Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # Print whether single or distributed
    if size == 1:
        print("Single process mode: no gradient averaging needed.")
    else:
        print(f"Distributed mode: Rank {world.rank()} - averaging gradients across {size} ranks.")

    parser = argparse.ArgumentParser(
        description="Run fine-tuning with MLX LM + LoRA.")
    parser.add_argument("--model", type=str, default="../Mistral-7B-Instruct-v0.3-4bit",
                        help="Path or name of the base model.")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data-sets")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=1000)  # Disable frequent validation in distributed mode
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory at the cost of speed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-parameters", type=dict,
                        default={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0})
    parser.add_argument("--lr-schedule", type=str, default=None)
    parser.add_argument("--wandb", type=str, default=None, help="Weights & Biases project name or None to disable.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type (e.g., adam, sgd, etc.).")
    parser.add_argument("--optimizer-config", type=dict, default={}, help="Additional configuration for the optimizer (e.g., learning rate schedules, weight decay, etc.).")

    args = parser.parse_args()

    # Override validation settings for distributed training to avoid timeouts
    if size > 1:
        print(f"Rank {world.rank()}: Adjusting settings for distributed training...")
        if world.rank() == 0:
          args.steps_per_eval = max(args.iters, 1000)  # Only validate at the end

        args.val_batches = 5  # Reduce validation batch size significantly
        print(f"Rank {world.rank()}: Validation will run every {args.steps_per_eval} steps with {args.val_batches} batches")

    start_time = time.time()

    # Load the model using the --model parameter
    print(f"Rank {world.rank() if size > 1 else 0}: Loading model...")
    model = load(args.model)
    
    # Synchronize after model loading
    if size > 1:
        print(f"Rank {world.rank()}: Model loaded")
    
    # Force garbage collection and memory cleanup
    gc.collect()
    mx.clear_cache()

    # Create the callback that does both:
    #  distributed gradient averaging
    #  metrics logging
    metrics_callback = MetricsCallback()

    try:
        # Run the LoRA fine-tuning
        # Orchestrates the training loop and calls callback hooks for training/validation loss, backward pass, etc.
        print(f"Rank {world.rank() if size > 1 else 0}: Starting training...")
        run(types.SimpleNamespace(**vars(args)),
            training_callback=metrics_callback)
            
    except Exception as e:
        print(f"Rank {world.rank() if size > 1 else 0}: Training failed with error: {e}")
        # Clean up GPU memory before exiting
        mx.clear_cache()
        raise

    # Plot the collected metrics (only on rank 0 to avoid conflicts)
    if world.rank() == 0 or size == 1:
        metrics_name = f"graphs/metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_metrics(metrics_callback, save_path=metrics_name)

    end_time = time.time()
    print(f"Rank {world.rank() if size > 1 else 0}: Script execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()