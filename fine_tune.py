import argparse
import time
import types
import matplotlib.pyplot as plt # <-- this is for producing a graph that is helpful for visualizing our training accuracy
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run

# This is how we define the "world" of our distributed training. MLX needs to know that we're using MPI, and it can figure out the rest
world = mx.distributed.init()
size = world.size()

def all_reduce_grads(grads):
    # I added this check so that we can easily run this script as a single process. Size is always 1 if we only have one slot, or aren't using MPI
    if size == 1:
        return grads
    # Sum across all ranks, then divide
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)

# We need this to extend the TrainingCallback class in order to add our custom gradient averaging function
class MetricsCallback(TrainingCallback):

    def __init__(self):
        # Initialize lists for loss tracking
        self.train_losses = []
        self.val_losses = []

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
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")

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
        print(f"Distributed mode: Rank {
              world.rank()} - averaging gradients across {size} ranks.")

    parser = argparse.ArgumentParser(
        description="Run fine-tuning with MLX LM + LoRA.")
    parser.add_argument("--model", type=str, default="../Mistral-7B-Instruct-v0.3-4bit",
                        help="Path or name of the base model.")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data1/")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-parameters", type=dict,
                        default={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0})
    parser.add_argument("--lr-schedule", type=str, default=None)

    args = parser.parse_args()

    start_time = time.time()

    # Load the model using the --model parameter
    model = load(args.model)

    # Create the callback that does both:
    #  distributed gradient averaging
    #  metrics logging
    metrics_callback = MetricsCallback()

    # Run the LoRA fine-tuning
    # Orchestrates the training loop and calls callback hooks for training/validation loss, backward pass, etc.
    run(types.SimpleNamespace(**vars(args)),
        training_callback=metrics_callback)

    # Plot the collected metrics
    metrics_name = f"graphs/metrics_{
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_metrics(metrics_callback, save_path=metrics_name)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()