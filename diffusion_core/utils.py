import torch

import matplotlib.pyplot as plt

class LiveLossPlot:
    def __init__(self):
        self.losses = []
        self.epochs = []
        plt.ion() # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Step (x100)") # or Epoch
        self.ax.set_ylabel("MSE Loss")
        self.line, = self.ax.plot([], [], 'r-')

    def update(self, loss_val):
        self.losses.append(loss_val)
        self.line.set_ydata(self.losses)
        self.line.set_xdata(range(len(self.losses)))
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def save_training_checkpoint(path,
                             epoch,
                             model,
                             optimizer,
                             scaler,
                             ema
                             ):
    
    """
    Saves a checkpoint containing all training states.
    """

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scale_state_dict": scaler.state_dict(),
        "ema_model_state_dict": ema.ema_model.state_dict(),
        "ema_step": ema.step
    }, path)
    
    print(f"Checkpoint saved at {path}")