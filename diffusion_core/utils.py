import torch
import math

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