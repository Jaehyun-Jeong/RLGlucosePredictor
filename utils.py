import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100) : i+1])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == "__main__":
    import torch
    writer = SummaryWriter(log_dir="./runs/")

    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    def train_model(iter):
        for epoch in range(iter):
            y1 = model(x)
            loss = criterion(y1, y)
            print(loss)
            writer.add_scalar("Loss/train", loss.item(), epoch) # Log scalars
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_model(10)
    writer.flush()
    writer.close()
