import matplotlib.pyplot as plt

def plot_losses(train_loss, val_loss, title=None):
    fontsize = 16
    x_axis = list(range(len(train_loss)))
    plt.legend(fontsize=fontsize)
    plt.plot(train_loss, label="Training")
    plt.scatter(x_axis, train_loss, )
    plt.plot(val_loss, label='Validation')
    plt.scatter(x_axis, val_loss)
    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if title is not None:
        plt.savefig(f'./resources/plots/{title}')
        