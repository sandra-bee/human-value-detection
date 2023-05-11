import matplotlib.pyplot as plt

# Make a graph of the loss per step or epoch:
def make_loss_graph(loss_data, split, detailed):

    # Plot the data points:
    if isinstance(loss_data[0], list):
        # Training + validation loss:
        plt.plot(loss_data[0], label = "Training")
        plt.plot(loss_data[1], label = "Validation")
    else:
        # Only Training or only Test loss:
        plt.plot(loss_data)

    # Determine what should be on the x-axis:
    if detailed:
        x_axis = "Step"
    else:
        x_axis = "Epoch"

    # Plot the axes:
    plt.xlabel(x_axis, fontsize = 14)
    plt.ylabel('Binary cross-entropy loss', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', alpha=0.2)

    # Make the legend, if useful:
    if isinstance(loss_data[0], list):
        # Training + validation loss:
        plt.legend(loc="upper right")

    # Plot the title:
    plt.title(label= split + " loss per " + x_axis.lower(),
              fontsize=18)

    # Save the plot:
    plt.savefig('plots/' + split + ".png")

    # Show the plot:
    plt.show()
