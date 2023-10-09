import json
import pandas as pd
import matplotlib.pyplot as plt

with open("test_1.json", "r") as f:
    stats = json.load(f)


model_types = ['nothing', 'relu', 'softmax', 'both']
epochs_x = range(1, 11)

sorted_data = {i:None for i in model_types}

for count, z in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
    for x in model_types:  # loop for each type of test

        trainstats = [  # assemble the data into its own array
            [i[1] for i in stats[x][3*count+0]['trainstats']],
            [i[1] for i in stats[x][3*count+1]['trainstats']],
            [i[1] for i in stats[x][3*count+2]['trainstats']]
        ]
        testloss = [
            [i['testloss'] for i in stats[x][3*count+0]['teststats']],
            [i['testloss'] for i in stats[x][3*count+1]['teststats']],
            [i['testloss'] for i in stats[x][3*count+2]['teststats']]
        ]
        accuracy = [
            [i['accuracy'] for i in stats[x][3*count+0]['teststats']],
            [i['accuracy'] for i in stats[x][3*count+1]['teststats']],
            [i['accuracy'] for i in stats[x][3*count+2]['teststats']]
        ]

        trainstats_mean = []
        testloss_mean = []
        accuracy_mean = []
        argparse_mean = (stats[x][3*count+0]['argparse'] + stats[x][3*count+1]['argparse'] + stats[x][3*count+2]['argparse']) / 3
        dataloading_mean = (stats[x][3*count+0]['dataloading'] + stats[x][3*count+1]['dataloading'] + stats[x][3*count+2]['dataloading']) / 3
        modelling_mean = (stats[x][3*count+0]['modelling'] + stats[x][3*count+1]['modelling'] + stats[x][3*count+2]['modelling']) / 3
        total_mean = (stats[x][3*count+0]['total'] + stats[x][3*count+1]['total'] + stats[x][3*count+2]['total']) / 3

        for i in range(0, 10):  # average over the 3 arrays to get the avg of the data
            trainstats_mean.append(
                (trainstats[0][i] + trainstats[1][i] + trainstats[2][i]) / 3
            )
            testloss_mean.append(
                (testloss[0][i] + testloss[1][i] + testloss[2][i]) / 3
            )
            accuracy_mean.append(
                (accuracy[0][i] + accuracy[1][i] + accuracy[2][i]) / 3
            )

        sorted_data[x] = {
            "trainstats":trainstats_mean,
            "testloss":testloss_mean,
            "accuracy":accuracy_mean,
            "argparse":argparse_mean,
            "dataloading":dataloading_mean,
            "modelling":modelling_mean,
            "total":total_mean
        }


    markers = ['1', '2', '3', '4']


    plt.figure()
    for marker, x in enumerate(model_types):
        plt.scatter(epochs_x, sorted_data[x]['trainstats'], label=x, marker=f"{marker+1}", alpha=1)

    plt.title(f'Training Stats vs. Epoch for Each Model Type (z = {z})')
    plt.xlabel('Epoch')
    plt.ylabel('Training Stats') #TODO what are these called???

    plt.legend()

    plt.grid(True)

    plt.savefig(f'images/test1/trainstats_{z}.png')
    plt.close()


    plt.figure()
    for marker, x in enumerate(model_types):
        plt.scatter(epochs_x, sorted_data[x]['testloss'], label=x, marker=f"{marker+1}", alpha=1)

    plt.title(f'Test Loss vs. Epoch for Each Model Type (z = {z})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')

    plt.legend()

    plt.grid(True)

    plt.savefig(f'images/test1/testloss_{z}.png')
    plt.close()



    plt.figure()
    for marker, x in enumerate(model_types):
        plt.scatter(epochs_x, sorted_data[x]['accuracy'], label=x, marker=f"{marker+1}", alpha=1)

    # Add a title and axis labels
    plt.title(f'Accuracy vs. Epoch for Each Model Type (z = {z})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Add a legend
    plt.legend()

    # Add grid lines (optional)
    plt.grid(True)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(f'images/test1/accuracy_{z}.png')
    plt.close()