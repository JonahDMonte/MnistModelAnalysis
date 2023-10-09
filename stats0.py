import json
import pandas as pd
import matplotlib.pyplot as plt

with open("test_0.json", "r") as f:
    stats = json.load(f)


model_types = ['nothing', 'relu', 'softmax', 'both']
epochs_x = range(1, 11)

sorted_data = {i:None for i in model_types}

for x in model_types:  # loop for each type of test

    trainstats = [  # assemble the data into its own array
        [i[1] for i in stats[x][0]['trainstats']],
        [i[1] for i in stats[x][1]['trainstats']],
        [i[1] for i in stats[x][2]['trainstats']]
    ]
    testloss = [
        [i['testloss'] for i in stats[x][0]['teststats']],
        [i['testloss'] for i in stats[x][1]['teststats']],
        [i['testloss'] for i in stats[x][2]['teststats']]
    ]
    accuracy = [
        [i['accuracy'] for i in stats[x][0]['teststats']],
        [i['accuracy'] for i in stats[x][1]['teststats']],
        [i['accuracy'] for i in stats[x][2]['teststats']]
    ]

    trainstats_mean = []
    testloss_mean = []
    accuracy_mean = []
    argparse_mean = (stats[x][0]['argparse'] + stats[x][1]['argparse'] + stats[x][2]['argparse']) / 3
    dataloading_mean = (stats[x][0]['dataloading'] + stats[x][1]['dataloading'] + stats[x][2]['dataloading']) / 3
    modelling_mean = (stats[x][0]['modelling'] + stats[x][1]['modelling'] + stats[x][2]['modelling']) / 3
    total_mean = (stats[x][0]['total'] + stats[x][1]['total'] + stats[x][2]['total']) / 3

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

plt.title('Training Stats vs. Epoch for Each Model Type')
plt.xlabel('Epoch')
plt.ylabel('Training Stats') #TODO what are these called???

plt.legend()

plt.grid(True)

plt.savefig('images/test0/trainstats.png')


plt.figure()
for marker, x in enumerate(model_types):
    plt.scatter(epochs_x, sorted_data[x]['testloss'], label=x, marker=f"{marker+1}", alpha=1)

plt.title('Test Loss vs. Epoch for Each Model Type')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')

plt.legend()

plt.grid(True)

plt.savefig('images/test0/testloss.png')



plt.figure()
for marker, x in enumerate(model_types):
    plt.scatter(epochs_x, sorted_data[x]['accuracy'], label=x, marker=f"{marker+1}", alpha=1)

# Add a title and axis labels
plt.title('Accuracy vs. Epoch for Each Model Type')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Add a legend
plt.legend()

# Add grid lines (optional)
plt.grid(True)

# Save the plot as an image (e.g., PNG format)
plt.savefig('images/test0/accuracy.png')