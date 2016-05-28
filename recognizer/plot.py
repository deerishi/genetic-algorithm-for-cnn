#!/usr/bin/env python
import os
import csv
import json
import glob

import numpy as np
import matplotlib.pylab as plt


def load_csv(path, header=True):
    data = {
        "gen_best": [],
        "gen_best_score": [],
        "all_time_best": [],
        "all_time_best_score": [],
        "failed": [],
        "seen": [],
        "diversity": [],
        "evaluated": []
    }

    with open(path, "r") as datafile:
        reader = csv.reader(datafile)

        # skip the headers
        if header:
            next(reader, None)

        # parse csv file
        for row in reader:
            r = []
            for i in row:
                if i.strip() == "None":
                    r.append(None)
                else:
                    r.append(float(i))

            data["gen_best_score"].append(r[0])
            data["gen_best"].append(r[1])
            data["all_time_best_score"].append(r[2])
            data["all_time_best"].append(r[3])
            data["failed"].append(r[4])
            data["seen"].append(r[5])
            data["diversity"].append(r[6])
            data["evaluated"].append(r[7])

        return data


def load_score_csv(path, header=True):
    data = {
        "chromosome": [],
        "score": [],
    }

    with open(path, "r") as datafile:
        reader = csv.reader(datafile)

        # skip the headers
        if header:
            next(reader, None)

        # parse csv file
        for row in reader:
            data["chromosome"].append(row[0])
            data["score"].append(float(row[1] if row[1] else 0.0))

        return data


def load_all_score_csv(path, header=True):
    pattern = os.path.join(path, "exp*/score_*.dat")
    score_files = [f for f in glob.iglob(pattern)]

    data = {
        "chromosome": [],
        "score": []
    }
    for f in score_files:
        score_file = load_score_csv(f)
        data["chromosome"].extend(score_file["chromosome"])
        data["score"].extend(score_file["score"])

    return data


def plot_aggr_runs(path):
    pattern = os.path.join(path, "exp*/execution_*.dat")
    execution_files = [f for f in glob.iglob(pattern)]
    # summarize just score vs generation
    aggr_data = {"scores": []}
    for f in execution_files:
        data = load_csv(f)
        aggr_data["scores"].append(data["all_time_best_score"])

    # plot summary
    plt.figure()
    for i in range(len(execution_files)):
        plt.plot(aggr_data["scores"][i])
    # plt.title("RW Tuner: Classification Accuracy vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    # plt.ylim((0.8, 1.0))
    # plt.xlim((0, 200))
    # plt.yticks(np.arange(-1.0, 1.1, 0.1))
    # plt.xticks([0, 50, 100, 150, 200])
    # plt.yticks([0.8, 0.9, 1.0])
    plt.show()
    # plt.savefig("summary.png", dpi=100)


def plot_ga(data, save=False, save_dest=None):
    plt.figure()

    # generation and all time best score plot
    plt.subplot(311)
    plt.plot(data["gen_best_score"], label="Generation Best")
    plt.plot(data["all_time_best_score"], label="All Time best")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.xticks(range(0, 20))
    plt.xlim([0, 9])
    plt.ylim([0, 0.01])
    # plt.legend(loc=0, prop={'size': 8})

    # # failed, seen, evaluated plot
    # plt.subplot(312)
    # plt.plot(data["failed"], label="Failed")
    # plt.plot(data["seen"], label="Seen")
    # plt.plot(data["evaluated"], label="Evaluated")
    # max_y = max(
    #     max(data["failed"]),
    #     max(data["seen"]),
    #     max(data["evaluated"])
    # )
    # max_y += 2
    # plt.ylim([0, max_y])
    # plt.xlabel("Generation")
    # plt.ylabel("Frequency")
    # plt.xticks(range(0, 20))
    # plt.xlim([0, 9])
    # plt.legend(loc=0, prop={'size': 8})
    #
    # # diversity plot
    # plt.subplot(313)
    # plt.plot(data["diversity"], label="Diversity")
    # plt.xlabel("Generation")
    # plt.ylabel("Diversity")
    # max_y = max(data["diversity"])
    # plt.ylim([0, max_y + 0.1])
    # plt.xticks(range(0, 20))
    # plt.xlim([0, 9])
    # plt.legend(loc=0, prop={'size': 8})

    # show plot or save as picture
    if save is False:
        plt.show()
    else:
        if save_dest is None:
            raise RuntimeError("save_dest not set!!")
        plt.savefig(save_dest, dpi=100)


def plot_ga_cnn(data, save=False, save_dest=None):
    plt.figure()

    # generation and all time best score plot
    # plt.subplot(211)
    # plt.title("GA CNN tuner on dataset D2")
    # plt.plot(data["dataset1"]["gen_best_score"], label="Generation Best")
    # plt.plot(data["dataset1"]["all_time_best_score"], label="All Time best")
    # plt.xlabel("Generation")
    # plt.ylabel("Score")
    # plt.xticks(range(0, 20))
    # plt.xlim([0, 9])
    # plt.ylim([0, 0.01])
    # plt.legend(loc=0)

    # plt.subplot(212)
    # plt.title("GA CNN tuner on dataset D3")
    plt.plot(data["gen_best_score"], label="Generation Best")
    plt.plot(data["all_time_best_score"], label="All Time best")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.xticks(range(0, 20))
    plt.xlim([0, 9])
    plt.ylim([0, 0.01])
    plt.legend(loc=0)

    # show plot or save as picture
    if save is False:
        plt.show()
    else:
        if save_dest is None:
            raise RuntimeError("save_dest not set!!")
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_dest, dpi=200)


def plot_random_walk(data):
    plt.figure(1)

    # generation and all time best score plot
    # plt.subplot(311)
    plt.plot(data["gen_best_score"], label="Generation Best")
    plt.plot(data["all_time_best_score"], label="All Time best")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend(loc=0)
    plt.show()


def plot_random_walk_runs(path):
    # plot aggr summary
    plot_aggr_runs(path)


def plot_ga_runs(path):
    pattern = os.path.join(path, "exp*/execution_*.dat")
    execution_files = [f for f in glob.iglob(pattern)]

    # # plot each run's summary
    # for f in execution_files:
    #     data = load_csv(f)
    #     print "plotting graphs for [{0}]".format(f)
    #     plot_img_path = os.path.basename(f).replace(".dat", ".png")
    #     plot_ga(data, True, plot_img_path)

    # summarize just score vs generation
    aggr_data = {"scores": []}
    for f in execution_files:
        data = load_csv(f)
        aggr_data["scores"].append(data["all_time_best_score"])

    # plot aggr summary
    plot_aggr_runs(path)


def plot_score_distribution(data, save=False, save_dest=None, title=None):
    # generation and all time best score plot
    plt.hist(data["score"], color="#3F5D7D", bins=100)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 1.1, 0.1))
    if save and save_dest and title:
        plt.title(title)
        plt.savefig(save_dest, dpi=100)

    plt.show()


def plot_results_all(path, save=False, save_dest=None):
    pattern = os.path.join(path, "*.dat")
    results = [f for f in glob.iglob(pattern)]
    data = []
    plt.figure(figsize=(8, 20))

    # load files
    for r in results:
        data_file = open(r, "r")
        d = json.loads(data_file.read())
        d["results_file"] = r
        data_file.close()
        epoch = range(0, d["nb_epoch"])
        data.append(d)

    # plot accuracy and loss
    for d in sorted(data, key=lambda k: k["results_file"]):
        filename = os.path.basename(d["results_file"])

        # training accuracy
        ax = plt.subplot(411)
        ax.plot(
            epoch,
            d['acc'],
            label="{0}".format(filename)
        )
        plt.ylabel("Training Accuracy")
        plt.xticks(range(0, max(epoch) + 20, 20))

        # validation accuracy
        plt.subplot(412)
        plt.plot(
            epoch,
            d['val_acc'],
            label="{0}".format(filename)
        )
        plt.ylabel("Validation Accuracy")
        plt.xticks(range(0, max(epoch) + 20, 20))

        # training loss
        plt.subplot(413)
        plt.plot(
            epoch,
            d['loss'],
            label="{0}".format(filename)
        )
        plt.ylabel("Training Loss")
        plt.xticks(range(0, max(epoch) + 20, 20))

        # validation loss
        plt.subplot(414)
        plt.plot(
            epoch,
            d['val_loss'],
            label="{0}".format(filename)
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.xticks(range(0, max(epoch) + 20, 20))

    # legend
    ax.legend(bbox_to_anchor=(0.85, 1.25), ncol=3)

    if save:
        if save_dest is None:
            raise RuntimeError("save_dest not set!!")
        plt.savefig(save_dest, dpi=100)

    else:
        plt.show()



def plot_results_file(fpath, save=False, save_dest=None):
    data_file = open(fpath, "r")
    data = json.loads(data_file.read())
    data_file.close()
    epoch = range(0, data["nb_epoch"])

    plt.figure(figsize=(8, 10))
    # plot 1
    plt.subplot(211)
    plt.plot(
        epoch,
        data['acc'],
        'b-',
        label="Training"
    )
    plt.plot(
        epoch,
        data['val_acc'],
        'b--',
        label="Validation"
    )
    plt.ylabel("Accuracy")
    plt.xlim([0, max(epoch) - 1])
    plt.xticks(range(0, max(epoch) + 20, 20))
    plt.legend(loc=0)

    # plot 2
    plt.subplot(212)
    plt.plot(
        epoch,
        data['loss'],
        'r-',
        label="Training"
    )
    plt.plot(
        epoch,
        data['val_loss'],
        'r--',
        label="Validation"
    )
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xlim([0, max(epoch) - 1])
    plt.xticks(range(0, max(epoch) + 20, 20))
    plt.legend(loc=0)

    if save:
        if save_dest is None:
            raise RuntimeError("save_dest not set!!")
        plt.savefig(save_dest, dpi=100)

    else:
        plt.show()


def plot_results(path):
    pattern = os.path.join(path, "*/results.dat")
    results = [f for f in glob.iglob(pattern)]
    aggr_data = {
        "acc": [],
        "loss": [],
        "val_acc": [],
        "val_loss": [],

        "labels": [],
        "save_paths": []
    }

    for rf in results:
        label = os.path.dirname(rf).replace("./", "")
        save_path = rf.replace("results.dat", label + ".png")

        # load results
        data_file = open(rf, "r")
        data = json.loads(data_file.read())
        data_file.close()

        # aggregate results
        aggr_data["nb_epoch"] = data["nb_epoch"]
        aggr_data["acc"].append(data["acc"])
        aggr_data["loss"].append(data["loss"])
        aggr_data["val_acc"].append(data["val_acc"])
        aggr_data["val_loss"].append(data["val_loss"])

        aggr_data["labels"].append(label)
        aggr_data["save_paths"].append(save_path)


    # plot 1
    plt.figure()
    for i in range(len(aggr_data["acc"])):
        epoch = aggr_data["nb_epoch"]
        y = aggr_data["acc"][i]
        plt.plot(range(epoch), y, label=aggr_data["labels"][i])

    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xlim([0, epoch - 1])
    plt.legend(loc=0)
    plt.show()

    # plot 2
    plt.figure()
    for i in range(len(aggr_data["acc"])):
        epoch = aggr_data["nb_epoch"]
        y = aggr_data["val_acc"][i]
        plt.plot(range(epoch), y, label=aggr_data["labels"][i])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xlim([0, epoch - 1])
    plt.legend(loc=0)
    plt.show()

    # plot 3
    plt.figure()
    for i in range(len(aggr_data["acc"])):
        epoch = aggr_data["nb_epoch"]
        y = aggr_data["loss"][i]
        plt.plot(range(epoch), y, label=aggr_data["labels"][i])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xlim([0, epoch - 1])
    plt.legend(loc=0)
    plt.show()

    # plot 4
    plt.figure()
    for i in range(len(aggr_data["acc"])):
        epoch = aggr_data["nb_epoch"]
        y = aggr_data["val_loss"][i]
        plt.plot(range(epoch), y, label=aggr_data["labels"][i])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xlim([0, epoch - 1])
    plt.legend(loc=0)
    plt.show()

