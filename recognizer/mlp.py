#!/usr/bin/env python2
import json

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator


def analyze_mnist_data(**kwargs):
    x_train = kwargs["x_train"]
    y_train = kwargs["y_train"]
    x_test = kwargs["x_test"]
    y_test = kwargs["y_test"]
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    result_file = kwargs["result_file"]

    # MLP architecture
    model = Sequential()
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation("tanh"))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation("tanh"))
    model.add(Dropout(0.25))

    model.add(Dense(10))
    model.add(Activation("tanh"))

    # optimizer - Stochastic Gradient Descent
    model.compile(loss="mse", optimizer="rmsprop")

    # fit the data
    # model.load_weights("weights.hd5")
    fitlog = model.fit(
        x_train,
        y_train,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_split=0.1,
        show_accuracy=True,
        verbose=2
    )

    # import theano
    # print "layers:", model.layers
    # get_activations = theano.function(
    #     [model.layers[0].input],
    #     model.layers[4].get_output(train=False),
    #     allow_input_downcast=True
    # )
    # activations = get_activations(x_train)
    # print activations.shape
    # print activations[0]

    # model weights
    # for layer in model.layers[0:1]:
    #     g = layer.get_config()
    #     h = layer.get_weights()
    #     print g, h[0], len(h)

    # evaluate
    score = model.evaluate(
        x_test,
        y_test,
        show_accuracy=True,
        batch_size=batch_size
    )

    # model.save_weights("weights.hd5")
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # save fitlog
    results = {}
    results["nb_epoch"] = nb_epoch
    results["batch_size"] = batch_size
    results.update(fitlog.history)

    rf = open(result_file, "w")
    json.dump(results, rf)
    rf.close()

    return results

def mlp(**kwargs):
    X_train = kwargs["X_train"]
    Y_train = kwargs["Y_train"]
    X_test = kwargs["X_test"]
    Y_test = kwargs["Y_test"]

    input_shape = kwargs["input_shape"]
    nb_classes = kwargs["nb_classes"]

    nb_layers = kwargs["nb_layers"]
    hidden_neurons = kwargs["hidden_neurons"]
    activations = kwargs["activations"]
    dropouts = kwargs["dropouts"]

    loss = kwargs["loss"]
    optimizer = kwargs["optimizer"]
    nb_epoch = kwargs["nb_epoch"]
    batch_size = kwargs["batch_size"]

    model_file = kwargs["model_file"]
    results_file = kwargs["results_file"]
    weights_file = kwargs["weights_file"]

    # MLP architecture
    model = Sequential()
    for i in range(nb_layers):
        # dense layer
        if i == 0:
            model.add(Dense(hidden_neurons[i], input_shape=input_shape))
        elif (i + 1) == nb_layers:
            model.add(Dense(nb_classes))
        else:
            model.add(Dense(hidden_neurons[i]))

        # activation
        if activations[i]:
            model.add(Activation(activations[i]))

        # dropout
        if dropouts[i]:
            model.add(Dropout(dropouts[i]))

    # save model
    model_data = model.to_json()
    model_file = open(model_file, "w")
    model_file.write(json.dumps(model_data))
    model_file.close()

    # loss function and optimizer
    model.compile(loss=loss, optimizer=optimizer)

    # fit the data and save weights
    fitlog = model.fit(
        X_train,
        Y_train,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        show_accuracy=True,
        verbose=2,
    )
    model.save_weights(weights_file, overwrite=True)

    # save fitlog
    results = {}
    results = fitlog.history
    results["nb_epoch"] = nb_epoch
    results["batch_size"] = batch_size

    rf = open(results_file, "w")
    json.dump(results, rf)
    rf.close()

    # evaluate
    score = model.evaluate(
        X_test,
        Y_test,
        show_accuracy=True,
        batch_size=batch_size
    )
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
