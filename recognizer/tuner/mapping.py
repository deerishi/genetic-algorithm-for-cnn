#!/usr/bin/env python2


def binary2int(binary):
    return int("".join(map(str, binary)), 2)


def pop_genes(chromosome, pop_length=4):
    retval = binary2int(chromosome[0:pop_length])
    del chromosome[0:pop_length]
    return retval


def activation_mapping(chromosome):
    activation_types = [
        "model.add(Activation('sigmoid'))",
        "model.add(Activation('tanh'))",
        "model.add(Activation('softmax'))",
        "model.add(Activation('relu'))",
        "model.add(Activation('linear'))"
    ]

    index = pop_genes(chromosome, 2)
    return activation_types[index]


def dropout_mapping(chromosome):
    prob = pop_genes(chromosome, 6) / 100.0

    if prob >= 0.5:
        return ""
    else:
        return "model.add(Dropout({0}))".format(prob)


def dense_layer(chromosome, **kwargs):
    layer = []
    first_layer = kwargs.get("first_layer", False)
    last_layer = kwargs.get("last_layer", False)
    input_shape = kwargs.get("input_shape", None)
    n_outputs = kwargs.get("n_outputs", None)
    n_nodes = 0

    if len(chromosome) == 0:
        return ""

    # layer attributes
    n_nodes = pop_genes(chromosome, 8)

    # layer strings
    if first_layer:
        layer.append(
            "model.add(Dense({0}, input_shape={1}))".format(
                n_nodes,
                input_shape
            )
        )

    elif last_layer:
        layer.append("model.add(Dense({0}))".format(n_outputs))

    else:
        layer.append("model.add(Dense({0}))".format(n_nodes))

    # activation
    layer.append(activation_mapping(chromosome))

    # dropout
    layer.append(dropout_mapping(chromosome))

    return "\n".join(layer)


def _nb_filters(index):
    values = [
        16, 28, 40, 52, 64,
        76, 88, 100, 112, 134,
        146, 158, 170, 182
    ]

    if index >= len(values):
        return 10
    else:
        return values[index]


def _nb_filter_size(index):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if index >= len(values):
        return 1
    else:
        return values[index]


def convolution_layer(chromosome, **kwargs):
    layer = []
    first_layer = kwargs.get("first_layer", False)
    input_shape = kwargs.get("input_shape", None)
    nb_filters = 0
    nb_filter_size = 0

    if len(chromosome) == 0:
        return ""

    # num nb_filters
    value = pop_genes(chromosome)
    nb_filters = _nb_filters(value)

    # nb_filter size
    value = pop_genes(chromosome)
    nb_filter_size = _nb_filter_size(value)

    # layer string
    if first_layer:
        layer.append(
            "model.add(Convolution2D({0}, {1}, {1}, input_shape={2}))".format(
                nb_filters,
                nb_filter_size,
                input_shape
            )
        )
    else:
        layer.append(
            "model.add(Convolution2D({0}, {1}, {1}))".format(
                nb_filters,
                nb_filter_size,
            )
        )

    # activation
    value = pop_genes(chromosome)
    layer.append(activation_mapping(value))

    # dropout
    layer.append(dropout_mapping(chromosome))

    return "\n".join(layer)


def _pool_size_mapping(index):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if index >= len(values):
        return (1, 1)
    else:
        return (values[index], values[index])


def maxpool_layer(chromosome):
    layer = []

    if len(chromosome) == 0:
        return ""

    # pool size
    value = pop_genes(chromosome)
    pool_size = _pool_size_mapping(value)

    # layer
    layer.append("model.add(MaxPooling2D(pool_size={0}))".format(pool_size,))

    # dropout
    layer.append(dropout_mapping(chromosome))

    return "\n".join(layer)


def flatten_layer(chromosome, **kwargs):
    return "model.add(Flatten())"


def sgd_optimizer(chromosome):
    learning_rate = pop_genes(chromosome, 7)
    nesterov = bool(pop_genes(chromosome, 1))
    return "optimizer = keras.optimizers.SGD({0}, nesterov={1})".format(
        learning_rate / 100.0,
        nesterov
    )


def adagrad_optimizer(chromosome):
    return "optimizer = keras.optimizers.Adagrad()"


def adadelta_optimizer(chromosome):
    return "optimizer = keras.optimizers.Adadelta()"


def rmsprop_optimizer(chromosome):
    return "optimizer = keras.optimizers.RMSprop()"


def adam_optimizer(chromosome):
    return "optimizer = keras.optimizers.Adam()"


def optimizer_mapping(chromosome):
    code = []

    # optimizer
    optimizer_type = pop_genes(chromosome, 2)
    if optimizer_type == 0:
        code.append(sgd_optimizer(chromosome))

    elif optimizer_type == 1:
        code.append(adagrad_optimizer(chromosome))

    elif optimizer_type == 2:
        code.append(adadelta_optimizer(chromosome))

    elif optimizer_type == 3:
        code.append(rmsprop_optimizer(chromosome))

    elif optimizer_type == 4:
        code.append(adam_optimizer(chromosome))

    # error
    error_types = [
        "mse",
        "mae",
        "mape",
        "msle",
        "squared_hinge",
        "hinge",
        "binary_crossentropy",
        "categorical_crossentropy"
    ]
    error_func = error_types[pop_genes(chromosome, 3)]
    code.append(
        "model.compile(loss='{0}', optimizer=optimizer)".format(
            error_func
        )
    )

    return "\n".join(code)


def keras_code_full(architecture):
    code = """\
import random

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# code here
{0}

# fit the data
model.fit(
    X_train,
    Y_train,
    nb_epoch=nb_epoch,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    show_accuracy=True,
    verbose=0
)

# evaluate
model_score = model.evaluate(
    X_test,
    Y_test,
    show_accuracy=True,
    verbose=0
)
    """.format(architecture)

    return code


def keras_mapping(individual, **kwargs):
    keras_code = ["# setup", "model = Sequential()"]
    chromosome = list(individual["chromosome"])
    input_shape = kwargs["input_shape"]
    n_outputs = kwargs["n_outputs"]

    # seed
    # seed = pop_genes(chromosome, 7)  # 0 - 127
    # keras_code.append("random.seed({0})".format(seed))

    # nb_epoch
    nb_epoch = pop_genes(chromosome, 7)  # 0 - 127
    keras_code.append("nb_epoch = {0}".format(nb_epoch))

    # batch_size
    batch_size = pop_genes(chromosome, 8)  # 0 - 256
    keras_code.append("batch_size = {0}".format(batch_size))
    keras_code.append("")

    # obtain number of layers
    num_layers = pop_genes(chromosome, 3)

    # create layers
    first_layer = True
    for i in range(num_layers):
        keras_code.append("# layer {0}".format(i + 1))

        # break if end of chromosome
        if len(chromosome) == 0:
            break

        # layer type
        # layer_type = pop_genes(chromosome)
        last_layer = True if (i + 1) == num_layers else False

        keras_code.append(
            dense_layer(
                chromosome,
                first_layer=first_layer,
                last_layer=last_layer,
                input_shape=input_shape,
                n_outputs=n_outputs
            )
        )
        keras_code.append("")

        # # dense layer
        # if layer_type == 0 or layer_type > 2:
        #     keras_code.append(
        #         dense_layer(
        #             chromosome,
        #             first_layer=first_layer,
        #             last_layer=last_layer,
        #             input_shape=input_shape,
        #             n_outputs=n_outputs
        #         )
        #     )

        # # convolution layer
        # elif layer_type == 1:
        #     keras_code.append(
        #         convolution_layer(
        #             chromosome,
        #             first_layer=first_layer,
        #             input_shape=(1, 28, 28)
        #         )
        #     )

        # # maxpool layer
        # elif layer_type == 2:
        #     keras_code.append(maxpool_layer(chromosome))

        # # flatten layer
        # elif layer_type == 3:
        #     keras_code.append(flatten_layer(chromosome))

        # set first layer to False
        if first_layer is True:
            first_layer = False

    # optimizer
    keras_code.append("")
    keras_code.append(optimizer_mapping(chromosome))

    return keras_code_full("\n".join(keras_code))


def keras_mapping_mlp(individual):
    chromosome = list(individual["chromosome"])

    # number of layers
    nb_layers = pop_genes(chromosome, 3)

    # hidden_neurons
    hidden_neurons = [pop_genes(chromosome, 6) for i in range(nb_layers)]

    # activations
    activations = []
    activation_types = [
        "sigmoid",
        "tanh",
        "softmax",
        "relu",
        "linear"
    ]
    for i in range(nb_layers):
        index = pop_genes(chromosome, 2)
        activations.append(activation_types[index])

    # dropout
    dropouts = []
    for i in range(nb_layers):
        prob = pop_genes(chromosome, 6) / 100.0
        if prob >= 0.5:
            dropouts.append(None)
        else:
            dropouts.append(prob)

    # loss
    loss_types = [
        "mse",
        "mae",
        "mape",
        "msle",
        "squared_hinge",
        "hinge",
        "binary_crossentropy",
        "categorical_crossentropy"
    ]
    loss_func = loss_types[pop_genes(chromosome, 3)]

    # optimizer
    optimizer_types = [
        "sgd",
        "adagrad",
        "adadelta",
        "rmsprop",
        "adam"
    ]
    optimizer = optimizer_types[pop_genes(chromosome, 2)]

    # nb_epoch
    nb_epoch = pop_genes(chromosome, 7)  # 0 - 127

    # batch_size
    batch_size = pop_genes(chromosome, 8)  # 0 - 256

    return {
        "nb_layers": nb_layers,

        "hidden_neurons": hidden_neurons,
        "activations": activations,
        "dropouts": dropouts,

        "loss": loss_func,
        "optimizer": optimizer,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size
    }


def keras_mapping2(individual):
    chromosome = list(individual["chromosome"])

    # number of convolution layers
    nb_convo_layers = pop_genes(chromosome, 3)

    # number of convolutional filters
    nb_filters = [pop_genes(chromosome, 6) for i in range(nb_convo_layers)]

    # convolutional filter size
    nb_conv = [pop_genes(chromosome, 3) for i in range(nb_convo_layers)]

    # convolution activations
    convo_activations = []
    activation_types = [
        "sigmoid",
        "tanh",
        "softmax",
        "relu",
        "linear"
    ]
    for i in range(nb_convo_layers):
        index = pop_genes(chromosome, 2)
        convo_activations.append(activation_types[index])

    # maxpools and pool_sizes
    maxpools = []
    pool_sizes = []
    for i in range(nb_convo_layers):
        value = pop_genes(chromosome, 1)
        if value:
            maxpools.append(True)
            pool_sizes.append(pop_genes(chromosome, 3) + 1)
        else:
            maxpools.append(False)
            pool_sizes.append(None)

    # convolution dropout
    convo_dropouts = []
    for i in range(nb_convo_layers):
        prob = pop_genes(chromosome, 6) / 100.0
        if prob >= 0.5:
            convo_dropouts.append(None)
        else:
            convo_dropouts.append(prob)

    # number of dense layers
    nb_dense_layers = pop_genes(chromosome, 3)

    # hidden_neurons
    dense_hidden_neurons = [pop_genes(chromosome, 6) for i in range(nb_dense_layers)]

    # dense activations
    dense_activations = []
    for i in range(nb_dense_layers):
        index = pop_genes(chromosome, 2)
        dense_activations.append(activation_types[index])

    # dense dropouts
    dense_dropouts = []
    for i in range(nb_dense_layers):
        prob = pop_genes(chromosome, 6) / 100.0
        if prob >= 0.5:
            dense_dropouts.append(None)
        else:
            dense_dropouts.append(prob)

    # loss
    # loss_types = [
    #     "mse",
    #     "mae",
    #     "mape",
    #     "msle",
    #     "squared_hinge",
    #     "hinge",
    #     "binary_crossentropy",
    #     "categorical_crossentropy"
    # ]
    # loss_func = loss_types[pop_genes(chromosome, 3)]
    loss_func = "categorical_crossentropy"

    # optimizer
    optimizer_types = [
        "sgd",
        "adagrad",
        "adadelta",
        "rmsprop",
        "adam"
    ]
    optimizer = optimizer_types[pop_genes(chromosome, 2)]

    # nb_epoch
    nb_epoch = pop_genes(chromosome, 7)  # 0 - 127

    # batch_size
    batch_size = pop_genes(chromosome, 8)  # 0 - 256

    return {
        "nb_convo_layers": nb_convo_layers,
        "nb_filters": nb_filters,
        "nb_conv": nb_conv,

        "convo_activations": convo_activations,
        "maxpools": maxpools,
        "pool_sizes": pool_sizes,
        "convo_dropouts": convo_dropouts,

        "nb_dense_layers": nb_dense_layers,
        "dense_hidden_neurons": dense_hidden_neurons,
        "dense_activations": dense_activations,
        "dense_dropouts": dense_dropouts,

        "loss": loss_func,
        "optimizer": optimizer,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size
    }
