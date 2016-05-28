#!/usr/bin/env python2
import timeit

import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        # matrix W with zeros of shape (n_in by n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name="W",
            borrow=True
        )

        # biases b as a vector of n_out
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name="b",
            borrow=True
        )

        # symbolic expression
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # params
        self.params = [self.W, self.b]

        # input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # pre-check
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                "y should have same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type)
            )

        if y.dtype.startswith("int"):
            return T.mean(T.neq(self.y_pred, y))

        else:
            raise NotImplementedError()


def sgd_mnist(dataset, learning_rate=0.13, n_epochs=1000, batch_size=600):
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # setup
    x = T.matrix("x")
    y = T.ivector("y")

    # instanciate logistic regression object
    classifier = LogisticRegression(
        input=x,  # training data as a matrix
        n_in=28,  # images are 28 x 28 pixels
        n_out=10  # classifying 10 digits
    )

    # cost function
    cost = classifier.negative_log_likelihood(y)

    # data
    train_set_x, train_set_y = dataset["train"]
    test_set_x, test_set_y = dataset["test"]
    validate_set_x, validate_set_y = dataset["validate"]

    # test model
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        given={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # validate model
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        given={
            x: validate_set_x[index * batch_size: (index + 1) * batch_size],
            y: validate_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # calculate gradient of W and b
    W_gradient = T.grad(cost=cost, wrt=classifier.W)
    b_gradient = T.grad(cost=cost, wrt=classifier.b)

    updates = [
        (classifier.W, classifier.W - learning_rate + W_gradient),
        (classifier.b, classifier.b - learning_rate + b_gradient)
    ]

    # compile train_model which returns the cost and updates the parameter at
    # the same time based on rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        given={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    minibatch_training(
        n_epochs=1000,
        batch_size=600,

        train_model=train_model,
        test_model=test_model,
        validate_model=validate_model,

        train_set_x=train_set_x,
        test_set_x=test_set_x,
        validate_set_x=validate_set_x
    )


def minibatch_training(**kwargs):
    n_epochs = kwargs["n_epochs"]
    batch_size = kwargs["batch_size"]

    train_model = kwargs["train_model"]
    test_model = kwargs["test_model"]
    validate_model = kwargs["validate_model"]

    train_set_x = kwargs["train_set_x"]
    test_set_x = kwargs["test_set_x"]
    valid_set_x = kwargs["valid_set_x"]

    # early-stopping parameters
    patience = 5000         # look as this many examples regardless
    patience_increase = 2   # wait when a new best is found

    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many minibatche before checking the network
    # on the validation set; in this case we check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            print minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [
                    validate_model(i)
                    for i in xrange(n_valid_batches)
                ]
                validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = validation_loss

                    # test it on the test set
                    test_losses = [
                        test_model(i) for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)

                    print(
                        ('     epoch %i, minibatch %i/%i, test error of' ' best model %f %%') %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    # with open('best_model.pkl', 'w') as f:
                    #     cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print "best validation score %f %%" % (best_validation_loss * 100.0)
    print "test performance %f %%" % (test_score * 100.0)
    print "epochs: %d" % (epoch)
    print "epochs/sec: %d" % (1.0 * epoch / (end_time - start_time))
