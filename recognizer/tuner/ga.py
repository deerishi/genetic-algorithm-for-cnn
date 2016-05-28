#!/usr/bin/env python2
import os
# import uuid
import copy
import random

import recognizer.cnn as cnn
import recognizer.tuner.mapping as mapping

# DEBUG = False
DEBUG = True
EVALUATION_COUNTER = 0


def evaluate(eval_data):
    individual = eval_data["individual"]
    nn_data = eval_data["nn_data"]
    score_cache = eval_data["score_cache"]
    error_cache = eval_data["error_cache"]
    X_train = nn_data["X_train"]
    Y_train = nn_data["Y_train"]
    X_test = nn_data["X_test"]
    Y_test = nn_data["Y_test"]
    input_shape = nn_data["input_shape"]
    n_outputs = nn_data["n_outputs"]

    # just to stop syntax checker complaining vars not being used
    X_train
    Y_train
    X_test
    Y_test

    # check cache
    chromo_str = "".join(map(str, individual["chromosome"]))
    if chromo_str in score_cache:
        print " in cache! score is: {1}".format(
            chromo_str,
            score_cache[chromo_str]
        )
        score = score_cache[chromo_str]
        individual["score"] = score
        return False

    # mapping
    code = mapping.keras_mapping(
        individual,
        input_shape=input_shape,
        n_outputs=n_outputs
    )
    # model = None
    model_score = None

    # execute
    try:
        exec(code)
        print("[score: {0}]".format(model_score[1]))

        individual["score"] = model_score[1]
        score_cache[chromo_str] = model_score[1]
        error_cache[chromo_str] = model_score[0]

    except Exception:
        print "[failed]"

        if DEBUG:
            import traceback
            print "- code -"
            print code
            print "-" * 20
            traceback.print_exc()

        score = -1

    return True


def evaluate_cnn(eval_data):
    individual = eval_data["individual"]
    nn_data = eval_data["nn_data"]
    score_cache = eval_data["score_cache"]
    error_cache = eval_data["error_cache"]
    X_train = nn_data["X_train"]
    Y_train = nn_data["Y_train"]
    X_test = nn_data["X_test"]
    Y_test = nn_data["Y_test"]
    input_shape = nn_data["input_shape"]
    n_outputs = nn_data["n_outputs"]

    # just to stop syntax checker complaining vars not being used
    X_train
    Y_train
    X_test
    Y_test

    # check cache
    chromo_str = "".join(map(str, individual["chromosome"]))
    if chromo_str in score_cache:
        print " in cache! score is: {1}".format(
            chromo_str,
            score_cache[chromo_str]
        )
        score = score_cache[chromo_str]
        individual["score"] = score
        return False

    # create folder
    global EVALUATION_COUNTER
    model_save_dir = nn_data["model_save_dir"]
    model_path = os.path.join(
        model_save_dir,
        "model_{0}".format(EVALUATION_COUNTER)
    )
    os.mkdir(model_path)
    EVALUATION_COUNTER += 1

    # mapping
    kwargs = mapping.keras_mapping2(individual)
    kwargs["X_train"] = X_train
    kwargs["Y_train"] = Y_train
    kwargs["X_test"] = X_test
    kwargs["Y_test"] = Y_test
    kwargs["input_shape"] = input_shape
    kwargs["nb_classes"] = n_outputs
    kwargs["model_file"] = os.path.join(model_path, "model.json")
    kwargs["weights_file"] = os.path.join(model_path, "weights.dat")
    kwargs["results_file"] = os.path.join(model_path, "results.dat")

    # execute
    try:
        results, model_score = cnn.cnn(**kwargs)
        print("[score: {0}]".format(model_score[1]))

        individual["score"] = model_score[1]
        score_cache[chromo_str] = model_score[1]
        error_cache[chromo_str] = model_score[0]

    except Exception:
        print "[failed]"
        if DEBUG:
            import traceback
            traceback.print_exc()
        score = -1

    return True


def evaluate_population(population, eval_data, eval_func=evaluate):
    nn_data = eval_data["nn_data"]
    score_cache = eval_data["score_cache"]
    error_cache = eval_data["error_cache"]

    failed = 0
    seen = 0
    eval_count = 0
    pop_str = []
    i = 0

    for individual in population:
        print "evaluating {0} ".format(i),
        chromo_str = "".join(map(str, individual["chromosome"]))
        if chromo_str in score_cache:
            seen += 1

        args = {
            "individual": individual,
            "nn_data": nn_data,
            "score_cache": score_cache,
            "error_cache": error_cache
        }
        evaluated = eval_func(args)

        if individual["score"] == -1:
            failed += 1

        if evaluated:
            eval_count += 1

        pop_str.append(chromo_str)
        i += 1

    unique = set(pop_str)
    diversity = (len(unique) / float(len(population)))

    return failed, seen, diversity, eval_count


def init_chromosome(chromo_size=10):
    chromosome = []

    for i in range(chromo_size):
        chromosome.append(random.randint(0, 1))

    return chromosome


def init_population(pop_size=100, chromo_size=10):
    population = []

    for i in range(pop_size):
        chromosome = init_chromosome(chromo_size)
        population.append({
            "chromosome": chromosome,
            "score": None
        })

    return population


def point_crossover(prob, x1, x2):
    # crossover if random number > cross over probability
    if random.random() > prob:
        return

    # pre-check
    if len(x1) != len(x2):
        raise RuntimeError("length of x1 != x2!")

    # perform point crossover at halfway
    chromo_1 = x1["chromosome"]
    chromo_2 = x2["chromosome"]

    tmp = list(chromo_2)
    chromo_2[0:len(chromo_2) / 2] = chromo_1[0:len(chromo_1) / 2]
    chromo_1[0:len(chromo_1) / 2] = tmp[0:len(chromo_1) / 2]

    x1["chromosome"] = chromo_1
    x2["chromosome"] = chromo_2


def point_mutation(prob, x, mutations=1, min=0, max=1):
    # mutate if random number > mutation probability
    if random.random() > prob:
        return

    for i in range(mutations):
        index = int(len(x["chromosome"]) * random.random())
        value = random.randint(min, max)
        x["chromosome"][index] = value


def tournament_selection(population, t_size=1):
    new_population = []

    # pre-check
    if population is None:
        raise RuntimeError("population is None!")

    if len(population) == 0:
        raise RuntimeError("population cannot be empty!")

    if t_size is None or t_size < 0 or t_size > len(population):
        raise RuntimeError("invalid t_size: {0}!".format(t_size))

    # create new population
    for i in range(len(population)):
        sample = random.sample(population, t_size)

        # find the best out of the random sample
        best = sample[0]
        best_score = 0.0
        best_score = sample[0]["score"]

        for contender in sample:
            # evaluate contender
            score = 0.0
            score = contender["score"]

            # check to see if contender's score is better
            if score > best_score:
                best = dict(contender)
                best_score = score

        # add winner to new population
        new_population.append(copy.deepcopy(best))

    return new_population


def best_individual(population):
    return sorted(population, key=lambda i: i["score"], reverse=True)[0]


def init_recording(record_file_path):
    record_file = open(record_file_path, "w")
    record_file.write(
        "# {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}\n".format(
            "gen_best_score",
            "gen_best",
            "all_time_best_score",
            "all_time_best",
            "failed",
            "seen",
            "diversity",
            "evaluated"
        )
    )
    return record_file


def print_stats(gen_best, all_time_best, nn_data):
    print ""
    print "generation best score: {0}".format(gen_best["score"])
    print "all time best score: {0}".format(all_time_best["score"])

    print "- code -"
    print "-" * 20
    # print mapping.keras_mapping(
    #     all_time_best,
    #     input_shape=nn_data["input_shape"],
    #     n_outputs=nn_data["n_outputs"]
    # )
    import pprint
    pprint.pprint(mapping.keras_mapping2(all_time_best))
    print "score: {0}".format(all_time_best["score"])
    print "-" * 20
    print ""


def record_generation(record_file, gen_best, all_time_best, eval_stats):
    failed, seen, diversity, eval_count = eval_stats
    record_file.write(
        "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}\n".format(
            gen_best["score"],
            "".join(map(str, gen_best["chromosome"])),
            all_time_best["score"],
            "".join(map(str, all_time_best["chromosome"])),
            failed,
            seen,
            diversity,
            eval_count
        )
    )
    record_file.flush()


def record_score_cache(cache_file_path, score_cache):
    cache_file = open(cache_file_path, "w")
    cache = sorted(
        score_cache.items(),
        key=lambda score_cache: score_cache[1],
        reverse=True
    )
    for item in cache:
        cache_file.write("{0},{1}\n".format(item[0], item[1]))
    cache_file.close()


def record_error_cache(cache_file_path, error_cache):
    cache_file = open(cache_file_path, "w")
    cache = sorted(
        error_cache.items(),
        key=lambda error_cache: error_cache[1],
        reverse=True
    )
    for item in cache:
        cache_file.write("{0},{1}\n".format(item[0], item[1]))
    cache_file.close()


def run(**kwargs):
    gen = 0
    max_gen = kwargs.get("max_gen", 5)
    pop_size = kwargs.get("pop_size", 10)
    chromo_size = kwargs.get("chromo_size", 2)
    t_size = kwargs.get("t_size", 2)
    c_prob = kwargs.get("c_prob", 0.5)
    m_prob = kwargs.get("m_prob", 0.8)
    population = init_population(pop_size, chromo_size)
    nn_data = kwargs["nn_data"]
    eval_func = kwargs.get("eval_func", evaluate_cnn)
    record_file_path = kwargs["record_file_path"]
    score_file_path = kwargs["score_file_path"]
    error_file_path = kwargs["error_file_path"]

    # setup
    score_cache = {}
    error_cache = {}
    record_file = init_recording(record_file_path)
    all_time_best = {"chromosome": population[0]["chromosome"], "score": -1.0}
    if os.path.exists(nn_data["model_save_dir"]) is False:
        os.mkdir(nn_data["model_save_dir"])

    # iterate generations
    while gen < max_gen:
        print "\ngeneration: {0}".format(gen)

        # evaluate
        eval_stats = evaluate_population(
            population,
            {
                "nn_data": nn_data,
                "score_cache": score_cache,
                "error_cache": error_cache
            },
            eval_func
        )

        # find generation best and update all time best
        gen_best = best_individual(population)
        if gen_best["score"] >= all_time_best["score"]:
            all_time_best = copy.deepcopy(gen_best)

        # record both generation best and all time best into file
        print_stats(gen_best, all_time_best, nn_data)
        record_generation(
            record_file,
            gen_best,
            all_time_best,
            eval_stats
        )

        # selection
        population = tournament_selection(population, t_size)

        # crossover and mutation
        for i in range(0, len(population), 2):
            x = population[i]
            y = population[i + 1]

            point_crossover(c_prob, x, y)
            num_mutations = int(len(x["chromosome"]) * 0.1)
            point_mutation(m_prob, x, num_mutations)
            point_mutation(m_prob, y, num_mutations)

        # update counter
        gen += 1

    # close record file
    record_file.close()
    record_score_cache(score_file_path, score_cache)
    record_error_cache(error_file_path, error_cache)
