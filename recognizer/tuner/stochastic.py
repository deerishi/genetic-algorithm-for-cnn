#!/usr/bin/env python2
import copy
import recognizer.tuner.ga as ga


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


def random_walk(**kwargs):
    max_iter = kwargs["max_iter"]
    chromo_size = kwargs["chromo_size"]
    nn_data = kwargs["nn_data"]
    record_file_path = kwargs["record_file_path"]
    score_file_path = kwargs["score_file_path"]
    error_file_path = kwargs["error_file_path"]

    # setup
    score_cache = {}
    error_cache = {}
    record_file = ga.init_recording(record_file_path)
    individual = {
        "score": -1.0,
        "chromosome": ga.init_chromosome(chromo_size)
    }
    all_time_best = {
        "chromosome": individual["chromosome"],
        "score": -1.0
    }

    # iterate generations
    i = 0
    while i < max_iter:
        print "iteration: {0}".format(i),
        print "\n"

        # evaluate
        retval = ga.evaluate({
            "individual": individual,
            "nn_data": nn_data,
            "score_cache": score_cache,
            "error_cache": error_cache
        })
        num_mutations = int(len(individual["chromosome"]) * 0.1)
        # find generation best and update all time best
        if individual["score"] >= all_time_best["score"]:
            all_time_best = copy.deepcopy(individual)

        # record both generation best and all time best into file
        failed = 1 if retval is False else 0
        seen = None
        diversity = None
        eval_count = None
        eval_stats = (failed, seen, diversity, eval_count)
        record_generation(
            record_file,
            individual,
            all_time_best,
            eval_stats
        )

        # mutate
        ga.point_mutation(1.0, individual, num_mutations)

        # update counter
        i += 1
        print "\n"

    # close record file
    record_file.close()
    record_score_cache(score_file_path, score_cache)
    record_error_cache(error_file_path, error_cache)
