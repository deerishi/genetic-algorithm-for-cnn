#!/bin/bash
COMMAND='python'
TESTS_DIR='tests'
TEST_FILE_PATTERN='*_test.py'
LOG_TESTS=0
LOG_PATH=$TESTS_DIR/log
TMP_OUTPUT=/tmp/output.tmp

GREEN="\e\033[32m"
RED="\e\033[31m"
ENDTC="\e\033[0m"


strip_term_colors()
{
    SED_ARGS="s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    SED_FLAGS=-r

    cat $LOG_FILE | sed $SED_FLAGS $SED_ARGS > $LOG_FILE.processed
    mv $LOG_FILE.processed $LOG_FILE
}

print_passed()
{
    echo -e "${GREEN}PASSED!${ENDTC}"
}

print_failed()
{
    echo -e "${RED}FAILED!${ENDTC}"
    echo "--------------------------------------------------"
    cat $TMP_OUTPUT
    echo "--------------------------------------------------"
}

check_exit_status()
{
    FAILED=$?
}

analyze_test_results()
{
    check_exit_status

    if [ $FAILED != 0 ]
    then
        print_failed
    else
        print_passed
    fi
}

log_test()
{
    mkdir -p $LOG_PATH
    LOG_FILE=$LOG_PATH/$(basename ${TEST}.log)
    cp $TMP_OUTPUT $LOG_FILE
}

run_test()
{
    # run test
    printf "TEST [$TEST] "
    $COMMAND $TEST > $TMP_OUTPUT 2>&1
    analyze_test_results

    # log test
    if [ $LOG_TESTS -eq 1 ]
    then
        log_test
        strip_term_colors
    fi
}

main()
{
    TESTS=$(find ${TESTS_DIR} -type f \
        -name ${TEST_FILE_PATTERN} \
        ! -path "${LOG_PATH}/*")
    for TEST in $TESTS
    do
        run_test
    done
}
main
