#!/usr/bin/env bash
# Original author: deajan <https://stackoverflow.com/users/2635443/deajan>
# Code below was obtained from https://stackoverflow.com/a/39189370

function f_log {
    echo "$1"
}

# Take a list of commands to run, runs them sequentially with numberOfProcesses commands simultaneously runs
# Returns the number of non zero exit codes from commands
function f_ParallelExec {
    local numberOfProcesses="${1}" # Number of simultaneous commands to run
    local commandsArg="${2}" # Semi-colon separated list of commands

    local pid
    local runningPids=0
    local counter=0
    local commandsArray
    local pidsArray
    local newPidsArray
    local retval
    local retvalAll=0
    local pidState
    local commandsArrayPid

    IFS=';' read -r -a commandsArray <<< "$commandsArg"

    f_log "Runnning ${#commandsArray[@]} commands in $numberOfProcesses simultaneous processes."

    while [ $counter -lt "${#commandsArray[@]}" ] || [ ${#pidsArray[@]} -gt 0 ]; do

        while [ $counter -lt "${#commandsArray[@]}" ] && [ ${#pidsArray[@]} -lt $numberOfProcesses ]; do
            f_log "Running command [${commandsArray[$counter]}]."
            eval "${commandsArray[$counter]}" &
            pid=$!
            pidsArray+=($pid)
            commandsArrayPid[$pid]="${commandsArray[$counter]}"
            counter=$((counter+1))
        done


        newPidsArray=()
        for pid in "${pidsArray[@]}"; do
            # Handle uninterruptible sleep state or zombies by ommiting them from running process array (How to kill that is already dead ? :)
            if kill -0 $pid > /dev/null 2>&1; then
                pidState=$(ps -p$pid -o state= 2 > /dev/null)
                if [ "$pidState" != "D" ] && [ "$pidState" != "Z" ]; then
                    newPidsArray+=($pid)
                fi
            else
                # pid is dead, get it's exit code from wait command
                wait $pid
                retval=$?
                if [ $retval -ne 0 ]; then
                    f_log "Command [${commandsArrayPid[$pid]}] failed with exit code [$retval]."
                    retvalAll=$((retvalAll+1))
                fi
            fi
        done
        pidsArray=("${newPidsArray[@]}")

        # Add a trivial sleep time so bash won't eat all CPU
        sleep .05
    done

    return $retvalAll
}
