#!/usr/bin/bash
set -e

# Exec the specified command or fall back on bash
if [ $# -eq 0 ]; then
    cmd=( "bash" )
else
    cmd=( "$@" )
fi

# Handle special flags if we're root
if [ $(id -u) == 0 ] ; then
    echo "nope as root"
else
    if [[ ! -z "${EXTRA_PIP_PACKAGES}" ]]; then
        echo "Installing Extra Packages"
        python3 -m pip install --user ${EXTRA_PIP_PACKAGES}
    fi
    echo "Python is $(which python3)"
    echo "Executing the command: ${cmd[@]}"
    exec "${cmd[@]}"
fi

