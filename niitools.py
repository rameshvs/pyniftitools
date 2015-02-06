#!/usr/bin/env python

import sys
import inspect

import niitools

functions = {
        'majority_vote': niitools.labels.majority_vote,
        'jaccard': niitools.labels.jaccard_nii,
        'dice': niitools.labels.dice_nii,
        'mask': niitools.volumes.mask,
        'trim': niitools.volumes.trim_bounding_box,
        'neutralize_affine': niitools.headers.neutralize_affine,
        'copy_header': niitools.headers.copy_header,
        'convert_type': niitools.headers.convert_type,
        'upsample': niitools.resampling.upsample_axis,
        'masked_threshold': niitools.volumes.masked_threshold,
        'masked_threshold_count': niitools.volumes.masked_threshold_count,
        }

def print_help(func, name):
    signature = '{script} {name} {args}'.format(
            script=sys.argv[0],
            name=name,
            args=' '.join(inspect.getargspec(func).args))
    print(signature)
    print(func.__doc__)

if __name__ == '__main__':
    usage = 'USAGE: \n{script} <command> <arguments>\n    Commands: [ {commands} ]\n{script} -h <command>'.format(
            script=sys.argv[0],
            commands = ' | '.join(functions.keys()),
            )
    if len(sys.argv) == 1:
        print(usage)
        sys.exit(1)

    command = sys.argv[1]
    # Handle help
    if command == '-h':
        if len(sys.argv) == 2:
            print(usage)
        else:
            func_name = sys.argv[2]
            if func_name not in functions:
                print("Unknown command " + func_name)
                sys.exit(1)
            func = functions[func_name]
            print_help(func, func_name)
        sys.exit(0)

    # Handle normal usage
    arguments = sys.argv[2:]
    func = functions[command]
    argspec = inspect.getargspec(func)
    max = len(argspec.args)
    if argspec.defaults is None:
        min = max
    else:
        min = max - len(argspec.defaults)
    if argspec.varargs is not None: # varargs means we can accept an unbounded #
        max = len(arguments)+1

    if not (min <= len(arguments) <= max):
        print_help(func, command)
        sys.exit(1)

    func(*arguments)

