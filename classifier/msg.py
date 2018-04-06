#!/usr/bin/env python

import numpy as np
import time
import math
import logging

def ini(path):
    '''initialization

    Args:


    Returns:

    '''
    global start_time
    start_time = time.time()
    logging.basicConfig(filename=path,level=logging.DEBUG, format='%(asctime)s %(message)s')


def timemsg (msg):
    '''Outputs a message in the format[h, min, sec] and the message

    Args:
        msg: message

    Returns:

    '''
    global start_time
    elapsed_time = time.time() - start_time # seconds

    hour = math.floor( elapsed_time / 3600 )
    minute = math.floor( (elapsed_time%3600) / 60 )
    second = math.floor( elapsed_time%60 )
    output = str(hour) +  ":" +  str(minute) +  ":" + str(second) + " " + msg
    print (output)
    logging.info(output)

def debug(msg):
    print(msg)
    logging.debug(msg)
