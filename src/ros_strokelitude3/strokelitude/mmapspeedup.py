#!/usr/bin/env python

import numpy as np
import multiprocessing as mp
import time
import tempfile
import os

class FastInSlowOutSharedArray(object):

    def __init__(self, shape, dtype):
        self._fn = os.path.join(tempfile.mkdtemp(), 'sharedarr.dat')
        self._shape = shape
        self._dtype = dtype

    def get_input_array(self):
        arr = np.memmap(self._fn, dtype=self._dtype, shape=self._shape, mode='w+')
        return arr

    def _loop(self, fn, datafunc, loop_delay):
        myarr = np.zeros(self._shape, dtype=self._dtype)
        while True:
            myarr[:] = np.memmap(fn, dtype=self._dtype, mode='r', shape=self._shape)
            datafunc(myarr)
            time.sleep(loop_delay)

    def get_output_process(self, datafunc=lambda arr:0, loop_delay=0.0):
        p = mp.Process(target=self._loop, args=(self._fn, datafunc, loop_delay))
        p.daemon = True
        return p


if __name__ == "__main__":

    m = FastInSlowOutSharedArray((480, 640), np.int64)
    arrW = m.get_input_array()
    def datafunc(arr):
        print "sideloop: %f" % np.mean(arr)
    p = m.get_output_process(datafunc=datafunc, loop_delay=0.1)

    def mainloop():
        i = 0
        while True:
            i += 1
            ab = np.ones((480,640), dtype=np.uint8) * i
            s = time.time()
            arrW[:] = ab
            e = time.time()
            print "Mainloop: %d, took %f" % (i, e-s)

    p.start()
    mainloop()
    p.join()

