"""microfview.video module

Provides FMFCapture class for FlyMovieFormat videos.
"""
import motmot.FlyMovieFormat.FlyMovieFormat as fmf
import time

import logging
logger = logging.getLogger('microfview')

# workaround for microfview compatibility, because ...
class _NOException(Exception):
    pass

class FMFCapture(fmf.FlyMovie):

    def __init__(self, filename, check_integrity=False, force_framerate=20):
        """class for interfacing fmf videos.

        Args:
          filename (str): fmf filename.
          check_integrity (bool, optional): check integrity of fmf file on
            load. Defaults to False.
          force_framerate (float, optional): forces a maximum framerate.
            Defaults to 20.

        """
        fmf.FlyMovie.__init__(self, filename, check_integrity)
        self._frame_timestamp = 0.0
        self._frame_number = -1
        self.noncritical_errors = (_NOException,)  # ... all errors are critical
        self._frame_delay = 1./float(force_framerate)

    def grab_next_frame_blocking(self):
        """returns next frame."""
        frame, timestamp = self.get_next_frame()
        self._frame_timestamp = timestamp
        self._frame_number += 1
        time.sleep(self._frame_delay)
        return frame

    def get_last_timestamp(self):
        """returns the timestamp of the last frame."""
        return self._frame_timestamp

    def get_last_framenumber(self):
        """returns the framenumber of the last frame."""
        return self._frame_number

