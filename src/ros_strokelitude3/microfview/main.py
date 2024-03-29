"""microfview.main module

Provides Microfview class, which manages the frame capture
and delegates frames to all plugins.

"""
import bisect
import threading
import time

import logging
logger = logging.getLogger('microfview')

# helper function for frame_capture checks
def _has_method(obj, method):
    return hasattr(obj, method) and callable(getattr(obj, method))


class Microfview(threading.Thread):

    def __init__(self, frame_capture, flipRL=False, flipUD=False):
        """Microfview main class.

        Args:
          frame_capture (microfview capture): A capture class for aquiring frames.
            See microfview.camera or microfview.video.

        """
        threading.Thread.__init__(self)
        self.daemon = True
        self._lock = threading.Lock()

        # These three methods must be provided by a frame capture
        if not (_has_method(frame_capture, 'grab_next_frame_blocking') and
                _has_method(frame_capture, 'get_last_timestamp') and
                _has_method(frame_capture, 'get_last_framenumber')):
            raise TypeError("frame_capture_device does not provide required methods.")

        self.frame_capture = frame_capture
        self.frame_capture_noncritical_errors = frame_capture.noncritical_errors

        self.frame_count = 0
        self.frame_number_current = -1

        self._run = False
        self._callbacks = []
        self._plugins = []

        self._flip = flipRL or flipUD
        self._slice = (slice(None, None, -1 if flipUD else None),
                       slice(None, None, -1 if flipRL else None))

    def attach_callback(self, callback_func, every=1):
        """Attaches a callback function, which is called on every Nth frame.

        Args:
          callback_func:  takes two parameters (buffer and timestamp)
          every:  integer > 0

        returns:
          handle:  can be used to detach callback
        """
        if not hasattr(callback_func, '__call__'):
            raise TypeError("callback_func has to be callable")
        if not isinstance(every, int):
            raise TypeError("every has to be of type int")
        if every < 1:
            raise ValueError("every has to be bigger than 0")
        handle = (every, callback_func)
        if handle in self._callbacks:
            raise ValueError("callback_func, every combination exist.")
        bisect.insort(self._callbacks, handle)
        return handle

    def detach_callback(self, handle):
        """Detaches a callback."""
        if handle in self._callbacks:
            self._callbacks.remove(handle)
        else:
            raise ValueError("handle not attached.")

    def attach_plugin(self, plugin):
        """Attaches a plugin."""
        # check if plugin provides the required methods and attributes
        if not (_has_method(plugin, 'start') and
                _has_method(plugin, 'stop') and
                _has_method(plugin, 'push_frame') and
                hasattr(plugin, 'every')):
            raise TypeError("plugin does not have the required methods/attributes.")
        self._plugins.append(plugin)
        handle = self.attach_callback(plugin.push_frame, every=plugin.every)
        return (plugin, handle)

    def detach_plugin(self, handle):
        """Detaches a plugin."""
        plugin, cb_handle = handle
        self._plugins.remove(plugin)
        self.detach_callback(cb_handle)

    def run(self):
        """main loop. do not call directly."""
        # start all plugins
        for plugin in self._plugins:
            plugin.start()
        self._run = True
        try:
            while True:
                # grab frame
                try:
                    buf = self.frame_capture.grab_next_frame_blocking()
                except self.frame_capture_noncritical_errors:
                    logger.exception("error when retrieving frame")
                    continue

                frame_timestamp = self.frame_capture.get_last_timestamp()
                frame_number = self.frame_capture.get_last_framenumber()

                # warn if frames were skipped
                skip = frame_number - self.frame_number_current
                if skip != 1:
                    logger.warning('skipped %d frames' % skip)
                self.frame_number_current = frame_number

                self.frame_count += 1

                # flip
                if self._flip:
                    buf = buf[self._slice]

                now = time.time()
                # call all attached callbacks.
                for n, cb in self._callbacks:
                    if self.frame_number_current % n == 0:
                        cb(frame_timestamp, now, buf)

                with self._lock:
                    if not self._run:
                        logger.info("exiting main loop")
                        break
        finally:
            # stop the plugins
            for plugin in self._plugins:
                plugin.stop()
        logger.info("microfview mainloop exit")

    def stop(self):
        """stop the mainloop."""
        logger.info("calling microfview.stop()")
        with self._lock:
            self._run = False
