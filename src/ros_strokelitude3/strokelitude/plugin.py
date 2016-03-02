"""strokelitude.plugin module.

Provides the microfview.Strokelitude3Plugin

All methods with leading underscores are just used for drawing the cv2 HUD.

The algorithm for wing angle detection is defined in process_frame():
  - polar transform the defined wing boxes to two arrays wb_L, wb_R.
  - average those arrays along the radius axis wb_L --> L, wb_R --> R
  - correlate the one dimensional arrays L and R with -x*exp(-(x/10.)**2)
  - find index of maximum in resulting array (idx <--> angle)

The errors are estimated by counting the values in the L,R arrays, which
are greater than half the maximum. It's basically a poor man's FWHM 
measurement.

The flying state first is determined on a frame by frame basis, by
checking if the detected maximum is strongly localised. This is done by
comparing the peak-to-peak amplitude of a local patch around the maximum
with the ptp amplitude of the full trace. The resulting flying state
is then put into a length 10 fifo buffer. The flying state is determined
by the most occurring state in the buffer. (For reducing single false-
positives or false negatives)

"""
import collections
import time
import cv2
import numpy as np
import ros_strokelitude3.microfview as microfview

import logging
logger = microfview.getLogger()

import mmapspeedup
import multiprocessing as mp

# HACK CAMERA FRAME SHAPE
CAMERA_FRAME_SHAPE = (494, 659)
CAMERA_FRAME_DTYPE = np.uint8
#/HACK


class Strokelitude3Plugin(microfview.BlockingPlugin):

    def __init__(self, center_L=(270, 270),
                       center_R=(370, 270),
                       roi_radius=(200, 265),
                       roi_angles=(-10, 140),
                       roi_mshape=(520, 40),
                       mask_width=10.,
                       flying_queue_len=10):
        super(Strokelitude3Plugin, self).__init__(every=1) # , max_start_delay_sec=0.001)
        # basic config
        self.wing_center_L = tuple(center_L)
        self.wing_center_R = tuple(center_R)
        self.roi_radius = tuple(roi_radius)
        self.roi_angles = tuple(roi_angles)
        self.roi_mshape = tuple(roi_mshape)
        self.mask_width = float(mask_width)
        self.map_L, self.map_R = self.create_coordinates_maps(self.roi_angles,
                                                              self.roi_radius,
                                                              self.roi_mshape)
        self.correlation_edge_mask = self.create_correlation_edge_mask(self.mask_width)

        # flying detection
        self.is_flying_queue = collections.deque([False]*int(flying_queue_len),
                                                         int(flying_queue_len))

        # frequency detection
        self._init_frequency_measurement()

        # HUD config
        self.HUDPROCESS = Strokelitude3HUD(self.wing_center_L, self.wing_center_R, self.roi_angles, self.roi_radius)

        # time display
        self._debug_process_frame_now = time.time()

    def create_correlation_edge_mask(self, width=10.):
        """calculate the mask used to detect the wing edge."""
        N = 2*int(abs(width))
        _x = np.arange(-N, N+1, 1)
        return -_x*np.exp(-(_x/width)**2)

    def create_coordinates_maps(self, angles, radius, mshape):
        """calculate coordinate maps for transforming to polar coordinates."""
        H, W = mshape
        R0, R1 = radius
        phi0, phi1 = angles

        r, phi = np.meshgrid(np.arange(W), np.arange(H))
        phi = phi / float(H) * float(phi1 - phi0) + phi0
        r = r / float(W) * float(R1 - R0) + R0
        # left map
        x0, y0 = self.wing_center_L
        map_L_x = np.array(r * np.cos(np.radians(270 - phi)) + x0, dtype=np.float32)
        map_L_y = np.array(r * np.sin(np.radians(270 - phi)) + y0, dtype=np.float32)
        map_L = cv2.convertMaps(map_L_x, map_L_y, cv2.CV_16SC2, nninterpolation=False)
        # right map
        x0, y0 = self.wing_center_R
        map_R_x = np.array(r * np.cos(np.radians(270 + phi)) + x0, dtype=np.float32)
        map_R_y = np.array(r * np.sin(np.radians(270 + phi)) + y0, dtype=np.float32)
        map_R = cv2.convertMaps(map_R_x, map_R_y, cv2.CV_16SC2, nninterpolation=False)
        return map_L, map_R

    def wingbeatangle_from_idx(self, idx):
        """helper to translate roi index to wing beat angle."""
        phi0, phi1 = self.roi_angles
        angle_norm = float(idx) / self.roi_mshape[0]
        return np.radians(180 - ((phi1 - phi0) * angle_norm + phi0))

    def is_flying(self, L, R, i_L, i_R):
        """determine if the detected wing angle are valid."""
        N = 3*int(abs(self.mask_width))
        try:
            is_flying =  (L[i_L-N:i_L+N].ptp() > (L.ptp()/3.) and
                          R[i_R-N:i_R+N].ptp() > (R.ptp()/3.))
        except ValueError as e:
            self.logger.debug(e.message)
            is_flying = False
        # get rid off single false negatives
        self.is_flying_queue.append(is_flying)
        return sum(self.is_flying_queue) > len(self.is_flying_queue)/2

    def estimate_wingbeat_angle_error(self, L, R, i_L, i_R):
        """estimates the wingbeat angle error."""
        scale = 1. / self.roi_mshape[0] * (self.roi_angles[1] - self.roi_angles[0])
        err_L = min(np.radians(np.sum(L > (L[i_L]/2.)) * scale)/2., np.pi)
        err_R = min(np.radians(np.sum(R > (R[i_R]/2.)) * scale)/2., np.pi)
        return err_L, err_R

    def process_frame(self, frame, now, buf):
        """calculates the wingbeat amplitude on a frame by frame basis.

        This takes on average ~4msec.

        It calculates wing angles and estimates errors for those wing angles.
        Additionally it provides a boolean value which determines the flying
        status of the fly.
        """
        #===========================
        # WING EDGE ANGLE DETECTION
        #===========================
        wb_L = cv2.remap(buf, self.map_L[0], self.map_L[1], cv2.INTER_LINEAR)
        wb_R = cv2.remap(buf, self.map_R[0], self.map_R[1], cv2.INTER_LINEAR)
        ###<HACK>
        THRESH = 120
        wb_L = wb_L.astype(np.float32)
        wb_R = wb_R.astype(np.float32)
        wb_L[wb_L<THRESH] = float('nan')
        wb_R[wb_R<THRESH] = float('nan')
        ###</HACK>
        L = np.nanmean(wb_L, axis=1)
        R = np.nanmean(wb_R, axis=1)
        L_end, L_start = L[-10:].mean(), L[:10].mean()
        R_end, R_start = R[-10:].mean(), R[:10].mean()
        L -= (L_end - L_start) * np.linspace(0, 1.0, L.size, endpoint=False) + L_start
        R -= (R_end - R_start) * np.linspace(0, 1.0, R.size, endpoint=False) + R_start
        L = np.correlate(L, self.correlation_edge_mask, mode="same")
        R = np.correlate(R, self.correlation_edge_mask, mode="same")
        i_L = np.argmax(L)
        i_R = np.argmax(R)

        #=========================================================================
        # convert detected angles, determine if flying or not and estimate errors
        #=========================================================================
        angle_L = self.wingbeatangle_from_idx(i_L)
        angle_R = self.wingbeatangle_from_idx(i_R)
        err_L, err_R = self.estimate_wingbeat_angle_error(L, R, i_L, i_R)
        is_flying = self.is_flying(L, R, i_L, i_R)

        #==============================
        # WING BEAT FREQUENCY ESTIMATE
        #==============================
        freq, freq_err = self.get_frequency_estimate(wb_L[-self.freq_roi_Nrows:,:].sum(),
                                                 wb_R[-self.freq_roi_Nrows:,:].sum(), now)
        # publish
        self.publish_message(is_flying, angle_L, angle_R, err_L, err_R, freq, freq_err)

        if not is_flying:
            angle_L = 0.0
            angle_R = 0.0
        self.HUDPROCESS.SEND_ARRAY[:] = buf
        self.HUDPROCESS.wbaL.value = angle_L
        self.HUDPROCESS.wbaR.value = angle_R

        # time debug 
        self._debug_process_frame_end(now)


    def _init_frequency_measurement(self, win_len=40,
                                          ptp_thresh=0.2,
                                          std_thresh=300.,
                                          compensate_alias=200.,
                                          freq_roi_Nrows=20):
        self.freq_window_len = win_len
        self.freq_roi_Nrows = freq_roi_Nrows
        self.freq_ptp_thresh = ptp_thresh
        self.freq_std_thresh = std_thresh
        self.freq_compensate_alias = compensate_alias
        self.freq_sum_L = collections.deque([], self.freq_window_len)
        self.freq_sum_R = collections.deque([], self.freq_window_len)
        self.freq_L = collections.deque([float('nan')], self.freq_window_len)
        self.freq_R = collections.deque([float('nan')], self.freq_window_len)
        self.freq_lastT = 0.0
        self.freq_lastLT = 0.0
        self.freq_lastRT = 0.0

    def get_frequency_estimate(self, L_sum, R_sum, now):
        """gives a rough estimate for wingbeat frequency"""
        self.freq_sum_L.append(L_sum)
        self.freq_sum_R.append(R_sum)

        thresh_L = np.mean(self.freq_sum_L) + np.ptp(self.freq_sum_L) * self.freq_ptp_thresh
        thresh_R = np.mean(self.freq_sum_R) + np.ptp(self.freq_sum_R) * self.freq_ptp_thresh

        # Left frequency
        if (np.std(self.freq_sum_L) > self.freq_std_thresh and
            len(self.freq_sum_L) > 2 and
            self.freq_sum_L[-1] > thresh_L and
            self.freq_sum_L[-3] <= self.freq_sum_L[-2] >= self.freq_sum_L[-1]):
            dt = self.freq_lastT - self.freq_lastLT
            freq = 1./dt + self.freq_compensate_alias
            self.freq_lastLT = self.freq_lastT
            self.freq_L.append(freq)
        else:
            self.freq_L.append(self.freq_L[-1])

        # Right frequency
        if (np.std(self.freq_sum_R) > self.freq_std_thresh and
            len(self.freq_sum_R) > 2 and
            self.freq_sum_R[-1] > thresh_R and
            self.freq_sum_R[-3] <= self.freq_sum_R[-2] >= self.freq_sum_R[-1]):
            dt = self.freq_lastT - self.freq_lastRT
            freq = 1./dt + self.freq_compensate_alias
            self.freq_lastRT = self.freq_lastT
            self.freq_R.append(freq)
        else:
            self.freq_R.append(self.freq_R[-1])

        fL = self.freq_L[-1]
        fR = self.freq_R[-1]
        err = (np.std(self.freq_L) + np.std(self.freq_R))/2.
        self.freq_lastT = now

        return (fL + fR)/2., err


    def publish_message(self, is_flying, angle_L, angle_R, err_L, err_R, freq, freq_err):
        isf = "F" if is_flying else "X"
        self.logger.info("%s [L]%6.1f+-%5.1f deg [R]%6.1f+-%5.1f deg [F]%6.1f+-%5.1f", isf,
                                        np.degrees(angle_L), np.degrees(err_L),
                                        np.degrees(angle_R), np.degrees(err_R),
                                        freq, freq_err)

    def stop(self):
        """stops the plugin."""
        self.logger.info("Strokelitude calling stop")
        self.HUDPROCESS.stop()
        self.logger.info("StrokelitudeHUD exited.")
        super(Strokelitude3Plugin, self).stop()
        self.logger.info("PluginBaseClass stopped.")

    def _debug_process_frame_end(self, process_frame_now):
        """helper function for duration debugging."""
        if self.logger.level == logging.DEBUG:
            now = time.time()
            duration = (now - process_frame_now)
            fps = 1. / (process_frame_now - self._debug_process_frame_now)
            self.logger.debug("fps: %8.3f took: %f", fps, duration)
            self._debug_process_frame_now = process_frame_now




class Strokelitude3HUD(object):

    def __init__(self, center_L, center_R, roi_angles, roi_radius):
        # basic config
        self.wing_center_L = tuple(center_L)
        self.wing_center_R = tuple(center_R)
        self.roi_radius = tuple(roi_radius)
        self.roi_angles = tuple(roi_angles)
        self.color_hud = (0, 255, 0)
        self.color_detected = (0, 0, 255)
        self.window = "wingbeat_angles"
        self.framerate = 45.
        self._STOP = mp.Value('b', 0)
        self.wbaL = mp.Value('d', 0.0, lock=False)
        self.wbaR = mp.Value('d', 0.0, lock=False)
        m = mmapspeedup.FastInSlowOutSharedArray(CAMERA_FRAME_SHAPE, CAMERA_FRAME_DTYPE)
        self.SEND_ARRAY = m.get_input_array()
        self.p = mp.Process(target=self.mainloop, args=(self._STOP, self.wbaL, self.wbaR, m._fn))
        self.p.daemon = True
        self.p.start()

    def mainloop(self, stop, wbaL, wbaR, fn):
        self.interval_msec = max(int(1000./self.framerate), 1)
        self._show_hud_prepare_windows()
        buf = np.zeros(CAMERA_FRAME_SHAPE, dtype=CAMERA_FRAME_DTYPE)
        while True:
            buf[:] = np.memmap(fn, dtype=CAMERA_FRAME_DTYPE, mode='r', shape=CAMERA_FRAME_SHAPE)
            _wbaL = wbaL.value
            _wbaR = wbaR.value
            self._show_hud_fly(buf, _wbaL, _wbaR)
            if (stop.value) != 0:
                logger.info("StrokelitudeHUD exiting mainloop")
                break
            cv2.waitKey(self.interval_msec)

    def _show_hud_prepare_windows(self):
        """helper function to create the windows and move them in place."""
        space = 10
        cv2.namedWindow(self.window)

    def _show_hud_fly(self, frame, angle_L, angle_R):
        """show the fly image with HUD and detected wingbeat angles."""
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._draw_hud_point(out, self.wing_center_L)
        self._draw_hud_point(out, self.wing_center_R)
        self._draw_hud_wing_arc(out, self.wing_center_L, 'left')
        self._draw_hud_wing_arc(out, self.wing_center_R, 'right')
        self._draw_hud_wing_line(out, self.wing_center_L, angle_L, 'left')
        self._draw_hud_wing_line(out, self.wing_center_R, angle_R, 'right')
        cv2.imshow(self.window, out)

    def _draw_hud_point(self, img, point, radius=5):
        """helper function for drawing the HUD."""
        cv2.circle(img, point, radius, self.color_hud, thickness=3)

    def _draw_hud_wing_arc(self, img, center, pos):
        """helper function for drawing the HUD."""
        radius = self.roi_radius
        angle = -90
        start_angle = 360 - self.roi_angles[1] if pos == 'left' else self.roi_angles[0]
        stop_angle = 360 - self.roi_angles[0] if pos == 'left' else self.roi_angles[1]
        cv2.ellipse(img, center, (radius[0], radius[0]),
                    angle, start_angle, stop_angle, self.color_hud, 2)
        cv2.ellipse(img, center, (radius[1], radius[1]),
                    angle, start_angle, stop_angle, self.color_hud, 2)
        phi0_cos = np.cos((start_angle + angle) / 180. * np.pi)
        phi0_sin = np.sin((start_angle + angle)/ 180. * np.pi)
        phi1_cos = np.cos((stop_angle + angle) / 180. * np.pi)
        phi1_sin = np.sin((stop_angle + angle) / 180. * np.pi)
        x0 = int(radius[0]*phi0_cos + center[0])
        y0 = int(radius[0]*phi0_sin + center[1])
        x1 = int(radius[1]*phi0_cos + center[0])
        y1 = int(radius[1]*phi0_sin + center[1])
        cv2.line(img, (x0, y0), (x1, y1), self.color_hud, 2)
        x0 = int(radius[0]*phi1_cos + center[0])
        y0 = int(radius[0]*phi1_sin + center[1])
        x1 = int(radius[1]*phi1_cos + center[0])
        y1 = int(radius[1]*phi1_sin + center[1])
        cv2.line(img, (x0, y0), (x1, y1), self.color_hud, 2)

    def _draw_hud_wing_line(self, img, center, angle, side):
        """helper function for drawing the HUD."""
        if side == 'left':
            angle = np.pi/2. + angle
        else:  # side == right
            angle = np.pi/2. - angle
        cx, cy = center
        R = 2 * self.roi_radius[1]
        px, py = (R * np.cos(angle)) + cx, (R * np.sin(angle)) + cy
        cv2.line(img, (cx, cy), (int(px), int(py)), self.color_detected, 2)

    def stop(self):
        logger.info("StrokelitudeHUD calling stop")
        try:
            self._STOP.value = 1
            self.p.join(2)
        finally:
            logger.info("StrokelitudeHUD escalating to SIGTERM")
            self.p.terminate()
            self.p = None
            logger.info("StrokelitudeHUD terminated.")
            cv2.destroyAllWindows()
            logger.info("StrokelitudeHUD cv2.destroyAllWindows() returned.")





if __name__ == "__main__":

    TESTA = "/home/poehlmann/.gvfs/groups on storage.imp.ac.at/straw/data/FMFs-from-peter-higgins-to-improve-strokelitude/loose_head_tests/movie20130701_152239.fmf"
    TESTB = "/home/poehlmann/.gvfs/groups on storage.imp.ac.at/straw/data/FMFs-from-peter-higgins-to-improve-strokelitude/loose_head_tests/movie20130701_152513.fmf"
    TESTC = '/home/poehlmann/movie20120808_160423.fmf'
    TESTD = '../../movie-001-peter.fmf'

    fmf = microfview.FMFCapture(TESTD, force_framerate=100)
    fview = microfview.Microfview(fmf)

    testplugin = Strokelitude3Plugin()
    testplugin.logger.setLevel(logging.DEBUG)
    fview.attach_plugin(testplugin)

    try:
        fview.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        fview.stop()
        fview.join()
