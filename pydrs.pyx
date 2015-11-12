# distutils: language = c++
# distutils: sources = DRS.cpp mxml.c strlcpy.c averager.cpp musbstd.c
# distutils: libraries = usb-1.0 util
from libcpp cimport bool
from libc.math cimport fabs
cimport numpy as np
cimport cython
import numpy as npy
cdef bool can_remove = 1
from datetime import datetime
from struct import pack


cdef extern from "DRS.h" nogil:
    cdef cppclass DRSBoard:
        int GetBoardSerialNumber()
        int GetFirmwareVersion()
        int Init()
        int SetFrequency(double, bool)
        int SetInputRange(double)
        int SetTranspMode(int)
        int EnableTrigger(int, int)
        int SetTriggerSource(int)
        int SetTriggerDelayPercent(int)
        int SetTriggerPolarity(int)
        int SetTriggerLevel(int)
        int GetBoardType()
        int StartDomino()
        int SetDominoMode(unsigned char)
        int SoftTrigger()
        int IsBusy()
        int TransferWaves()
        int TransferWaves(unsigned char*, int, int)
        int IsEventAvailable()
        int GetWave(unsigned int, unsigned char, float *)
        int GetWave(unsigned int, unsigned char, float *, bool, int,
                    int, bool, float, bool)
        int GetWave(unsigned char *, unsigned int, unsigned char, float *,
                    bool, int, int, bool, float, bool)
        double GetTemperature()
        int GetChannelCascading()
        int GetTimeCalibration(unsigned int, int, int, float *, bool)
        int GetTriggerCell(unsigned int)
        int GetStopCell(unsigned int)
        int GetWaveformBufferSize()
        unsigned char GetStopWSR(unsigned int)

cdef extern from "DRS.h" nogil:
    cdef cppclass DRS:
        DRS() except +
        int GetNumberOfBoards()
        DRSBoard *GetBoard(int)


cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t *)


cdef class PyDRS:
    cdef DRS *drs

    def __cinit__(self):
        self.drs = new DRS()

    #dealloc deletes board :/
    def free(self):
        del self.drs

    def get_number_of_boards(self):
        return self.drs.GetNumberOfBoards()

    def get_board(self, int i):
        board = PyBoard()
        board.from_board(self.drs.GetBoard(i), self.drs)
        return board


cdef class PyBoard:
    cdef DRSBoard *board
    cdef DRS *drs
    cdef public int sn, fw
    cdef float data[4][1024]
    cdef float center
    # cdef np.ndarray[np.float, ndim=2] arr
    cdef readonly bool normaltrigger
    # for filtering when removing spikes
    cdef object ba
    cdef int eventnum
    cdef unsigned char[18432] buf

    cdef void from_board(self, DRSBoard *board, DRS *drs):
        self.board = board
        self.drs = drs

    def __dealloc__(self):
        del self.board

    def get_board_serial_number(self):
        self.sn = self.board.GetBoardSerialNumber()
        return self.sn

    def get_firmware_version(self):
        self.fw = self.board.GetFirmwareVersion()
        return self.fw

    def get_board_type(self):
        return self.board.GetBoardType()

    def get_temperature(self):
        return self.board.GetTemperature()

    def init(self):
        self.board.Init()

    def set_frequency(self, freq, wait=True):
        return self.board.SetFrequency(freq, wait)

    def set_input_range(self, center):
        self.center = center
        return self.board.SetInputRange(center)

    def set_transp_mode(self, flag):
        return self.board.SetTranspMode(flag)

    def enable_trigger(self, flag1, flag2):
        return self.board.EnableTrigger(flag1, flag2)

    def set_trigger_mode(self, mode):
        self.normaltrigger = mode

    def set_trigger_source(self, source):
        return self.board.SetTriggerSource(source)

    def set_trigger_delay_percent(self, percent):
        return self.board.SetTriggerDelayPercent(percent)

    def set_trigger_polarity(self, pol):
        return self.board.SetTriggerPolarity(pol)

    def set_trigger_level(self, lvl):
        return self.board.SetTriggerLevel(lvl)

    def start_domino(self):
        return self.board.StartDomino()

    def soft_trigger(self):
        return self.board.SoftTrigger()

    def is_busy(self):
        return self.board.IsBusy()

    def transfer_waves(self):
        return self.board.TransferWaves()

    def is_event_available(self):
        return self.board.IsEventAvailable()

    def get_channel_cascading(self):
        return self.board.GetChannelCascading()

    def get_stop_cell(self, chip):
        return self.board.GetStopCell(chip)

    @cython.boundscheck(False)
    cdef get_waveform(self, unsigned int chip_index, unsigned char channel):
        assert channel < 4
        # trying new method
        self.board.TransferWaves(self.buf, 0, 8)
        cdef int trig_cell = self.board.GetStopCell(chip_index);
        self.board.GetWave(self.buf, chip_index, channel*2, self.data[channel], True,
                           trig_cell, 0, False, 0, True)

        # extrapolate first two samples
        self.data[channel][1] = 2*self.data[channel][2] - self.data[channel][3]
        self.data[channel][0] = 2*self.data[channel][1] - self.data[channel][2]

        if channel != 3:
            self.board.GetWave(self.buf, chip_index, 6, self.data[3], True,
                               trig_cell, 0, False, 0, True)

        self.data[3][1] = 2.*self.data[3][2] - self.data[3][3]
        self.data[3][0] = 2.*self.data[3][1] - self.data[3][2]

    @cython.boundscheck(False)
    cdef get_waveforms(self, unsigned int chip_index,
                        np.ndarray[long] channels):
        cdef int i, channel
        for channel in channels:
            assert channel < 4
        self.board.TransferWaves(self.buf, 0, 8)
        cdef int trig_cell = self.board.GetStopCell(chip_index)

        for channel in channels:
            self.board.GetWave(self.buf, chip_index, channel*2,
                               self.data[channel], True, trig_cell, 0, False,
                               0, True)

            self.data[channel][1] = 2*self.data[channel][2] - self.data[channel][3]
            self.data[channel][0] = 2*self.data[channel][1] - self.data[channel][2]

        if 3 not in channels:
            self.board.GetWave(self.buf, chip_index, 6,
                               self.data[3], True, trig_cell, 0, False,
                               0, True)

            self.data[3][1] = 2.*self.data[3][2] - self.data[3][3]
            self.data[3][0] = 2.*self.data[3][1] - self.data[3][2]


    cpdef get_corrected(self, int channel, bool remove=True):
        assert channel < 4
        cdef int i, j
        self.get_waveform(0, channel)
        for i in range(1024):
            for j in range(4):
                self.data[j][i] = (self.data[j][i] / 1000. - self.center + 0.5) * 65535
        if remove:
            remove_spikes_new(self.data, npy.arange(channel,channel+1))
        self.eventnum += 1
        cdef np.ndarray[float] parr = npy.zeros((1024,),
                                                         dtype=npy.float32)
        for i in range(1024):
            parr[i] = self.data[channel][i]
        return parr

    cpdef get_raw(self, int channel, bool remove=True):
        assert channel < 4
        cdef int i, j
        if self.get_trigger():
            self.get_waveform(0, channel)
            if remove:
                remove_spikes_new(self.data, npy.arange(channel,channel+1))
            self.eventnum += 1
            cdef np.ndarray[float] parr = npy.zeros((1024,),
                                                             dtype=npy.float32)
            for i in range(1024):
                parr[i] = self.data[channel][i]
            return parr

    cdef _get_multiple(self, np.ndarray[long] channels):
        cdef int i, j
        self.get_waveforms(0, channels)
        for i in range(1024):
            for j in range(4):
                self.data[j][i] = (self.data[j][i] / 1000. - self.center + 0.5) * 65535
        remove_spikes_new(self.data, channels)
        self.eventnum += 1


    def set_domino_mode(self, mode):
        self.board.SetDominoMode(mode)

    def write_header(self, bytes filename, object channels):
        _write_header(self.board, filename, channels)
        self.eventnum = 0

    def write_event(self, bytes filename, np.ndarray[long] channels):
        cdef np.ndarray[float, ndim=2] parr = self.get_multiple(channels)
        _write_data(self.eventnum, filename, channels, self.board, parr)

    cpdef get_multiple(self, np.ndarray[long] channels):
        self._get_multiple(channels)
        cdef np.ndarray[float, ndim=2] parr = npy.zeros((4, 1024),
                                                         dtype=npy.float32)
        cdef int i, j
        for i in range(4):
            for j in range(1024):
                parr[i][j] = self.data[i][j]
        return parr

    def get_trigger(self):
        cdef int t
        self.board.StartDomino()
        if not self.normaltrigger:
            self.board.SoftTrigger()
            while self.board.IsBusy():
                pass
            return True
        else:
            t = time(NULL)
            while not self.board.IsEventAvailable() or self.board.IsBusy():
                if (time(NULL) - t) > 5:
                    return False
            return True

    def get_triggered(self, np.ndarray[long] channels):
        if self.get_trigger():
            return self.get_multiple(channels)


cdef void diff(float inarr[1024], float outarr[1023]):
    cdef int i
    for i in range(1023):
        outarr[i] = inarr[i+1] - inarr[i]

cdef int where(float inarr[1023], int outarr[1023], float threshold):
    cdef int i
    cdef int c = 0
    for i in range(1023 - 5):
        if inarr[i] > threshold:
            outarr[c] = i
            c += 1
    return c

@cython.boundscheck(False)
cdef void remove_spikes_new(float inarr[4][1024],
                            np.ndarray[long] channels, int ref=3):
    #cdef np.ndarray[float] arr = npy.diff(inarr[ref][:-5])
    #cdef np.ndarray[long] indices = npy.where(arr > 600)[0]
    cdef float diffed[1023]
    diff(inarr[ref], diffed)
    cdef int thresholds[1023]
    cdef int limit = where(diffed, thresholds, 600)

    cdef int chnl, ind, index
    cdef float val
    for chnl in channels:
        for ind in range(limit):
            index = thresholds[ind]
            val = (inarr[chnl][index -2] + inarr[chnl][index + 4]) / 2.
            inarr[chnl][index] = val
            inarr[chnl][index+1] = val
            inarr[chnl][index+2] = val

            # inarr[chnl][ind:ind+3] = (inarr[chnl][ind -2] + inarr[chnl][ind + 4]) / 2.


cdef void _write_header(DRSBoard *board, bytes filename, object channels):
    cdef float tcal[2048]
    with open(filename, 'wb') as f:
        f.write('TIME')
        f.write('B#')
        f.write(pack('h', board.GetBoardSerialNumber()))

        for channel in channels:
            f.write('C00{}'.format(channel + 1))

            # get time calibration
            board.GetTimeCalibration(0, channel*2, 0, tcal, 0)
            timecal = [(tcal[i] + tcal[i+1])/2. for i in range(0, 2048, 2)]
            f.write(pack('f'*1024, *timecal))

@cython.boundscheck(False)
cdef void _write_data(int eventnum, bytes filename, np.ndarray[long] chnls,
                      DRSBoard *board, np.ndarray[float, ndim=2] voltages):
    cdef int sn = board.GetBoardSerialNumber()
    cdef int tc = board.GetTriggerCell(0)
    with open(filename, 'ab') as f:
        # event header
        f.write('EHDR')
        f.write(pack('i', eventnum))

        date = datetime.now()
        datearr = [date.year, date.month, date.day, date.hour, date.minute,
                   date.second, date.microsecond/1000, 0]
        f.write(pack('h'*8, *datearr))
        f.write('B#')
        f.write(pack('h', sn))
        f.write('T#')
        f.write(pack('h', tc))
        # channel header
        for channel in chnls:
            f.write('C00{}'.format(channel + 1))
            voltarr = voltages[channel].astype(npy.uint16)
            f.write(voltarr.tostring())
