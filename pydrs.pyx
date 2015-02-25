# distutils: language = c++
# distutils: sources = DRS.cpp mxml.c strlcpy.c averager.cpp musbstd.c
# distutils: libraries = usb-1.0 util
from libcpp cimport bool
from libc.math cimport fabs
cimport numpy as np
cdef bool can_remove = 1
from datetime import datetime
from struct import pack
try:
    import numpy as np
    from scipy.signal import iirfilter, filtfilt
except:
    can_remove = 0
    print 'scipy.signal not found, cannot remove spikes'


cdef extern from "DRS.h":
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

cdef extern from "DRS.h":
    cdef cppclass DRS:
        DRS() except +
        int GetNumberOfBoards()
        DRSBoard *GetBoard(int)


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
    cdef object arr
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
        del self.drs

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
        self.arr = np.ndarray((1024,), dtype='float')
        self.ba = iirfilter(1, 0.4, btype='highpass')

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

    def get_waveform(self, unsigned int chip_index, unsigned char channel):
        assert channel < 4
        # trying new method
        self.board.TransferWaves(self.buf, 0, 8)
        cdef int trig_cell = self.board.GetStopCell(chip_index);
        self.board.GetWave(self.buf, chip_index, channel*2, self.data[channel], True,
                           trig_cell, 0, False, 0, True)

        cdef int i
        for i in range(1024):
            self.arr[i] = self.data[channel][i]

        # extrapolate first two samples
        self.arr[1] = 2*self.arr[2] - self.arr[3]
        self.arr[0] = 2*self.arr[1] - self.arr[2]
        return self.arr

    def get_corrected(self, int channel):
        assert channel < 4
        cdef int i
        self.get_waveform(0, channel)
        if can_remove:
            remove_spikes(self.arr, self.ba, 3.9)
        self.arr = (self.arr / 1000. - self.center + 0.5) * 65535
        self.eventnum += 1
        return self.arr

    def set_domino_mode(self, mode):
        self.board.SetDominoMode(mode)

    def _return_orig(self, int channel):
        cdef int i
        for i in range(1024):
            self.arr[i] = self.data[channel][i]
        return self.arr

    def write_header(self, object filename, int channel):
        _write_header(self.board, filename, channel)
        self.eventnum = 0

    def write_event(self, object filename, int channel):
        self.get_corrected(channel)
        _write_data(self.eventnum, filename, channel, self.board, self.arr)


cdef void remove_spikes(object inarr, object ba, float threshold):
    ar = filtfilt(ba[0], ba[1], inarr)
    indices = np.where(ar > threshold)[0]
    singles = []
    doubles = []

    for i, spike in enumerate(zip(ar[indices], ar[indices+1], ar[indices+2])):
        # single
        if spike[1] < 1:
            if not ar[indices[i]-1] > threshold:
                singles.append(i)
        # double
        else:
            doubles.append(i)

    # remove spikes
    for index in indices[singles]:
        inarr[index-1:index+2] = (inarr[index - 2] + inarr[index + 3]) / 2.
    for index in indices[doubles]:
        inarr[index-1:index+3] = (inarr[index - 2] + inarr[index + 4]) / 2.
    # print '{} spikes removed'.format(len(singles+doubles))
    # print indices[singles+doubles]


cdef void _write_header(DRSBoard *board, object filename, int channel):
    cdef float tcal[2048]
    with open(filename, 'wb') as f:
        f.write('TIME')
        f.write('B#')
        f.write(pack('h', board.GetBoardSerialNumber()))
        # only one channel for now
        f.write('C00{}'.format(channel + 1))

        # get time calibration
        board.GetTimeCalibration(0, channel*2, 0, tcal, 0)
        timecal = [(tcal[i] + tcal[i+1])/2. for i in range(0, 2048, 2)]
        f.write(pack('f'*1024, *timecal))


cdef void _write_data(int eventnum, object filename, int chnl, DRSBoard *board,
                      object voltages):
    with open(filename, 'ab') as f:
        # event header
        f.write('EHDR')
        f.write(pack('i', eventnum))

        date = datetime.now()
        datearr = [date.year, date.month, date.day, date.hour, date.minute,
                   date.second, date.microsecond/1000, 0]
        f.write(pack('h'*8, *datearr))
        f.write('B#')
        f.write(pack('h', board.GetBoardSerialNumber()))
        f.write('T#')
        f.write(pack('h', board.GetTriggerCell(0)))
        # channel header
        f.write('C00{}'.format(chnl + 1))
        voltarr = voltages.astype(np.uint16)
        f.write(voltarr.tostring())




