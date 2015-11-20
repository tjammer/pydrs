from pydrs import PyDRS
from lxml import etree
from time import time
from datetime import datetime
from sys import stdout
from struct import pack
import numpy as np


class DRSConfig(object):
    """docstring for DRSConfig"""
    def __init__(self, board, configfile=None):
        super(DRSConfig, self).__init__()
        if not configfile:
            configfile = 'drsosc.cfg'
        cfg = etree.parse(configfile).getroot().find(
            'Board-{}'.format(board.sn))
        self.freq = float(cfg.find('SamplingSpeed').text)
        self.rnge = float(cfg.find('Range').text)
        # 1: normal, 0: auto
        self.tr_mode = int(cfg.find('TrgMode').text)
        self.tr_source = int(cfg.find('TrgSource').text)
        self.tr_delay = int(cfg.find('TrgDelay').text)
        self.tr_polarity = int(cfg.find('TrgNegative').text)
        if not self.tr_source == 4:
            self.tr_level = float(cfg.find(
                'TrgLevel{}'.format(self.tr_source+1)).text)
        else:
            self.tr_level = None

    def apply(self, board):
        board.init()
        board.set_frequency(self.freq)
        board.set_input_range(self.rnge)
        board.set_transp_mode(1)
        board.enable_trigger(1, 0)
        board.set_trigger_mode(self.tr_mode)
        board.set_trigger_source((1 << self.tr_source))
        board.set_trigger_delay_percent(self.tr_delay)
        board.set_trigger_polarity(self.tr_polarity)
        if self.tr_level:
            print self.tr_level
            board.set_trigger_level(self.tr_level)
        """# flush first event
        board.start_domino()
        board.transfer_waves()"""


class DrsBoard(object):
    """docstring for DrsBoard"""
    def __init__(self):
        super(DrsBoard, self).__init__()
        self.board = init_board()
        self.config = DRSConfig(self.board)
        self.config.apply(self.board)

    def trigger(self):
        return trigger(self.board)

    def measure(self, filename, nevents, channels=(0,), abort=True, bar=True):
        n = 0
        self.board.write_header(filename, channels)
        # dump first event
        if self.trigger():
            self.board.get_corrected(channels[0])
        else:
            print 'no trigger found'
            return
        dt = datetime.now()
        for i in range(nevents):
            if self.trigger():
                self.board.write_event(filename, channels)
                n += 1
                if bar and n % 100 == 0:
                    print_progress(float(n) / nevents)
            else:
                if abort:
                    print 'no trigger found, aborting measurement'
                    break
        timedelta = datetime.now() - dt
        if bar:
            print ''
            print ('{} events recorded in time: {} s'
                   + '\n{:.3g} events per second').format(
                n, timedelta, n / timedelta.total_seconds())

    def get_raw(self, channel):
        return self.board.get_raw(channel)

    def write_header(self, filename, channel):
        self.board.write_header(filename, (channel, ))

    def write_raw_event(self, event, filename, channel):
        with open(filename, 'ab') as f:
            f.write('EHDR')
            f.write(pack('i', self.board.eventnum))

            date = datetime.now()
            datearr = [date.year, date.month, date.day, date.hour, date.minute,
                       date.second, date.microsecond/1000, 0]
            f.write(pack('h'*8, *datearr))
            f.write('B#')
            f.write(pack('h', self.board.get_board_serial_number()))
            f.write('T#')
            f.write(pack('h', self.board.get_trigger_cell()))

            f.write('C00{}'.format(channel + 1))
            event = (event - self.board.center + 0.5) * 65535

            f.write(event.astype(np.uint16).tostring())


def init_board():
    drs = PyDRS()
    a = drs.get_number_of_boards()
    if a:
        board = drs.get_board(0)
        sn = board.get_board_serial_number()
        fw = board.get_firmware_version()
        print ('found board with serial number #{} ' +
               'and firmware version {}.').format(sn, fw)
        return board
    else:
        return None


def trigger(board):
    # auto triggering
    board.start_domino()
    if not board.normaltrigger:
        board.soft_trigger()
        while board.is_busy():
            pass
        return True
    else:
        t = time()
        while not board.is_event_available() or board.is_busy():
            if (time() - t) > 5:
                print False
                return False
        return True


def print_progress(frac):
    length = 30
    block = int(round(length*frac))
    text = '\r[{}] {:.3g}%'.format('#'*block + '-'*(length - block), frac*100)
    stdout.write(text)
    stdout.flush()
