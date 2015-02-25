from pydrs import PyDRS
from lxml import etree
from time import time
from datetime import datetime
from sys import stdout


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
            # 100 is not the correct factor, but it works better than without
            board.set_trigger_level(self.tr_level * 10)
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

    def measure(self, filename, nevents, channel=0, abort=True, bar=True):
        n = 0
        self.board.write_header(filename, channel)
        # dump first event
        if self.trigger():
            self.board.get_corrected(channel)
        else:
            print 'no trigger found'
            return
        dt = datetime.now()
        for i in range(nevents):
            if self.trigger():
                self.board.write_event(filename, channel)
                n += 1
                if bar:
                    print_progress(float(n) / nevents)
            else:
                if abort:
                    print 'no trigger found, aborting measurement'
                    break
        timedelta = datetime.now() - dt
        if bar:
            print ''
            print ('{} events recorded in time: {}'
                   + '\n{:.3g} events per second').format(
                n, timedelta, n / timedelta.total_seconds())


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
        # board.transfer_waves()
        return True
    else:
        t = time()
        while not board.is_event_available() or board.is_busy():
            if (time() - t) > 5:
                print False
                return False
        # board.transfer_waves()
        return True


def print_progress(frac):
    length = 30
    block = int(round(length*frac))
    text = '\r[{}] {:.3g}%'.format('#'*block + '-'*(length - block), frac*100)
    stdout.write(text)
    stdout.flush()


if __name__ == "__main__":
    t = time()
    board = init_board()
    if board:
        config = DRSConfig(board)
        config.apply(board)
        for i in range(5):
            if trigger(board, config):
                board.get_corrected(0)
                print True