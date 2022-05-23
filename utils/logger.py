import logging
import os
import time


class Logger(object):
    def __init__(self, to_file=False, dataname='', model=''):
        self.to_file = to_file
        # create dictionary
        file_name = f"./log/{dataname}_{model}_{time.strftime('%Y%m%d_%H%M%S ', time.localtime(time.time()))}.log"
        self.file_name = file_name
        if not os.path.exists(file_name):
            folder_name = './log'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # set CMD dairy
        self.sh = logging.StreamHandler()
        if self.to_file:
            # set log file
            self.fh = logging.FileHandler(file_name, encoding='utf-8')

    def debug(self, message):
        self.set_color('\033[0;34m%s\033[0m', logging.DEBUG)
        self.logger.debug(message)

    def info(self, message):
        self.set_color('\033[0;30m%s\033[0m', logging.INFO)
        self.logger.info(message)

    def war(self, message):
        self.set_color('\033[0;32m%s\033[0m', logging.WARNING)
        self.logger.warning(message)

    def error(self, message):
        self.set_color('\033[0;31m%s\033[0m', logging.ERROR)
        self.logger.error(message)

    def cri(self, message):
        self.set_color('\033[0;35m%s\033[0m', logging.CRITICAL)
        self.logger.critical(message)

    def set_color(self, color, level):
        fmt = logging.Formatter(color % '[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set CMD dairy
        self.sh.setFormatter(fmt)
        self.sh.setLevel(level)
        self.logger.addHandler(self.sh)
        if self.to_file:
            # set log file
            self.fh.setFormatter(fmt)
            self.fh.setLevel(level)
            self.logger.addHandler(self.fh)