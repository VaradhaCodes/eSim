# =========================================================================
# FILE: CosimLogger.py
#
# PURPOSE: Structured, dual-sink logging for the d_cosim (Icarus Verilog)
#          flow -- build, netlist staging and simulation run.
#
#   Before this module the d_cosim flow logged in two disconnected ways:
#     * build_cosim() appended raw HTML to the NgVeri GUI terminal only, and
#     * the netlist stager / run path used bare print()/logger calls that went
#       only to the OS terminal (and, with no logging config, often nowhere).
#   So no single audience ever saw the whole story, nothing was timestamped or
#   severity-tagged, and the exact compiler command / return code were hidden.
#
#   CosimLog fixes that: every event is emitted to BOTH
#     1. the Python 'esim.dcosim' logger -> the OS terminal eSim was launched
#        from AND a rotating ~/.esim/dcosim.log file (timestamped, severity-
#        tagged, grep-able for developers), and
#     2. an optional GUI QTextEdit (the NgVeri tab terminal) as a severity-
#        colored line, so an end user reading the GUI sees the same events.
#
# Developed for the eSim ubuntu-26.04-support fork.
# =========================================================================

import os
import sys
import logging
from html import escape

# Parent logger for the whole d_cosim flow. Module loggers can be children
# (e.g. 'esim.dcosim.build') and inherit these handlers.
_ROOT_NAME = 'esim.dcosim'
_CONFIGURED = False


def configure_logging():
    '''
        Idempotently attach a console + rotating-file handler to the
        'esim.dcosim' logger. Safe to call many times (guards against
        duplicate handlers). Honors ESIM_DEBUG=1 for verbose console output;
        the logfile is always DEBUG-level.
    '''
    global _CONFIGURED
    if _CONFIGURED:
        return

    console_level = logging.DEBUG if os.environ.get('ESIM_DEBUG') \
        else logging.INFO
    fmt = logging.Formatter(
        '%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S')

    root = logging.getLogger(_ROOT_NAME)
    # Root gates first, so keep it at the most verbose level any handler wants
    # and let each handler filter; otherwise the file handler never sees DEBUG.
    root.setLevel(logging.DEBUG)
    # Don't double-print through the application's root logger.
    root.propagate = False

    # Console -> the OS terminal eSim was launched from.
    if not any(getattr(h, '_esim_console', False) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(console_level)
        ch._esim_console = True
        root.addHandler(ch)

    # Rotating logfile -> persistent, always-verbose record under ~/.esim.
    if not any(getattr(h, '_esim_file', False) for h in root.handlers):
        try:
            from logging.handlers import RotatingFileHandler
            logdir = os.path.join(os.path.expanduser('~'), '.esim')
            os.makedirs(logdir, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(logdir, 'dcosim.log'),
                maxBytes=1_000_000, backupCount=3)
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG)
            fh._esim_file = True
            root.addHandler(fh)
        except OSError:
            # Read-only HOME etc. -- console logging still works.
            pass

    _CONFIGURED = True


class CosimLog:
    '''
        Severity-aware logger that fans every message out to the Python
        'esim.dcosim' logger (terminal + file) and, when given one, to a GUI
        QTextEdit (the NgVeri terminal) as a colored HTML line.

        param termedit: optional QTextEdit whose .append() renders HTML; pass
                        None for non-GUI contexts (netlist staging, sim run)
                        where only terminal+file output is wanted.
    '''

    _COLOR = {
        'info': '#000000',
        'ok': '#00AA00',
        'warn': '#E07B00',
        'error': '#FF0000',
        'phase': '#0000FF',
        'detail': '#666666',
        'fix': '#B30086',
        'stdout': '#333333',
        'stderr': '#B00000',
    }

    def __init__(self, termedit=None, name=_ROOT_NAME):
        configure_logging()
        self.termedit = termedit
        self.logger = logging.getLogger(name)

    # -- internal GUI sink --------------------------------------------------
    def _gui(self, html):
        if self.termedit is not None:
            self.termedit.append(html)

    def _span(self, msg, color, size=12, weight=500):
        self._gui(
            '<span style="font-size:%dpt; font-weight:%d; color:%s;">%s</span>'
            % (size, weight, color, escape(msg)))

    # -- public API ---------------------------------------------------------
    def phase(self, title):
        '''Bold blue section header marking a new step of the flow.'''
        self.logger.info('=== %s ===', title)
        self._gui(
            '<span style="font-size:14pt; font-weight:800; color:%s;">'
            '<br>&#9654;&nbsp;%s</span>' % (self._COLOR['phase'], escape(title)))

    def info(self, msg):
        '''Normal progress line (INFO).'''
        self.logger.info(msg)
        self._span(msg, self._COLOR['info'])

    def detail(self, msg):
        '''Low-priority diagnostic: DEBUG on terminal/file, dim grey in GUI.'''
        self.logger.debug(msg)
        self._span(msg, self._COLOR['detail'], size=11)

    def ok(self, msg):
        '''Success line (green).'''
        self.logger.info(msg)
        self._span(msg, self._COLOR['ok'], weight=600)

    def warn(self, msg):
        '''Warning line (orange).'''
        self.logger.warning(msg)
        self._span(msg, self._COLOR['warn'], weight=600)

    def error(self, msg):
        '''Failure line (red).'''
        self.logger.error(msg)
        self._span(msg, self._COLOR['error'], weight=700)

    def fix(self, msg):
        '''Actionable remediation hint shown after an error.'''
        self.logger.error('Fix: %s', msg)
        self._span('Fix: ' + msg, self._COLOR['fix'], weight=600)

    def output(self, text, stream='stdout'):
        '''
            Echo a captured subprocess stream block verbatim in a monospace
            box. stream='stderr' is logged at WARNING and rendered red.
        '''
        text = (text or '').rstrip()
        if not text:
            return
        if stream == 'stderr':
            self.logger.warning('[iverilog stderr]\n%s', text)
        else:
            self.logger.info('[iverilog stdout]\n%s', text)
        self._gui(
            '<pre style="margin:2px 0; font-size:11pt; color:%s;">%s</pre>'
            % (self._COLOR[stream], escape(text)))
