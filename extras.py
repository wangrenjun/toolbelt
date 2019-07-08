#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
    'reversed_dict',
    'Singleton',
    'ConstDotDictify',
    'align_size',
    'roundup',
    'get_terminal_size',
    'xstr',
    'make_list_of_dict_from_arrays',
    'has_all_keys',
    'has_any_keys',
    'joiniterable',
    'transposelist',
    'RingLooper',
    'urlscheme',
    'cut_integer',
    'count_set_bits',
    'isv2',
    'isv3',
    'inversed_textwrap',
    'streamistty',
    'fuzzy_match',
    'substr_match',
    'has_fuzzy_matched',
    'has_substr_matched',
    'namedtupledictify',
    'colorizeansi',
    'ttyautocolorizeansi',
    'combineansi',
    'colorize',
    'ttyautocolorize',
    'combine',
    'escapeset',
    'escapereset',
    'escapefgcolor',
    'escapebgcolor',
    'ColoredSetting',
    'ColoredArgParser',
    'ColoredFormatter',
    'ColoredLogger',
    'init_colored_logger',
    'FileRead',
    'chardet',
    'trydecodingbytes',
    'PrettyTable',
    'PrettyVTable',
    'exitcode',
    'signame_to_signo',
    'signo_to_signame',
    'set_trap_handler',
    'humanizedbin',
    'humanizedtwoscompbin',
    'humanizedoct',
    'humanizedhex',
    'humanizeddec',
    'humanizedbinip',
    'humanizedbytes',
    'humanizedpercentage',
    'humanizedpercentage2',
]

import sys, os, enum, math, os, itertools, abc, json, signal, logging, argparse, collections, urllib, textwrap, difflib, ipaddress

# Swap keys for values
reversed_dict = lambda d: dict(zip(d.values(), d.keys()))

class Singleton(type):
    __instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]

class ConstDotDictify():
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)
    def __getitem__(self, key):
        return self.__dict__[key]
    __getattr__ = __getitem__
    def __readonly__(self, *args, **kwargs):
        raise AttributeError("Cannot reassign members.")
    __setattr__ = __readonly__
    __setitem__ = __readonly__
    del __readonly__

# Only to be used to align on a power of 2 boundary.
align_size = lambda size, boundary: size + (boundary - 1) & ~(boundary - 1) if math.log(boundary, 2).is_integer() else None

roundup = lambda n, b: int(math.ceil(n / b)) * b

def get_terminal_size(file = sys.stdout):
    try:
        _ = os.get_terminal_size(file.fileno())
        columns, lines = _.columns, _.lines
    except OSError:
        columns, lines = 0, 0
    return columns, lines

# Convert None, 0, False, or any falsy value to empty string.
xstr = lambda s: str(s or '')

def make_list_of_dict_from_arrays(keys, *values):
    l = []
    if keys:
        for i in [[[k, v] for k, v in zip(keys, vals)] for vals in values ]:
            l.append(dict(i))
    return l

has_all_keys = lambda dict, keys: all(_ in dict for _ in keys)
has_any_keys = lambda dict, keys: any(_ in dict for _ in keys)

joiniterable = lambda sep, seq: sep.join(map(str, seq))

transposelist = lambda l: list(map(list, zip(*l)))

class RingLooper():
    def __init__(self, *array):
        self.__array = array
    def __iter__(self):
        self.__cycler = itertools.cycle(range(len(self.__array)))
        return self
    def __next__(self):
        if self.__array:
            return self.__array[next(self.__cycler)]
        else:
            raise StopIteration

urlscheme = lambda url: urllib.parse.urlparse(url).scheme

cut_integer = lambda num, bits: num & ((1<< bits) - 1)

def count_set_bits(n):
    count = 0
    while (n):
        n = n & (n - 1)
        count += 1
    return count

isv2 = lambda : True if sys.version_info[0] == 2 else False
isv3 = lambda : True if sys.version_info[0] == 3 else False

"""
>>> inversed_textwrap('1234567890', 3)
['1', '234', '567', '890']
"""
inversed_textwrap = lambda text, wrapwidth: list(map(lambda x:x[::-1], textwrap.wrap(text[::-1], wrapwidth)[::-1]))

# Check filestream is attached to terminal.
streamistty = lambda file: True if file.isatty() else False

_DEFAULT_THRESHOLD = 0.7

def fuzzy_match(needle, haystack, ignore_case = False, threshold = _DEFAULT_THRESHOLD):
    if ignore_case:
        matching = lambda entry: difflib.SequenceMatcher(a = needle.lower(), b = entry.lower()).ratio() >= threshold
    else:
        matching = lambda entry: difflib.SequenceMatcher(a = needle, b = entry).ratio() >= threshold
    return filter(match, haystack)

def substr_match(needle, haystack, ignore_case = False):
    if ignore_case:
        matching = lambda entry: needle.lower() in entry.lower()
    else:
        matching = lambda entry: needle in entry
    return filter(match, haystack)

def has_fuzzy_matched(needle, haystack, ignore_case = False, threshold = _DEFAULT_THRESHOLD):
    if ignore_case:
        matching = lambda entry: difflib.SequenceMatcher(a = needle.lower(), b = entry.lower()).ratio() >= threshold
    else:
        matching = lambda entry: difflib.SequenceMatcher(a = needle, b = entry).ratio() >= threshold
    for entry in haystack:
        if matching(entry):
            return True
    return False

def has_substr_matched(needle, haystack, ignore_case = False):
    if ignore_case:
        matching = lambda entry: needle.lower() in entry.lower()
    else:
        matching = lambda entry: needle in entry
    for entry in haystack:
        if matching(entry):
            return True
    return False

namedtupledictify = lambda nt: dict(nt._asdict())

_sets = dict(zip(
        ( 'bold', 'dim', 'italic', 'underlined', 'blink', 'rapid blink', 'reverse', 'hidden', 'crossed out', ),
        ( str(_) for _ in range(1, 10))))
_resets = dict(zip(
        ( 'bold', 'dim', 'italic', 'underlined', 'blink', 'rapid blink', 'reverse', 'hidden', 'crossed out', ),
        ( str(_) for _ in range(21, 30))))
_resets.update({ 'normal' : '0', 'fg reset' : '39', 'bg reset' : '49', })
_fgcolors = dict(zip(
        ( 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'silver', ),
        ( str(_) for _ in range(30, 38))))
_fgcolors.update(dict(zip(
        ( 'grey', 'light red', 'light green', 'light yellow', 'light blue', 'light magenta', 'light cyan', 'white', ),
        ( str(_) for _ in range(90, 98)))))
_bgcolors = dict(zip(
        ( 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'silver', ),
        ( str(_) for _ in range(40, 48))))
_bgcolors.update(dict(zip(
        ( 'grey', 'light red', 'light green', 'light yellow', 'light blue', 'light magenta', 'light cyan', 'white', ),
        ( str(_) for _ in range(100, 108)))))

def _esc(codes):
    if isinstance(codes, (int, str)):
        return '\033[{}m'.format(codes)
    elif isinstance(codes, collections.Iterable):
        return '\033[{}m'.format(joiniterable(';', codes))
    else:
        raise TypeError('must be one of int, str, iterable' + ', not ' + type(codes).__name__)

def colorizeansi(*args, enabling = True, **style):
    argstring = joiniterable(' ', args)
    if enabling:
        paint = combineansi(**style)
        if paint:
            argstring = paint + argstring + escapereset.NORMAL.value
    return argstring

def ttyautocolorizeansi(stream, *args, **style):
    return colorizeansi(*args, enabling = streamistty(stream), **style)

def combineansi(**style):
    fields = {
        'set' :     lambda x: [ _sets[_] for _ in x ],
        'reset' :   lambda x: [ _resets[_] for _ in x ],
        'fgcolor' : lambda x: [ _fgcolors[x] ],
        'bgcolor' : lambda x: [ _bgcolors[x] ],
    }
    l = []
    for k, v in style.items():
        f = fields.get(k)
        if f:
            l.extend(f(v))
    return _esc(l) if l else ''

def colorize(*args, enabling = True, **style):
    return colorizeansi(*args, enabling = enabling, **style)

def ttyautocolorize(stream, *args, **style):
    return ttyautocolorizeansi(stream, *args, **style)

def combine(**style):
    return combineansi(**style)

escapeset = enum.Enum('escapeset', { '_'.join(k.upper().split(' ')): _esc(v) for k, v in _sets.items() })
escapereset = enum.Enum('escapereset', { '_'.join(k.upper().split(' ')): _esc(v) for k, v in _resets.items() })
escapefgcolor = enum.Enum('escapefgcolor', { '_'.join(k.upper().split(' ')): _esc(v) for k, v in _fgcolors.items() })
escapebgcolor = enum.Enum('escapebgcolor', { '_'.join(k.upper().split(' ')): _esc(v) for k, v in _bgcolors.items() })

class ColoredSetting(metaclass = Singleton):
    def __init__(self, when = 'auto', autocb = streamistty):
        if when == 'always':
            self.__has_enabled_colorize = True
        elif when == 'never':
            self.__has_enabled_colorize = False
        else:
            self.__has_enabled_colorize = autocb
    def is_colorize(self, stream):
        return self.__has_enabled_colorize if isinstance(self.__has_enabled_colorize, bool) else self.__has_enabled_colorize(stream)

class ColoredArgParser(argparse.ArgumentParser):
    __styles = { 'usage' : { 'fgcolor' : 'yellow', 'set' : ( 'bold', ) },
                 'help' :  { 'fgcolor' : 'blue', 'set' : ( 'bold', ) },
                 'error' : { 'fgcolor' : 'red', 'set' : ( 'bold', ) }, }

    def __init__(self, **kwargs):
        styles = kwargs.pop('styles', None)
        super().__init__(**kwargs)
        if styles != None:
            self.__styles = styles

    def print_usage(self, file = sys.stdout):
        usage = self.format_usage()
        usage = usage[0].upper() + usage[1:]
        self._print_message(colorize(usage, enabling = ColoredSetting().is_colorize(file), **self.__styles['usage']), file)

    def print_help(self, file = sys.stdout):
        help = self.format_help()
        help = help[0].upper() + help[1:]
        self._print_message(colorize(help, enabling = ColoredSetting().is_colorize(file), **self.__styles['help']), file)

    def exit(self, status = 0, message = None):
        if message:
            self._print_message(colorize(message, enabling = ColoredSetting().is_colorize(sys.stderr), **self.__styles['error']), sys.stderr)
        sys.exit(status)

    def error(self, message):
        self.print_usage(sys.stderr)
        message = '%(prog)s: ERROR: %(message)s\n' % { 'prog': self.prog, 'message': message }
        self.exit(2, message)

    def print_error(self, message):
        message = '%(prog)s: ERROR: %(message)s\n' % { 'prog': self.prog, 'message': message }
        self._print_message(colorize(message, enabling = ColoredSetting().is_colorize(sys.stderr), **self.__styles['error']), sys.stderr)

_levelstyles = { 'INFO' :     { 'fgcolor' : 'green', 'set' : ( 'bold', ) },
                 'WARNING':   { 'fgcolor' : 'yellow', 'set' : ( 'bold', ) },
                 'ERROR' :    { 'fgcolor' : 'red', 'set' : ( 'bold', ) },
                 'CRITICAL' : { 'fgcolor' : 'light red', 'set' : ( 'bold', ) }, }

_default_format = '%(filename)s: %(levelname)s: %(message)s'
_default_dateformat = '%Y-%m-%d %H:%M:%S %p'

class ColoredFormatter(logging.Formatter):
    def __init__(self,
        fmt = None,
        datefmt = None,
        stream = sys.stderr):
        super().__init__(_default_format if fmt is None else fmt,
            _default_dateformat if datefmt is None else datefmt)
        self.__stream = stream

    def format(self, record):
        msg = super().format(record)
        return str(colorize(msg, enabling = ColoredSetting().is_colorize(self.__stream), **_levelstyles.get(record.levelname, {})))

class ColoredLogger(logging.Logger):
    def __init__(self, name, level = logging.NOTSET, stream = sys.stderr):
        super().__init__(name, level)
        self.__stream_handler = logging.StreamHandler(stream)
        self.__stream_handler.setFormatter(ColoredFormatter(stream = stream))
        self.addHandler(self.__stream_handler)

def init_colored_logger():
    logging.addLevelName(logging.DEBUG, 'DEBUG')
    logging.addLevelName(logging.INFO, 'INFO')
    logging.addLevelName(logging.WARNING, 'WARNING')
    logging.addLevelName(logging.ERROR, 'ERROR')
    logging.addLevelName(logging.CRITICAL, 'CRITICAL')
    logging.setLoggerClass(ColoredLogger)

class FileRead:
    def __init__(self,
        files = None,
        mode = 'r',
        prompt_when_stdin = None,
        openhook = None,
        readhook = None):
        if isinstance(files, str):
            files = (files, )
        elif isinstance(files, os.PathLike):
            files = (os.fspath(files), )
        else:
            if not files:
                files = ('-', )
            else:
                files = tuple(files)
        self.__files = files
        self.__filename = None
        self.__isstdin = False
        if mode not in ('r', 'rb', 'rt'):
            raise ValueError("FileRead opening mode must be one of 'r', 'rb' and 'rt'")
        self.__mode = mode
        self.__prompt_when_stdin = prompt_when_stdin
        self.__openhook = openhook or FileRead.hook_open()
        self.__readhook = readhook or (lambda f: f.read())

    def __read(self):
        for self.__filename in self.__files:
            self.__isstdin = False
            if self.__filename == '-':
                self.__filename = sys.stdin.name    # '<stdin>'
                self.__isstdin = True
                if self.__prompt_when_stdin:
                    print(self.__prompt_when_stdin, flush = True)
                if 'b' in self.__mode:
                    content = self.__readhook(getattr(sys.stdin, 'buffer', sys.stdin))
                else:
                    content = self.__readhook(sys.stdin)
            else:
                content = self.__readhook(self.__openhook(self.__filename, self.__mode))
            yield content

    def __iter__(self):
        return self.__read()

    def isstdin(self):
        return self.__isstdin

    def filename(self):
        return self.__filename

    @classmethod
    def hook_open(cls, encoding = None, errors = None, newline = None):
        return lambda f, m: open(f, mode = m, encoding = encoding, errors = errors, newline = newline)

def chardet(bs):
    try:
        import chardet
        return chardet.detect(bs)
    except ImportError:
        try:
            import cchardet
            return cchardet.detect(bs)
        except ImportError:
            raise

def trydecodingbytes(bs):
    try:
        import bs4
        dammit = bs4.UnicodeDammit(bs)
        unicode_markup, original_encoding, tried_encodings = dammit.unicode_markup, dammit.original_encoding, dammit.tried_encodings
    except ImportError:
        unicode_markup, original_encoding, tried_encodings = None, None, None
    if unicode_markup:
        return unicode_markup, original_encoding
    try:
        encoding = chardet(bs)['encoding']
    except ImportError:
        encoding = None
    if not encoding and tried_encodings and tried_encodings[0] and tried_encodings[0][0]:
        encoding = tried_encodings[0][0]
    if encoding:
        try:
            s = bs.decode(encoding = encoding)
        except Exception:
            s = str(bs)
            encoding = None
    return s, encoding

_default_field_paint = {'set' : ( 'bold', )}

_default_palette_paints = ( { 'fgcolor' : 'green',          'set' : ( 'bold', ) },
                            { 'fgcolor' : 'yellow',         'set' : ( 'bold', ) },
                            { 'fgcolor' : 'blue',           'set' : ( 'bold', ) },
                            { 'fgcolor' : 'magenta',        'set' : ( 'bold', ) },
                            { 'fgcolor' : 'cyan',           'set' : ( 'bold', ) },
                            { 'fgcolor' : 'light green',    'set' : ( 'bold', ) },
                            { 'fgcolor' : 'light yellow',   'set' : ( 'bold', ) },
                            { 'fgcolor' : 'light blue',     'set' : ( 'bold', ) },
                            { 'fgcolor' : 'light magenta',  'set' : ( 'bold', ) },
                            { 'fgcolor' : 'light cyan',     'set' : ( 'bold', ) }, )

class PrettyTable():
    def __init__(self,
        enable_painting = True,
        field_paint = None,
        palette_paints = None,
        padding_characters = '    ',
        align_boundary = 4):
        self.__enable_painting = enable_painting
        self.__field_paint = field_paint if field_paint != None else _default_field_paint
        self.__palette_paints = palette_paints if palette_paints != None else _default_palette_paints
        self.__padding_characters = padding_characters
        self.__align_boundary = align_boundary
        self.__fields = []
        self.__field_alignments = []
        self.__fixed_width_of_fields = []
        self.__records = []
        self.__max_size_of_columns = []

    def add_field(self, name, alignment = '<', fixed_width = None):
        self.__fields.append(name)
        if alignment not in '<^>':
            raise ValueError("alignment must be one of '<', '>' and '^'")
        self.__field_alignments.append(alignment)
        self.__fixed_width_of_fields.append(fixed_width)
        self.__max_size_of_columns.append(len(name))

    def add_record(self, *record):
        if len(record) != len(self.__fields):
            raise ValueError('record and fields must have the same length')
        record = list(map(str, record))
        self.__records.append(record)
        self.__max_size_of_columns = [ max(len(_), __) for _, __ in zip(record, self.__max_size_of_columns) ]

    def to_string(self):
        aligned_width_of_columns = map(lambda x: align_size(x, self.__align_boundary), self.__max_size_of_columns)
        aligned_width_of_columns = [ i if j == None else j for i, j in zip(aligned_width_of_columns, self.__fixed_width_of_fields) ]
        fmtstr = self.__padding_characters.join(['{:%s%d}' % (_, __) for _, __ in zip(self.__field_alignments, aligned_width_of_columns)])
        string = colorize(fmtstr.format(*self.__fields), enabling = self.__enable_painting, **self.__field_paint)
        string += os.linesep
        paletteiter = iter(RingLooper(*self.__palette_paints))
        for r in self.__records:
            string += colorize(fmtstr.format(*r), enabling = self.__enable_painting, **next(paletteiter))
            string += os.linesep
        return string

    def __str__(self):
        return self.to_string()

class PrettyVTable():
    def __init__(self,
        enable_painting = True,
        palette_paints = None,
        padding_characters = '    ',
        align_boundary = 4):
        self.__enable_painting = enable_painting
        self.__palette_paints = palette_paints if palette_paints != None else _default_palette_paints
        self.__padding_characters = padding_characters
        self.__align_boundary = align_boundary
        self.__fields = []
        self.__groups = []
        self.__column_alignments = []
        self.__fixed_width_of_columns = []
        self.__max_size_of_columns = []
        self.__column_cycler = None

    def add_column(self, alignment = '<', fixed_width = None):
        if alignment not in '<^>':
            raise ValueError("alignment must be one of '<', '>' and '^'")
        self.__column_alignments.append(alignment)
        self.__fixed_width_of_columns.append(fixed_width)
        self.__max_size_of_columns.append(0)

    def __make_default_columns(self, num, alignment = '<', fixed_width = None):
        for i in range(num):
            self.add_column(alignment, fixed_width)

    def add_field(self, name):
        if not self.__column_alignments:
            self.__make_default_columns(2)
        elif len(self.__column_alignments) < 2:
            raise ValueError('must be at least two columns')
        self.__fields.append(name)
        self.__max_size_of_columns[0] = max(len(name), self.__max_size_of_columns[0])

    def add_record(self, *record):
        if len(record) != len(self.__fields):
            raise ValueError('record and fields must have the same length')
        record = list(map(str, record))
        number_of_columns = len(self.__column_alignments)
        if self.__column_cycler is None:
            self.__column_cycler = itertools.cycle(range(1, number_of_columns))
        columnidx = next(self.__column_cycler)
        if columnidx == 1:
            empty_records_for_fill = [[''] * len(self.__fields)] * (number_of_columns - 2)
            self.__groups.append([self.__fields, record, *empty_records_for_fill])
        else:
            self.__groups[-1][columnidx] = record
        self.__max_size_of_columns[columnidx] = max(*map(len, record), self.__max_size_of_columns[columnidx])

    def to_string(self):
        aligned_width_of_columns = map(lambda x: align_size(x, self.__align_boundary), self.__max_size_of_columns)
        if self.__fixed_width_of_columns != None:
            aligned_width_of_columns = [ i if j == None else j for i, j in zip(aligned_width_of_columns, self.__fixed_width_of_columns) ]
        fmtstr = self.__padding_characters.join(['{:%s%d}' % (_, __) for _, __ in zip(self.__column_alignments, aligned_width_of_columns)])
        string = ''
        paletteiter = iter(RingLooper(*self.__palette_paints))
        for group in self.__groups:
            group = transposelist(group)
            paint = next(paletteiter)
            for row in group:
                string += colorize(fmtstr.format(*row), enabling = self.__enable_painting, **paint)
                string += os.linesep
        return string

    def __str__(self):
        return self.to_string()

class exitcode(enum.IntEnum):
    EC_OK = 0                   # Successful termination
    EC_ERROR = 1                # General errors
    EC_USAGE_ERROR = 2          # Usage error
    EC_ENOENT = 97              # No such file or directory
    EC_INVAL = 98               # Invalid argument
    EC_ASSERT_FAILED = 99
    EC_ILLEGAL_CMD = 127        # Command not found
    EC_FATAL_SIGNAL_BASE = 128  # Base value for fatal error signal "n"
    EC_CONTRL_C = 130           # Script terminated by Control-C(128 + 2)

_siglist = dict((k, v) for v, k in reversed(sorted(signal.__dict__.items())) if v.startswith('SIG') and not v.startswith('SIG_'))

def signame_to_signo(sname):
    return int(getattr(signal, sname, 0))

def signo_to_signame(sno):
    return _siglist.get(sno)

def _on_trapped(signo, frame):
    print(str(colorize('Interrupted by %s' % (signo_to_signame(signo)), enabling = ColoredSetting().is_colorize(sys.stderr), fgcolor = 'green', set = ( 'bold', ))), file = sys.stderr)
    sys.exit(exitcode.EC_FATAL_SIGNAL_BASE + signo)

def set_trap_handler(sigs, handler = None):
    if handler and not callable(handler):
        raise TypeError("'" + type(handler).__name__ + "'" + ' object is not callable')
    for _ in sigs:
        signal.signal(_, handler or _on_trapped)

def humanizedbin(num, wrapwidth = 4, sep = ' '):
    if num < 0:
        sign = '-'
        s = '{:b}'.format(num)[1:]
    else:
        sign = ''
        s = '{:b}'.format(num)
    return '%s%s' % (sign, sep.join(textwrap.wrap(s.zfill(roundup(len(s), wrapwidth)), wrapwidth)))

# Format two's complement representation
def humanizedtwoscompbin(num, bits = 32, wrapwidth = 4, sep = ' '):
    s = '{:b}'.format(cut_integer(num, bits))
    return '%s' % (sep.join(textwrap.wrap(s.zfill(bits), wrapwidth)))

def humanizedoct(num, wrapwidth = 3, sep = ' '):
    if num < 0:
        sign = '-'
        s = '{:o}'.format(num)[1:]
    else:
        sign = ''
        s = '{:o}'.format(num)
    return '%s%s' % (sign, sep.join(inversed_textwrap(s, wrapwidth)))

def humanizedhex(num, wrapwidth = 4, sep = ' ', with_capitals = True):
    if num < 0:
        sign = '-'
        s = '{:{type}}'.format(num, type = 'X' if with_capitals else 'x')[1:]
    else:
        sign = ''
        s = '{:{type}}'.format(num, type = 'X' if with_capitals else 'x')
    return '%s%s' % (sign, sep.join(inversed_textwrap(s, wrapwidth)))

def humanizeddec(num, wrapwidth = 3, sep = ','):
    if num < 0:
        sign = '-'
        s = '{:d}'.format(num)[1:]
    else:
        sign = ''
        s = '{:d}'.format(num)
    return '%s%s' % (sign, sep.join(inversed_textwrap(s, wrapwidth)))

def humanizedbinip(ipaddr):
    if ipaddr.version == 4:
        fillwidth = 32
        wrapwidth = 8
    else:
        fillwidth = 128
        wrapwidth = 16
    return '.'.join(textwrap.wrap('{:b}'.format(int(ipaddr)).zfill(fillwidth), wrapwidth))

__capacity_symbols = {
    'traditionalbytes' : (
    ('YB',  'yottabyte',    10 ** 24    ),
    ('ZB',  'zetabyte',     10 ** 21    ),
    ('EB',  'exabyte',      10 ** 18    ),
    ('PB',  'petabyte',     10 ** 15    ),
    ('TB',  'terabyte',     10 ** 12    ),
    ('GB',  'gigabyte',     10 ** 9     ),
    ('MB',  'megabyte',     10 ** 6     ),
    ('kB',  'kilobyte',     10 ** 3     ),),
    'iecbytes' : (
    ('YiB', 'yobibyte',     1 << 80     ),
    ('ZiB', 'zebibyte',     1 << 70     ),
    ('EiB', 'exbibyte',     1 << 60     ),
    ('PiB', 'pebibyte',     1 << 50     ),
    ('TiB', 'tebibyte',     1 << 40     ),
    ('GiB', 'gibibyte',     1 << 30     ),
    ('MiB', 'mebibyte',     1 << 20     ),
    ('KiB', 'kibibyte',     1 << 10     ),),
}

def humanizedbytes(size, to = 'traditionalbytes', precision = 1):
    if isinstance(size, (int, float)):
        if size < 0:
            raise ValueError('size < 0')
        bytes = size
    elif isinstance(size, str):
        for name, symbols in __capacity_symbols.items():
            for item in symbols:
                for i in item[:2]:
                    if size.lower().endswith(i.lower()):
                        bytes = float(size[:-len(i)]) * item[2]
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            raise ValueError("can't parse " + size)
    else:
        TypeError('must be int, float or str' + ', not ' + type(size).__name__)
    num = bytes
    symbol = 'B'
    for item in __capacity_symbols[to]:
        if bytes >= item[2]:
            num = float(bytes) / item[2]
            symbol = item[0]
            break
    return '{:.{precision}f} {}'.format(num, symbol, precision = precision)

humanizedpercentage = lambda n: '{:.{precision}f}%'.format(n, precision = 1)
humanizedpercentage2 = lambda n: '{:.{precision}%}'.format(n, precision = 1)
