# =========================================================================
#             FILE: kicad_symlib.py
#
#      DESCRIPTION: Robust S-expression helpers for the shared eSim KiCad
#                   symbol libraries (eSim_Ngveri / eSim_NgVeriCosim /
#                   eSim_Nghdl .kicad_sym). Each library is one KiCad symbol
#                   file appended to by every generated model. The original
#                   code mutated it with raw byte/line surgery (content[:-2],
#                   lines[0:-2], line.startswith("(symbol")) which, on repeated
#                   add/overwrite/delete, glued blocks together
#                   ("))(symbol ..."), shed a part's opening "(symbol" line, and
#                   left unbalanced parens. KiCad then rejects the whole file and
#                   the library disappears. These helpers parse the file into
#                   balanced top-level part blocks keyed by model name, so
#                   add/overwrite/delete are idempotent and always re-serialize a
#                   valid, balanced file (also healing an already-corrupted file
#                   on the next write).
#
#            NOTES: Self-contained on purpose (stdlib only: re/os/tempfile).
#                   This file is vendored byte-for-byte into BOTH the eSim source
#                   tree (src/maker/kicad_symlib.py) and the separately-packaged
#                   NGHDL tarball (nghdl/src/kicad_symlib.py). Each package
#                   imports its OWN copy, so a broken/missing NGHDL install can
#                   never break eSim and vice-versa. A drift-guard test asserts
#                   the two copies stay identical -- if you edit one, copy it to
#                   the other.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================

import re
import os
import tempfile


_LIB_HEADER = ('(kicad_symbol_lib (version 20211014) '
               '(generator kicad_symbol_editor)')

# A *part* opener is distinctively '(symbol "<name>" (pin_names'. Sub-units
# '(symbol "<name>_0_1"(rectangle' / '(symbol "<name>_1_1"' are nested inside it
# and never match this, so they are carried along inside their parent's balanced
# block rather than treated as separate parts.
_PART_RE = re.compile(r'\(symbol\s+"([^"]+)"\s+\(pin_names')


def _balanced_end(text, start):
    '''Index just past the ")" that closes the "(" at text[start], honoring
       quoted strings. Returns -1 if it never balances (truncated/corrupt).'''
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _extract_parts(content):
    '''Parse a (possibly corrupt) kicad_sym into an ordered {name: block}.
       Only well-formed balanced part blocks are kept; orphaned/duplicate
       blocks are dropped and the last definition of a name wins (freshest).'''
    parts = {}
    for m in _PART_RE.finditer(content):
        end = _balanced_end(content, m.start())
        if end != -1:
            parts[m.group(1)] = content[m.start():end].strip()
    return parts


def _read_parts(path):
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        content = ''
    return _extract_parts(content)


def _write_lib(path, parts):
    '''Serialize {name: block} back into a valid, balanced kicad_sym file.
       Written atomically (temp file in the same dir + os.replace) so a crash,
       full disk, or kill -9 mid-write can never leave the shared library
       truncated or empty -- the failure mode this module exists to fix.'''
    out = [_LIB_HEADER, '']
    for block in parts.values():
        out.append(block)
        out.append('')
    out.append(')')
    data = '\n'.join(out) + '\n'
    directory = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix='.eSim_symlib_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)           # atomic on POSIX and Windows
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
