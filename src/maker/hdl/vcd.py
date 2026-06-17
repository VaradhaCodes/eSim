"""VCD (Value Change Dump) parsing for waveform display.

Pure functions, extracted verbatim from VerilogVerifier so they can be unit
tested without Qt. The Verilog Simulator IDE feeds the output of
:func:`parse_vcd_for_plot` straight into eSim's native plot window.
"""
import re


def format_vcd_val(bin_str, size, var_name=""):
    """Format a raw binary VCD string to a human-readable value.

    Only decodes as ASCII if the signal is >= 24 bits (3+ characters), all
    decoded bytes are printable, AND the variable name hints it is a string
    (contains 'name', 'str', 'msg', 'text', 'label', or 'char').
    Everything else is rendered as hexadecimal to prevent false positives on
    opcodes, counters, and single-byte data registers.
    """
    if bin_str.lower() in ('x', 'z'):
        return bin_str

    if size == 1:
        return bin_str

    # Only attempt ASCII decoding if the signal is at least 24 bits (3 chars)
    # AND the variable name explicitly suggests a string type.
    STRING_NAME_HINTS = ('name', 'str', 'msg', 'text', 'label', 'char')
    is_named_string = any(h in var_name.lower() for h in STRING_NAME_HINTS)

    if is_named_string and size >= 24:
        try:
            if len(bin_str) % 8 != 0:
                padded_str = bin_str.zfill((len(bin_str) // 8 + 1) * 8)
            else:
                padded_str = bin_str
            bytes_list = [int(padded_str[i:i+8], 2) for i in range(0, len(padded_str), 8)]
            clean_bytes = [b for b in bytes_list if b != 0]
            if clean_bytes and all(32 <= b <= 126 for b in clean_bytes):
                return '"' + "".join(chr(b) for b in clean_bytes) + '"'
        except Exception:
            pass

    try:
        val = int(bin_str, 2)
        return hex(val)
    except Exception:
        return bin_str


def parse_vcd_for_plot(vcd_content):
    lines = vcd_content.splitlines()
    vars_map = {}
    symbol_to_val = {}

    timescale = "Time"
    timescale_match = re.search(r'\$timescale\s+(.*?)\s+\$end', vcd_content, re.DOTALL)
    if timescale_match:
        timescale = timescale_match.group(1).strip()

    in_header = True
    time_series = []
    current_time = 0
    current_changes = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if in_header:
            if line.startswith('$var'):
                parts = line.split()
                if len(parts) >= 5:
                    var_type = parts[1]
                    size = int(parts[2])
                    symbol = parts[3]
                    name = parts[4]
                    vars_map[symbol] = {'name': name, 'size': size, 'type': var_type}
                    symbol_to_val[symbol] = 'x'
            elif (line.startswith('$enddefinitions')
                  or line.startswith('$dumpvars')
                  or line.startswith('$dumpall')):
                in_header = False

        if not in_header or line.startswith('#') or (line and line[0] in '01zZxXbB'):
            if line.startswith('#'):
                if current_changes:
                    time_series.append((current_time, current_changes.copy()))
                    current_changes.clear()
                current_time = int(line[1:])
            else:
                if line[0] in 'bB':
                    parts = line.split()
                    val = parts[0][1:]
                    symbol = parts[1]
                    current_changes[symbol] = val
                    symbol_to_val[symbol] = val
                else:
                    val = line[0]
                    symbol = line[1:]
                    current_changes[symbol] = val
                    symbol_to_val[symbol] = val

    if current_changes:
        time_series.append((current_time, current_changes.copy()))

    if not time_series:
        return None, None, None, None, None

    timestamps = sorted(list(set([0] + [t for t, _ in time_series])))

    # Build a forward-filled state table.
    # Use a running dict and snapshot it at each recorded timestamp so that
    # signals that did NOT change at time t still carry their previous value
    # rather than falling back to the broken single-key fallback dict.
    running_state = {symbol: 'x' for symbol in vars_map}
    raw_states = {0: running_state.copy()}
    changes_by_time = {}
    for t, ch in time_series:
        changes_by_time.setdefault(t, {}).update(ch)
    for t in sorted(changes_by_time):
        running_state.update(changes_by_time[t])
        raw_states[t] = running_state.copy()

    signals_data = {}
    raw_signals_data = {}

    for symbol, info in vars_map.items():
        name = info['name']
        size = info['size']

        y_values = []
        raw_values = []
        for t in timestamps:
            # raw_states always has a full snapshot for every recorded time;
            # for timestamps between changes, find the most recent snapshot.
            nearest_t = max((k for k in raw_states if k <= t), default=0)
            raw_val = raw_states[nearest_t].get(symbol, 'x')

            if raw_val in ('x', 'X', 'z', 'Z'):
                formatted_val = raw_val
                dec_val = 0
            else:
                formatted_val = format_vcd_val(raw_val, size, name)
                try:
                    dec_val = int(raw_val, 2)
                except Exception:
                    dec_val = 0

            y_values.append(dec_val)
            raw_values.append(formatted_val)

        signals_data[name] = y_values
        raw_signals_data[name] = raw_values

    signal_types = {info['name']: info['type'] for info in vars_map.values()}

    return timestamps, signals_data, signal_types, raw_signals_data, timescale
