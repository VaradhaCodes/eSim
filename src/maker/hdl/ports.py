"""Verilog structural analysis: module discovery, port extraction, dependency
ordering, and testbench-stub generation.

Pure functions (no Qt), so the Verify stage's design-side actions can be tested
without driving the GUI. The guiding rule here is *be comment- and string-proof*
and *degrade to something usable* rather than silently emitting garbage:

- ``strip_comments`` neutralises ``//``/``/* */`` comments and string literals
  before any structural regex, so a module name mentioned in a comment can never
  be mistaken for real code.
- ``extract_ports`` prefers ``hdlparse`` but falls back to an in-house regex
  parser whenever hdlparse yields nothing -- notably for *single-line* ANSI
  headers (``module m(input a, output b);``), which hdlparse silently parses as
  zero ports. Bus widths (``[3:0]``) are carried through so the stub declares
  matching-width regs/wires instead of truncating to one bit.
- ``order_modules`` topologically orders multi-module designs (parent before the
  modules it instantiates), tolerant of ``#(...)`` parameter overrides, comments,
  duplicate names, and dependency cycles.
"""
import re

# Net/type keywords that may sit between a direction keyword and the port name.
_TYPE_KW = {'wire', 'reg', 'logic', 'bit', 'signed', 'unsigned', 'var',
            'tri', 'wand', 'wor', 'integer', 'time', 'real'}
_DIR_KW = {'input', 'output', 'inout'}
_KW = _TYPE_KW | _DIR_KW

_MODULE_RE = re.compile(r'\bmodule\s+(\w+)')


def strip_comments(code):
    """Return ``code`` with ``//`` line comments, ``/* */`` block comments and
    ``"..."`` string literals blanked to spaces, newlines preserved. Structural
    scans run on the result so a ``module``/instance name inside a comment or
    string is never seen as code."""
    out = []
    i, n = 0, len(code)
    state = None  # None | 'line' | 'block' | 'string'
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ''
        if state is None:
            if c == '/' and nxt == '/':
                state = 'line'; out.append('  '); i += 2; continue
            if c == '/' and nxt == '*':
                state = 'block'; out.append('  '); i += 2; continue
            if c == '"':
                state = 'string'; out.append(' '); i += 1; continue
            out.append(c); i += 1
        elif state == 'line':
            if c == '\n':
                state = None; out.append('\n')
            else:
                out.append(' ')
            i += 1
        elif state == 'block':
            if c == '*' and nxt == '/':
                state = None; out.append('  '); i += 2
            else:
                out.append('\n' if c == '\n' else ' '); i += 1
        else:  # string
            if c == '\\' and nxt:
                out.append('  '); i += 2
            elif c == '"':
                state = None; out.append(' '); i += 1
            else:
                out.append('\n' if c == '\n' else ' '); i += 1
    return ''.join(out)


def find_modules(code):
    """Names of every ``module`` definition in ``code``, in source order.
    Comments/strings are ignored."""
    return _MODULE_RE.findall(strip_comments(code))


def _strip_param_overrides(code):
    """Blank ``#( ... )`` parameter-override blocks (balanced parens) so an
    instantiation always reads as ``ModuleName instanceName (`` regardless of
    its parameter list, which may itself contain parentheses."""
    out = []
    i, n = 0, len(code)
    while i < n:
        if code[i] == '#':
            j = i + 1
            while j < n and code[j] in ' \t\r\n':
                j += 1
            if j < n and code[j] == '(':
                depth = 0
                k = j
                while k < n:
                    if code[k] == '(':
                        depth += 1
                    elif code[k] == ')':
                        depth -= 1
                        if depth == 0:
                            k += 1
                            break
                    k += 1
                out.append(' ' * (k - i))
                i = k
                continue
        out.append(code[i]); i += 1
    return ''.join(out)


def _instantiated_names(clean_no_params, candidates):
    """Subset of ``candidates`` (module names) instantiated in
    ``clean_no_params`` (comments AND ``#(...)`` already stripped). An
    instantiation reads ``ModuleName instanceName [array] (`` -- the required
    instance identifier between the type and ``(`` is what separates it from the
    module's own header (``module m (``) and from function calls (``m(``)."""
    found = set()
    for nm in candidates:
        if re.search(rf'\b{re.escape(nm)}\b\s+\w+\s*(?:\[[^\]]*\]\s*)?\(',
                     clean_no_params):
            found.add(nm)
    return found


def _pick_top(code, names):
    """The top-level module among ``names``: the one no sibling instantiates.
    Ties (or none) resolve to the last-defined, the usual place for a top
    module. A single-module file returns that module."""
    if len(names) == 1:
        return names[0]
    clean = _strip_param_overrides(strip_comments(code))
    instantiated = _instantiated_names(clean, set(names))
    tops = [n for n in names if n not in instantiated]
    return tops[-1] if tops else names[-1]


def _norm_width(data_type):
    """Pull the bus range out of an hdlparse ``data_type`` (e.g. ``'reg [3:0]'``
    -> ``'[3:0]'``; packed dims preserved). '' when scalar."""
    return ''.join(re.findall(r'\[[^\]]*\]', data_type or ''))


def _ports_hdlparse(code, module_name):
    """Ports of ``module_name`` via hdlparse, as ``(mode, name, width)``. Empty
    on any failure or when hdlparse finds no ports (its single-line-ANSI blind
    spot) so the caller can fall back to the regex parser."""
    try:
        import hdlparse.verilog_parser as vlog
        objs = vlog.VerilogExtractor().extract_objects_from_source(code)
        for obj in objs:
            if obj.name == module_name:
                return [(p.mode, p.name, _norm_width(p.data_type))
                        for p in obj.ports]
    except Exception as e:
        print("hdlparse failed, falling back to regex:", e)
    return []


def _module_region(clean, name):
    """``(header, body)`` for module ``name`` in already-comment-stripped
    ``clean``: ``header`` is the text inside the port-list parens (after any
    ``#(...)``), ``body`` runs to the matching ``endmodule``."""
    m = re.search(rf'\bmodule\s+{re.escape(name)}\b', clean)
    if not m:
        return '', ''
    n = len(clean)
    j = m.end()
    while j < n and clean[j] in ' \t\r\n':
        j += 1
    # skip optional #( ... ) parameter block
    if j < n and clean[j] == '#':
        k = j + 1
        while k < n and clean[k] in ' \t\r\n':
            k += 1
        if k < n and clean[k] == '(':
            depth = 0
            while k < n:
                if clean[k] == '(':
                    depth += 1
                elif clean[k] == ')':
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            j = k
    while j < n and clean[j] in ' \t\r\n':
        j += 1
    header = ''
    if j < n and clean[j] == '(':
        depth = 0
        start = j + 1
        while j < n:
            if clean[j] == '(':
                depth += 1
            elif clean[j] == ')':
                depth -= 1
                if depth == 0:
                    header = clean[start:j]
                    j += 1
                    break
            j += 1
    end = re.search(r'\bendmodule\b', clean[j:])
    body = clean[j:j + end.start()] if end else clean[j:]
    return header, body


def _split_commas(s):
    """Split on top-level commas only (commas inside ``()``/``[]``/``{}`` stay)."""
    out, cur, depth = [], [], 0
    for c in s:
        if c in '([{':
            depth += 1; cur.append(c)
        elif c in ')]}':
            depth -= 1; cur.append(c)
        elif c == ',' and depth == 0:
            out.append(''.join(cur)); cur = []
        else:
            cur.append(c)
    if ''.join(cur).strip():
        out.append(''.join(cur))
    return out


def _names_in(decl_fragment):
    """Identifier names in a declaration fragment, with ranges and keywords
    removed -- e.g. ``'wire [3:0] a'`` -> ``['a']``."""
    no_range = re.sub(r'\[[^\]]*\]', ' ', decl_fragment)
    return [t for t in re.findall(r'[A-Za-z_]\w*', no_range) if t not in _KW]


def _ports_regex(clean, module_name):
    """In-house port parser over comment-stripped ``clean``. Handles both ANSI
    headers (``module m(input [3:0] a, output b)``, incl. single-line, which
    hdlparse misses) and non-ANSI (names in header, directions declared in the
    body). Returns ``(mode, name, width)`` with bus widths preserved."""
    header, body = _module_region(clean, module_name)
    ports, seen = [], set()

    if re.search(r'\b(input|output|inout)\b', header):
        # ANSI: each comma item carries (or inherits) a direction.
        mode, width = None, ''
        for item in _split_commas(header):
            s = item.strip()
            if not s:
                continue
            mm = re.match(r'(input|output|inout)\b', s)
            rng = ''.join(re.findall(r'\[[^\]]*\]', s))
            if mm:
                mode = mm.group(1)
                width = rng  # new direction resets the width
            elif rng:
                width = rng  # continuation may restate a width
            if mode is None:
                continue
            names = _names_in(s)
            if names and names[-1] not in seen:
                seen.add(names[-1])
                ports.append((mode, names[-1], width))
    else:
        # Non-ANSI: direction declarations live in the body, one per ';'.
        for stmt in body.split(';'):
            mm = re.match(r'\s*(input|output|inout)\b', stmt)
            if not mm:
                continue
            mode = mm.group(1)
            width = ''.join(re.findall(r'\[[^\]]*\]', stmt))
            for nm in _names_in(stmt):
                if nm not in seen:
                    seen.add(nm)
                    ports.append((mode, nm, width))
    return ports


def extract_ports(verilog_code):
    """``(top_module_name, ports)`` for ``verilog_code``. ``ports`` is a list of
    ``(mode, name, width)`` where ``width`` is '' or a range like ``'[3:0]'``.
    The *top* module is chosen when several are present. Returns ``(None, [])``
    when no module is found."""
    names = find_modules(verilog_code)
    if not names:
        return None, []
    module_name = _pick_top(verilog_code, names)
    ports = _ports_hdlparse(verilog_code, module_name)
    if not ports:
        ports = _ports_regex(strip_comments(verilog_code), module_name)
    return module_name, ports


def order_modules(named_codes):
    """Dependency-order design units for display/serialisation: a unit appears
    before the units defining modules it instantiates (top-down).

    ``named_codes``: ``[(key, code), ...]`` -- ``key`` is the caller's id (a tab
    label). Every key is returned exactly once; independent units keep input
    order; dependency cycles terminate (each unit visited once) instead of
    recursing forever."""
    units = []
    for key, code in named_codes:
        units.append({
            'key': key,
            'clean': _strip_param_overrides(strip_comments(code)),
            'defines': set(find_modules(code)),
        })
    all_names = set()
    for u in units:
        all_names |= u['defines']

    name_to_units = {}
    for idx, u in enumerate(units):
        for nm in u['defines']:
            name_to_units.setdefault(nm, []).append(idx)

    deps = {i: set() for i in range(len(units))}
    for idx, u in enumerate(units):
        for nm in _instantiated_names(u['clean'], all_names - u['defines']):
            for dep_idx in name_to_units.get(nm, ()):
                if dep_idx != idx:
                    deps[idx].add(dep_idx)

    order, visited = [], set()

    def emit(i):
        if i in visited:
            return
        visited.add(i)
        order.append(i)                 # parent first ...
        for d in sorted(deps[i]):       # ... then what it instantiates
            emit(d)

    # Start only from top-level units (nobody instantiates them) so a child that
    # happens to come earlier in the input isn't emitted before its parent;
    # children are reached through their parent's recursion. Roots are taken in
    # input order, keeping independent units stable.
    has_parent = set().union(*deps.values()) if deps else set()
    for i in range(len(units)):
        if i not in has_parent:
            emit(i)
    # Anything left is inside a dependency cycle: emit in input order, no hang.
    for i in range(len(units)):
        emit(i)
    return [units[i]['key'] for i in order]


def _p3(p):
    """Normalise a port tuple to ``(mode, name, width)``."""
    return p if len(p) == 3 else (p[0], p[1], '')


def generate_stub_testbench(module_name, ports):
    """Ready-to-simulate testbench for ``module_name`` driving ``ports``
    (``(mode, name, width)`` tuples). Bus widths are honoured; ``clk``/``clock``
    gets a toggling stimulus; active-high (``rst``/``reset``) and active-low
    (``*_n``/``n*``) resets get a release pulse; ``$dumpfile``/``$dumpvars`` are
    wired for VCD capture."""
    ports = [_p3(p) for p in ports]
    inputs = [p for p in ports if p[0] == 'input']
    outputs = [p for p in ports if p[0] in ('output', 'inout')]

    def w(width):
        return (width + ' ') if width else ''

    regs_decl = "\n".join(f"  reg {w(wd)}{name};" for _, name, wd in inputs)
    wires_decl = "\n".join(f"  wire {w(wd)}{name};" for _, name, wd in outputs)
    names = [name for _, name, _ in ports]
    port_mapping = ", ".join(f".{name}({name})" for name in names)
    instance = (f"{module_name} uut (\n    {port_mapping}\n  );"
                if names else f"{module_name} uut ();")

    clk = next((name for _, name, _ in ports if name in ('clk', 'clock')), None)
    clk_stimulus = f"\n  always #5 {clk} = ~{clk};\n" if clk else ""

    init_stimulus = "\n  initial begin\n"
    if clk:
        init_stimulus += f"    {clk} = 0;\n"
    rst = next((name for _, name, _ in ports if name in ('reset', 'rst')), None)
    rst_n = next((name for _, name, _ in ports
                  if name in ('rst_n', 'reset_n', 'resetn', 'rstn', 'n_rst',
                              'nreset')), None)
    if rst:
        init_stimulus += f"    {rst} = 1;\n    #20 {rst} = 0;\n"
    elif rst_n:
        init_stimulus += f"    {rst_n} = 0;\n    #20 {rst_n} = 1;\n"

    init_stimulus += '    $dumpfile("sim_out.vcd");\n'
    init_stimulus += f'    $dumpvars(0, tb_{module_name});\n'
    init_stimulus += "    #500;\n"
    init_stimulus += "    $finish;\n"
    init_stimulus += "  end\n"

    return f"""`timescale 1ns/1ps

module tb_{module_name};
  // Inputs
{regs_decl}

  // Outputs
{wires_decl}

  // UUT Instance
  {instance}
{clk_stimulus}{init_stimulus}
endmodule
"""
