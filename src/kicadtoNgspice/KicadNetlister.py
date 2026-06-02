# ==============================================================================
#             FILE: KicadNetlister.py
#
#      DESCRIPTION: Generate an eSim-compatible spice netlist (<proj>.cir) from a
#                   KiCad schematic, independent of KiCad's own spice exporter.
#
#                   KiCad >= 7/8 `--format spice` only emits nodes for symbols
#                   that carry a simulation model; eSim's symbols do not, so that
#                   export degrades every component to "<ref> __<REF>" with no
#                   connectivity. The `orcadpcb2` netlist format, however, always
#                   carries full ref/value/pin->net data, so we generate that via
#                   kicad-cli and rewrite it into the flat spice form eSim's
#                   Processing expects:  "<ref> <net-per-pin-in-order> <value>".
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import re
import shutil
import subprocess

# A component header in orcadpcb2:  ( <uuid> <footprint> <refdes> <value>
_HDR_RE = re.compile(r'^\(\s+(/\S+)\s+(\S+)\s+(\S+)\s+(.+?)\s*$')
# A pin line:  ( <pin-number> <net-name> )
_PIN_RE = re.compile(r'^\(\s+(\d+)\s+(\S+)\s+\)\s*$')


def _sanitize_net(net):
    """Make a KiCad net name safe as an ngspice node (no parentheses/spaces)."""
    return net.replace('(', '').replace(')', '').replace(' ', '')


def orcad_to_spice_lines(orcad_text, title="KiCad schematic"):
    """Convert orcadpcb2 netlist text to eSim flat-spice component lines."""
    comps = []          # [(ref, value, [(pin, net), ...]), ...]
    cur = None
    for raw in orcad_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = _PIN_RE.match(s)
        if m and cur is not None:
            cur[2].append((int(m.group(1)), m.group(2)))
            continue
        m = _HDR_RE.match(s)
        if m:
            if cur is not None:
                comps.append(cur)
            cur = (m.group(3), m.group(4).strip(), [])
            continue
        if s == ')' and cur is not None:
            comps.append(cur)
            cur = None
    if cur is not None:
        comps.append(cur)

    lines = ['.title ' + title]
    for ref, value, pins in comps:
        pins.sort(key=lambda p: p[0])
        nets = ' '.join(_sanitize_net(n) for _, n in pins)
        lines.append((ref.lower() + ' ' + nets + ' ' + value).strip())
    lines.append('.end')
    return lines


def _kicad_cli():
    return os.environ.get('ESIM_KICAD_CLI') or shutil.which('kicad-cli')


def generate_netlist(proj_dir, proj_name):
    """Regenerate <proj>.cir from <proj>.kicad_sch via kicad-cli orcadpcb2.

    Returns (ok, message). Leaves any existing .cir untouched on failure so the
    legacy/manual workflow still applies.
    """
    sch = os.path.join(proj_dir, proj_name + '.kicad_sch')
    if not os.path.isfile(sch):
        return False, "No .kicad_sch found; using existing .cir if present."

    cli = _kicad_cli()
    if not cli:
        return False, "kicad-cli not found; using existing .cir if present."

    orcad_path = os.path.join(proj_dir, proj_name + '.orcad.tmp')
    try:
        proc = subprocess.run(
            [cli, 'sch', 'export', 'netlist', '--format', 'orcadpcb2',
             '-o', orcad_path, sch],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not os.path.isfile(orcad_path):
            return False, "kicad-cli netlist export failed: " + proc.stderr

        with open(orcad_path, 'r') as fh:
            orcad_text = fh.read()
        lines = orcad_to_spice_lines(orcad_text, title=proj_name)

        cir_path = os.path.join(proj_dir, proj_name + '.cir')
        with open(cir_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return True, "Generated " + cir_path + " from schematic (KiCad 8 safe)."
    except Exception as e:
        return False, "Netlist generation error: " + str(e)
    finally:
        if os.path.isfile(orcad_path):
            os.remove(orcad_path)


if __name__ == '__main__':
    import sys
    d, n = os.path.split(os.path.abspath(sys.argv[1]))
    print(generate_netlist(d, n.replace('.kicad_sch', '')))
