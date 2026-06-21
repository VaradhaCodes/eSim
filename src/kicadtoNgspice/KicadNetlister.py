# ==============================================================================
#             FILE: KicadNetlister.py
#
#      DESCRIPTION: Generate an eSim-compatible spice netlist (<proj>.cir) from a
#                   KiCad schematic, independent of KiCad's own spice exporter.
#
#                   KiCad >= 7/8 `--format spice` only emits nodes for symbols
#                   that carry a simulation model; eSim's symbols do not, so that
#                   export degrades every component to "<ref> __<REF>" with no
#                   connectivity. We instead use the `kicadxml` netlist, which
#                   lists every net (including single-node nets such as plot
#                   markers) with full ref/pin->net data, and rewrite it into the
#                   flat spice form eSim's Processing expects:
#                       "<ref> <net-per-pin-in-order> <value>"
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET


def _sanitize_net(net):
    """Make a KiCad net name safe + consistent as an ngspice node.

    Strips parentheses/spaces and lowercases (ngspice is case-insensitive; the
    historical eSim netlist used lowercase, e.g. 'gnd', 'net-_u1-pad1_').
    """
    return net.replace('(', '').replace(')', '').replace(' ', '').lower()


def xml_to_spice_lines(xml_path, title="KiCad schematic"):
    """Convert a KiCad `kicadxml` netlist into eSim flat-spice component lines."""
    root = ET.parse(xml_path).getroot()

    # Component reference -> value, preserving document order.
    order_refs = []
    values = {}
    for comp in root.iter('comp'):
        ref = comp.get('ref')
        order_refs.append(ref)
        v = comp.find('value')
        values[ref] = (v.text or '').strip() if v is not None else ''

    # Component reference -> [(pin_order, net_name), ...]
    pins = {}
    for net in root.iter('net'):
        raw = net.get('name') or ('net' + (net.get('code') or '0'))
        net_clean = _sanitize_net(raw)
        for node in net.findall('node'):
            ref = node.get('ref')
            pin = node.get('pin') or ''
            order = int(pin) if pin.isdigit() else 0
            pins.setdefault(ref, []).append((order, net_clean))

    lines = ['.title ' + title]
    for ref in order_refs:
        nodelist = sorted(pins.get(ref, []), key=lambda t: t[0])
        nets = ' '.join(n for _, n in nodelist)
        lines.append((ref.lower() + ' ' + nets + ' ' + values.get(ref, '')).strip())
    lines.append('.end')
    return lines


def _kicad_cli():
    return os.environ.get('ESIM_KICAD_CLI') or shutil.which('kicad-cli')


def generate_netlist(proj_dir, proj_name):
    """Regenerate <proj>.cir from <proj>.kicad_sch via kicad-cli kicadxml.

    Returns (ok, message). Leaves any existing .cir untouched on failure so the
    legacy/manual workflow still applies.
    """
    sch = os.path.join(proj_dir, proj_name + '.kicad_sch')
    if not os.path.isfile(sch):
        return False, "No .kicad_sch found; using existing .cir if present."

    cli = _kicad_cli()
    if not cli:
        return False, "kicad-cli not found; using existing .cir if present."

    xml_path = os.path.join(proj_dir, proj_name + '.netlist.xml')
    try:
        proc = subprocess.run(
            [cli, 'sch', 'export', 'netlist', '--format', 'kicadxml',
             '-o', xml_path, sch],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not os.path.isfile(xml_path):
            return False, "kicad-cli netlist export failed: " + proc.stderr

        lines = xml_to_spice_lines(xml_path, title=proj_name)
        cir_path = os.path.join(proj_dir, proj_name + '.cir')
        with open(cir_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return True, "Generated " + cir_path + " from schematic (KiCad 8 safe)."
    except Exception as e:
        return False, "Netlist generation error: " + str(e)
    finally:
        if os.path.isfile(xml_path):
            os.remove(xml_path)


if __name__ == '__main__':
    import sys
    d, n = os.path.split(os.path.abspath(sys.argv[1]))
    print(generate_netlist(d, n.replace('.kicad_sch', '')))
