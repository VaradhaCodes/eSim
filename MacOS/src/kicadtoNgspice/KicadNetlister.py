# ==============================================================================
#             FILE: KicadNetlister.py
#
#      DESCRIPTION: Generate an eSim-compatible spice netlist (<proj>.cir) from a
#                   KiCad schematic, independent of KiCad's own spice exporter.
#
#                   KiCad >= 7 rewrote `--format spice` around its Simulation
#                   Model system; symbols without a Sim.* model (every eSim
#                   symbol: plots, behavioural u-blocks, sources, custom models)
#                   are exported with their connectivity stripped, e.g.
#                   "U2 __U2" / "v3 __v3" (one placeholder node, no nets).
#                   That is unrecoverable, so we never use `--format spice`.
#
#                   Instead we use `--format kicadxml`, the generic netlist that
#                   always lists every comp + every net + pin->net mapping
#                   regardless of simulation models, and rewrite it into the flat
#                   spice form eSim's Processing expects:
#                       "<ref> <net-per-pin-in-node-order> <value>"
#                   Node order is pin-number order (eSim symbols number pins in
#                   spice node order); an optional `Spice_Node_Sequence` user
#                   field reorders when present. `Spice_Netlist_Enabled=N` drops
#                   a component.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET


def _kicad_cli():
    """Locate kicad-cli (KiCad >= 7). Env override for flatpak/appimage."""
    return os.environ.get('ESIM_KICAD_CLI') or shutil.which('kicad-cli')


def _sanitize_net(name):
    """Make a KiCad net name a safe, consistent ngspice node.

    Strips parentheses/spaces and lowercases (ngspice is case-insensitive; the
    historical eSim netlist used lowercase). Topology is preserved; exact net
    spelling differs from the KiCad 5/6 era and does not affect simulation.
    """
    return name.replace('(', '').replace(')', '').replace(' ', '').lower()


def _node_sort_key(pin, seen_index):
    """Order a component's pins: numeric pin numbers first (ascending), then any
    non-numeric/blank pins (e.g. eSim plot markers expose a blank pin) in the
    order encountered, so single-pin parts and oddities never crash."""
    if pin.isdigit():
        return (0, int(pin), seen_index)
    return (1, 0, seen_index)


def _apply_node_sequence(nodes, seq_field):
    """Reorder `nodes` (already in default pin order) by a Spice_Node_Sequence
    field: a comma/space list of 0-based indices, e.g. '2,1,0'. Ignored if it
    is malformed or not a permutation of range(len(nodes))."""
    tokens = seq_field.replace(',', ' ').split()
    try:
        order = [int(t) for t in tokens]
    except ValueError:
        return nodes
    if sorted(order) != list(range(len(nodes))):
        return nodes
    return [nodes[i] for i in order]


def xml_to_spice_lines(xml_path, title="KiCad schematic"):
    """Convert a KiCad `kicadxml` netlist into eSim flat-spice component lines."""
    root = ET.parse(xml_path).getroot()

    # Components in document order: ref -> value, ref -> {field name: value}
    order_refs = []
    value = {}
    fields = {}
    comps_el = root.find('components')
    for comp in (comps_el.findall('comp') if comps_el is not None else []):
        ref = comp.get('ref')
        order_refs.append(ref)
        v = comp.find('value')
        value[ref] = (v.text or '').strip() if v is not None else ''
        fd = {}
        fel = comp.find('fields')
        if fel is not None:
            for f in fel.findall('field'):
                fd[f.get('name')] = (f.text or '').strip()
        fields[ref] = fd

    # Connectivity: ref -> [(sort_key, net_name), ...]
    pins = {}
    seen = {}
    nets_el = root.find('nets')
    for net in (nets_el.findall('net') if nets_el is not None else []):
        raw = net.get('name') or ('net' + (net.get('code') or '0'))
        net_clean = _sanitize_net(raw)
        for node in net.findall('node'):
            ref = node.get('ref')
            pin = node.get('pin') or ''
            idx = seen.get(ref, 0)
            seen[ref] = idx + 1
            pins.setdefault(ref, []).append((_node_sort_key(pin, idx), net_clean))

    lines = ['* ' + title + ' (eSim netlist via kicad-cli kicadxml)',
             '* Sheet Name: /']
    for ref in order_refs:
        fd = fields.get(ref, {})
        if fd.get('Spice_Netlist_Enabled', '').strip().lower() == 'n':
            continue
        ordered = [n for _, n in sorted(pins.get(ref, []), key=lambda t: t[0])]
        seq = fd.get('Spice_Node_Sequence', '').strip()
        if seq:
            ordered = _apply_node_sequence(ordered, seq)
        nets = ' '.join(ordered)
        lines.append((ref.lower() + ' ' + nets + ' ' + value.get(ref, '')).strip())
    lines.append('.end')
    return lines


def generate_netlist(proj_dir, proj_name):
    """Regenerate <proj>.cir from <proj>.kicad_sch via kicad-cli kicadxml.

    Returns (ok, message). Leaves any existing .cir untouched on failure so the
    legacy/manual workflow still applies on KiCad < 7 or odd installs.
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
            return False, "kicad-cli netlist export failed: " + proc.stderr.strip()

        lines = xml_to_spice_lines(xml_path, title=proj_name)
        cir_path = os.path.join(proj_dir, proj_name + '.cir')
        with open(cir_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return True, "Generated " + cir_path + " from schematic (KiCad 7-10 safe)."
    except Exception as e:
        return False, "Netlist generation error: " + str(e)
    finally:
        if os.path.isfile(xml_path):
            os.remove(xml_path)


if __name__ == '__main__':
    import sys
    d, n = os.path.split(os.path.abspath(sys.argv[1]))
    name = n[:-len('.kicad_sch')] if n.endswith('.kicad_sch') else n
    ok, msg = generate_netlist(d, name)
    print(msg)
    if ok:
        with open(os.path.join(d, name + '.cir')) as fh:
            print(fh.read())
