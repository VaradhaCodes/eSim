# ==============================================================================
#  netlist_compare.py -- topology-aware equivalence check for two spice netlists.
#
#  Why not a text diff: the KiCad-9 netlister and the legacy KiCad-5/6 exporter
#  name auto-nets differently (`net__q1_c` vs `Net-(C1-Pad1)`), lowercase
#  differently, and emit components in a different order. None of that changes
#  the circuit. Two netlists are equivalent when they describe the same graph:
#
#    1. same set of component refs (case-insensitive: r1, q1, x1, ...),
#    2. same node count (arity) per ref,
#    3. same connectivity up to net renaming -- for every net, the set of
#       (ref, pin_position) it touches matches between the two netlists. Refs are
#       the anchors, so this is a direct set comparison, not a search.
#
#  Pin-order rules:
#    * Symmetric 2-terminal R/L/C: order is electrically irrelevant, so both pins
#      are recorded at position -1 (a terminal swap must not read as a failure).
#    * Every other device (D, Q, M, J, V, I, X subckt, U behavioural/plot): pin
#      position is significant -- this is what catches a real node-order error on
#      a transistor or subcircuit.
#    * Unconnected pins ("?" in legacy, "unconnected-..." in KiCad-9) are each a
#      distinct floating node; legacy lumps every "?" into one node, which would
#      otherwise fabricate connectivity.
#
#  Value/model string (eSim_NJF vs NJF) is reported but not part of the topology
#  verdict: the eSim DeviceModel step resolves the real model name downstream.
# ==============================================================================
SYMMETRIC = ("r", "l", "c")


def _is_floating(node):
    return node == "?" or node.startswith("unconnected")


def parse_cir(text):
    """Parse flat-spice component lines into (comps, canon).

    comps: {ref_lower: (arity, value_lower)}
    canon: frozenset of frozenset((ref, pin_position), ...)  -- one inner set per
           net, names discarded.
    """
    comps, net2pins, uid = {}, {}, [0]
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue
        t = s.split()
        if len(t) < 2:
            continue
        ref = t[0].lower()
        nodes = t[1:-1] if len(t) >= 3 else t[1:]
        value = t[-1].lower() if len(t) >= 3 else ""
        comps[ref] = (len(nodes), value)
        symmetric = ref[0] in SYMMETRIC and len(nodes) == 2
        for p, n in enumerate(nodes):
            nl = n.lower()
            if _is_floating(nl):
                key = "__float%d" % uid[0]
                uid[0] += 1
            else:
                key = nl
            pos = -1 if symmetric else p
            net2pins.setdefault(key, set()).add((ref, pos))
    canon = frozenset(frozenset(v) for v in net2pins.values())
    return comps, canon


def compare(generated_text, legacy_text):
    """Return a dict describing the comparison; `equivalent` is the verdict."""
    gc, gcanon = parse_cir(generated_text)
    lc, lcanon = parse_cir(legacy_text)
    refs_ok = set(gc) == set(lc)
    arity_ok = all(gc.get(r, (None,))[0] == lc.get(r, (None,))[0]
                   for r in set(gc) | set(lc))
    topo_ok = gcanon == lcanon
    value_ok = all(gc.get(r, (0, ""))[1] == lc.get(r, (0, ""))[1]
                   for r in set(gc) & set(lc))
    return {
        "equivalent": refs_ok and arity_ok and topo_ok,
        "refs_ok": refs_ok,
        "arity_ok": arity_ok,
        "topo_ok": topo_ok,
        "value_ok": value_ok,                 # informational only
        "missing_refs": sorted(set(lc) - set(gc)),
        "extra_refs": sorted(set(gc) - set(lc)),
    }
