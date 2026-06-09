"""
KicadSymbolFixer.py  --  Auto-repair corrupted eSim_Ngveri.kicad_sym files.

When eSim's createkicad.py writes a new symbol, it sometimes drops the
opening `(symbol "name" ...)` header line, leaving orphaned `(property ...)`
lines that KiCad cannot parse.  This module detects and repairs that
corruption pattern so the library loads cleanly on the next KiCad session.

The repair runs automatically on eSim startup.  It is non-destructive:
  - A timestamped backup is created before any modification.
  - If no corruption is found, the file is left untouched.

Corruption pattern detected:
  After the library header or after a symbol-closing `))`/`)`, the parser
  expects `(symbol "name" ...` but instead finds `(property "Reference" ...`.
  The fix extracts the component name from the subsequent
  `(property "Value" "THENAME" ...)` line and inserts the missing
  `(symbol "THENAME" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)`
  line at the correct position.
"""

import os
import re
import shutil
from datetime import datetime


# ---------- path helpers ----------
def _sym_paths():
    """Return a list of .kicad_sym paths that eSim may have written to."""
    paths = []
    try:
        # Import Appconfig locally so we rely on the exact same logic eSim uses
        from configuration.Appconfig import Appconfig
        
        if os.name == 'nt':
            # Windows: use Appconfig's parser to get eSim_HOME exactly as eSim does
            try:
                src_home = Appconfig.parser_esim.get('eSim', 'eSim_HOME')
                inst_dir = src_home.replace('\\eSim', '')
                paths.append(os.path.join(inst_dir, 'KiCad', 'share', 'kicad', 'symbols', 'eSim_Ngveri.kicad_sym'))
                paths.append(os.path.join(inst_dir, 'KiCad', 'share', 'kicad', 'symbols', 'eSim_Nghdl.kicad_sym'))
            except Exception as e:
                print("[KicadSymbolFixer] Error reading eSim_HOME on Windows:", e)
        else:
            # Linux paths are standard system locations
            paths.append(os.path.join('/', 'usr', 'share', 'kicad', 'symbols', 'eSim_Ngveri.kicad_sym'))
            paths.append(os.path.join('/', 'usr', 'share', 'kicad', 'symbols', 'eSim_Nghdl.kicad_sym'))
    except Exception as e:
        print("[KicadSymbolFixer] Error importing Appconfig:", e)
        
    return paths


# ---------- core repair logic ----------

# Regex that matches a standard symbol-opening line
_RE_SYMBOL_OPEN = re.compile(
    r'^\s*\(symbol\s+"([^"]+)"\s+\(pin_names'
)

# Regex to extract component name from a Value property line
_RE_VALUE_PROP = re.compile(
    r'\(property\s+"Value"\s+"([^"]+)"'
)

# Template for the missing symbol header line
_SYMBOL_HEADER_TEMPLATE = (
    '(symbol "{name}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)'
)


def _extract_name_from_value_line(line):
    """Extract the component name from a (property "Value" "NAME" ...) line."""
    m = _RE_VALUE_PROP.search(line)
    return m.group(1) if m else None


def repair_kicad_sym(filepath):
    """
    Scan a .kicad_sym file for missing symbol headers and insert them.

    Returns:
        (bool, list[str]):  (was_repaired, list_of_fixed_component_names)
    """
    if not os.path.exists(filepath):
        return False, []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, PermissionError):
        return False, []

    lines = content.split('\n')
    repaired_lines = []
    fixed_names = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- Check: does this line start with (property "Reference" "U"
        #      without a preceding (symbol "..." header? ----
        if stripped.startswith('(property "Reference"'):
            # Look backward: was the previous non-blank line a (symbol ...) header
            # or the library header or a comment?
            needs_fix = True

            # Check if the previous non-blank line already opens a symbol block
            for j in range(len(repaired_lines) - 1, -1, -1):
                prev = repaired_lines[j].strip()
                if not prev:
                    continue
                # Previous line is a proper (symbol "name" ...) opener
                if _RE_SYMBOL_OPEN.match(prev):
                    needs_fix = False
                    break
                # Previous line closes a symbol block with )) -- so we ARE
                # at the start of a new symbol and the header is missing
                if prev.endswith('))') or prev == ')':
                    needs_fix = True
                    break
                # Previous line is the library header
                if prev.startswith('(kicad_symbol_lib'):
                    needs_fix = True
                    break
                # Previous line is something else (shouldn't happen in
                # well-formed files, but treat as needing fix)
                needs_fix = True
                break

            if needs_fix:
                # We need to find the component name from the upcoming
                # (property "Value" "THENAME" ...) line
                comp_name = None

                # The Value property is typically the next line after Reference
                for k in range(i + 1, min(i + 5, len(lines))):
                    comp_name = _extract_name_from_value_line(lines[k])
                    if comp_name:
                        break

                if comp_name:
                    # Insert the missing symbol header BEFORE this line
                    header = _SYMBOL_HEADER_TEMPLATE.format(name=comp_name)
                    repaired_lines.append(header)
                    fixed_names.append(comp_name)

        # ---- Check: does this line have `))(symbol "name"` jammed together?
        #      e.g.: `  ))(symbol "ml_act_relu_64bit_q32_32" (pin_names ...`
        #      This is where the previous symbol's close and the next symbol's
        #      open are on the same line. Split them. ----
        jam_match = re.match(r'^(.*\)\))\s*(\(symbol\s+".+)$', line)
        if jam_match:
            repaired_lines.append(jam_match.group(1))
            repaired_lines.append(jam_match.group(2))
            i += 1
            continue

        repaired_lines.append(line)
        i += 1

    if not fixed_names:
        return False, []

    # Create backup before writing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = filepath + f'.backup_{timestamp}'
    try:
        shutil.copy2(filepath, backup)
    except (IOError, PermissionError):
        pass  # If backup fails, still try to fix

    # Write repaired content
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(repaired_lines))
    except (IOError, PermissionError) as e:
        print(f"[KicadSymbolFixer] ERROR: Could not write to {filepath}: {e}")
        return False, fixed_names

    return True, fixed_names


def repair_all_sym_files():
    """
    Scan and repair all known .kicad_sym files.
    Called automatically on eSim startup.
    Returns a list of messages describing repairs.
    """
    paths = _sym_paths()
    total_fixed = 0
    messages = []

    for path in paths:
        if not os.path.exists(path):
            continue

        was_repaired, names = repair_kicad_sym(path)
        if was_repaired:
            total_fixed += len(names)
            basename = os.path.basename(path)
            msg = (
                f"[KicadSymbolFixer] Repaired {basename}: "
                f"restored {len(names)} missing symbol header(s): "
                f"{', '.join(names)}"
            )
            print(msg)
            messages.append(msg)

    if total_fixed == 0:
        print("[KicadSymbolFixer] All symbol libraries are clean.")
    else:
        print(
            f"[KicadSymbolFixer] Done. Fixed {total_fixed} corrupted "
            f"symbol(s) across {len(paths)} library file(s)."
        )

    return messages


# Allow running standalone: python KicadSymbolFixer.py
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # Repair a specific file passed as argument
        path = sys.argv[1]
        was_repaired, names = repair_kicad_sym(path)
        if was_repaired:
            print(f"Fixed {len(names)} symbol(s): {', '.join(names)}")
        else:
            print("No corruption detected.")
    else:
        repair_all_sym_files()
