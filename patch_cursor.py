import re

with open('src/ngspiceSimulation/_cursor_mixin.py', 'r') as f:
    code = f.read()

# Add get colors line at the start of methods
for func in ['def _update_cursor_position', 'def _update_cursor_panel', 'def _refresh_cursor_readouts']:
    code = code.replace(
        func + "(self",
        func + "(self"
    )

# Replace #333, #555, #999, #aaa with variables
# But wait, we can just replace them inside the f-strings

# Since we don't want to mess up, let's just do targeted replacements:
code = code.replace('#333', '{fg_main}')
code = code.replace('#555', '{fg_sec}')
code = code.replace('#999', '{fg_mut}')
code = code.replace('#aaa', '{fg_mut}')

# Now we need to inject the variable assignments right after the method signatures.
# We will just inject it by finding the def lines.

def inject_colors(match):
    indent = match.group(1)
    return match.group(0) + f"\n{indent}fg_main, fg_sec, fg_mut = self._get_text_colors()"

code = re.sub(r'(\s+)def _update_cursor_position\(.*?\)\s*->\s*None:', inject_colors, code)
code = re.sub(r'(\s+)def _update_cursor_panel\(.*?\)\s*->\s*None:', inject_colors, code)
code = re.sub(r'(\s+)def _refresh_cursor_readouts\(.*?\)\s*->\s*None:', inject_colors, code)

with open('src/ngspiceSimulation/_cursor_mixin.py', 'w') as f:
    f.write(code)
