import re
import base64

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    def replacer(match):
        b64_str = match.group(1)
        decoded = base64.b64decode(b64_str).decode('utf-8')
        
        # We need utf8 format. 
        # Escape # to %23, and remove newlines.
        cleaned = decoded.replace('\n', '').replace('\r', '').replace('#', '%23')
        
        return f'url("data:image/svg+xml;utf8,{cleaned}")'
        
    pattern = r'url\("data:image/svg\+xml;base64,([^\"]+)"\)'
    new_content = re.sub(pattern, replacer, content)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Fixed {filepath}")

fix_file("src/frontEnd/style_light.qss")
fix_file("src/frontEnd/style_dark.qss")
