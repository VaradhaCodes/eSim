import re
import base64

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    def replacer(match):
        b64_str = match.group(1)
        decoded = base64.b64decode(b64_str).decode('utf-8')
        # REMOVE ALL NEWLINES AND CARRIAGE RETURNS
        cleaned = decoded.replace('\n', '').replace('\r', '')
        # re-encode
        encoded = base64.b64encode(cleaned.encode('utf-8')).decode('utf-8')
        return f'url("data:image/svg+xml;base64,{encoded}")'
        
    pattern = r'url\("data:image/svg\+xml;base64,([^\"]+)"\)'
    new_content = re.sub(pattern, replacer, content)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Fixed {filepath}")

fix_file("src/frontEnd/style_light.qss")
fix_file("src/frontEnd/style_dark.qss")
