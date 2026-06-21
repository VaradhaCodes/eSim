import re
import base64
import urllib.parse

def encode_svg(match):
    svg_content = match.group(1)
    # The svg_content might have %23 instead of #, so unquote it first
    svg_unquoted = urllib.parse.unquote(svg_content)
    # Ensure it's valid XML by fixing missing closing tags if any still exist
    if svg_unquoted.endswith("</svg>") and not svg_unquoted.endswith("/></svg>"):
        svg_unquoted = svg_unquoted.replace("</svg>", "></svg>") # just in case, though we fixed it
    encoded = base64.b64encode(svg_unquoted.encode('utf-8')).decode('utf-8')
    return f'url("data:image/svg+xml;base64,{encoded}")'

for qss_file in ["src/frontEnd/style_light.qss", "src/frontEnd/style_dark.qss"]:
    with open(qss_file, "r") as f:
        content = f.read()
    
    # regex to find url("data:image/svg+xml;utf8,<svg ...</svg>")
    pattern = r'url\("data:image/svg\+xml;utf8,(<svg[^>]*>.*?</svg>)"\)'
    new_content = re.sub(pattern, encode_svg, content, flags=re.DOTALL)
    
    with open(qss_file, "w") as f:
        f.write(new_content)
    
    print(f"Fixed SVGs in {qss_file}")
