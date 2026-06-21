import re

with open('src/frontEnd/style_light.qss', 'r') as f:
    css = f.read()

replacements = {
    '#FAFBFC': '#111827',  # Background
    '#FFFFFF': '#1F2937',  # Widgets background
    '#F3F4F6': '#374151',  # Tabs/progress
    '#EFF6FF': '#374151',  # Hover bg
    '#D1D5DB': '#4B5563',  # Border
    '#E5E7EB': '#4B5563',  # Hover/separators
    '#1F2937': '#F9FAFB',  # Main text
    '#6B7280': '#9CA3AF',  # Muted text
    '#9CA3AF': '#6B7280',  # Scrollbar handle
    '#165982': '#3B82F6',  # Primary blue
    '#0E324B': '#1D4ED8',  # Darker blue
    '#1E6FA0': '#60A5FA',  # Hover blue
}

# Also update the comment section
css = css.replace("eSim teal", "Bright blue")
css = css.replace("deep navy", "Dark blue")
css = css.replace("off-white background", "Dark background")
css = css.replace("near-black body text", "White text")
css = css.replace("soft gray border", "Dark gray border")

for old, new in replacements.items():
    # Case insensitive replace for hex codes
    css = re.sub(old, new, css, flags=re.IGNORECASE)

with open('src/frontEnd/style_dark.qss', 'w') as f:
    f.write(css)
