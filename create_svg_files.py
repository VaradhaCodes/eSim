import os
import re

svgs = {
    # Light Mode
    "dock_fullscreen_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='4 14 4 20 10 20'/><polyline points='20 10 20 4 14 4'/></svg>",
    "dock_fullscreen_light_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#165982' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='4 14 4 20 10 20'/><polyline points='20 10 20 4 14 4'/></svg>",
    "dock_pop_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 14h6v6'/><path d='M20 10h-6V4'/><path d='M14 10l7-7'/><path d='M3 21l7-7'/></svg>",
    "dock_pop_light_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#165982' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 14h6v6'/><path d='M20 10h-6V4'/><path d='M14 10l7-7'/><path d='M3 21l7-7'/></svg>",
    "dock_close_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2.4' stroke-linecap='round'><line x1='6' y1='6' x2='18' y2='18'/><line x1='18' y1='6' x2='6' y2='18'/></svg>",
    "dock_close_light_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#DC2626' stroke-width='2.4' stroke-linecap='round'><line x1='6' y1='6' x2='18' y2='18'/><line x1='18' y1='6' x2='6' y2='18'/></svg>",

    # Dark Mode
    "dock_fullscreen_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='4 14 4 20 10 20'/><polyline points='20 10 20 4 14 4'/></svg>",
    "dock_fullscreen_dark_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#F1F5F9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='4 14 4 20 10 20'/><polyline points='20 10 20 4 14 4'/></svg>",
    "dock_pop_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 14h6v6'/><path d='M20 10h-6V4'/><path d='M14 10l7-7'/><path d='M3 21l7-7'/></svg>",
    "dock_pop_dark_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#F1F5F9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 14h6v6'/><path d='M20 10h-6V4'/><path d='M14 10l7-7'/><path d='M3 21l7-7'/></svg>",
    "dock_close_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2.4' stroke-linecap='round'><line x1='6' y1='6' x2='18' y2='18'/><line x1='18' y1='6' x2='6' y2='18'/></svg>",
    "dock_close_dark_hover.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#EF4444' stroke-width='2.4' stroke-linecap='round'><line x1='6' y1='6' x2='18' y2='18'/><line x1='18' y1='6' x2='6' y2='18'/></svg>"
}

img_dir = "src/frontEnd/images"
os.makedirs(img_dir, exist_ok=True)

for name, content in svgs.items():
    with open(os.path.join(img_dir, name), "w") as f:
        f.write(content)
print("Saved SVGs to disk.")

