import os

svgs = {
    # Light Mode (#6B7280)
    "text_save_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'></path><polyline points='17 21 17 13 7 13 7 21'></polyline><polyline points='7 3 7 8 15 8'></polyline></svg>",
    "text_save_as_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'></path><polyline points='17 21 17 13 7 13 7 21'></polyline><polyline points='7 3 7 8 15 8'></polyline><path d='M12 17h.01'></path></svg>",
    "text_find_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'></circle><line x1='21' y1='21' x2='16.65' y2='16.65'></line></svg>",
    "text_wrap_light.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#6B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='15 10 20 15 15 20'></polyline><path d='M4 4v7a4 4 0 0 0 4 4h12'></path></svg>",

    # Dark Mode (#CBD5E1)
    "text_save_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'></path><polyline points='17 21 17 13 7 13 7 21'></polyline><polyline points='7 3 7 8 15 8'></polyline></svg>",
    "text_save_as_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'></path><polyline points='17 21 17 13 7 13 7 21'></polyline><polyline points='7 3 7 8 15 8'></polyline><path d='M12 17h.01'></path></svg>",
    "text_find_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'></circle><line x1='21' y1='21' x2='16.65' y2='16.65'></line></svg>",
    "text_wrap_dark.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#CBD5E1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='15 10 20 15 15 20'></polyline><path d='M4 4v7a4 4 0 0 0 4 4h12'></path></svg>",
}

img_dir = "src/frontEnd/images"
os.makedirs(img_dir, exist_ok=True)
for name, content in svgs.items():
    with open(os.path.join(img_dir, name), "w") as f:
        f.write(content)

print("Text Editor SVGs Created")
