import re

def fix(fname, replacements):
    with open(fname, 'r') as f:
        c = f.read()
    for pattern, rep in replacements:
        c = re.sub(pattern, rep, c, flags=re.DOTALL)
    with open(fname, 'w') as f:
        f.write(c)

light_reps = [
    (r'QTabBar::close-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QTabBar::close-button {\n    image: url("images/dock_close_light.svg");'),
     
    (r'QTabBar::close-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QTabBar::close-button:hover {\n    background-color: #EFF6FF;\n    border-color: #93C5FD;\n    image: url("images/dock_close_light_hover.svg");'),
     
    (r'QDockWidget::float-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::float-button {\n    image: url("images/dock_fullscreen_light.svg");'),
     
    (r'QDockWidget::float-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::float-button:hover {\n    background-color: #EFF6FF;\n    border-color: #93C5FD;\n    image: url("images/dock_fullscreen_light_hover.svg");'),
     
    (r'QDockWidget::close-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::close-button {\n    margin-right: 6px;\n    image: url("images/dock_close_light.svg");'),
     
    (r'QDockWidget::close-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::close-button:hover {\n    image: url("images/dock_close_light_hover.svg");')
]

dark_reps = [
    (r'QTabBar::close-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QTabBar::close-button {\n    image: url("images/dock_close_dark.svg");'),
     
    (r'QTabBar::close-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QTabBar::close-button:hover {\n    background-color: rgba(241, 245, 249, 0.1);\n    border-color: #334155;\n    image: url("images/dock_close_dark_hover.svg");'),
     
    (r'QDockWidget::float-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::float-button {\n    image: url("images/dock_fullscreen_dark.svg");'),
     
    (r'QDockWidget::float-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::float-button:hover {\n    background-color: rgba(241, 245, 249, 0.1);\n    border-color: #334155;\n    image: url("images/dock_fullscreen_dark_hover.svg");'),
     
    (r'QDockWidget::close-button\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::close-button {\n    margin-right: 6px;\n    image: url("images/dock_close_dark.svg");'),
     
    (r'QDockWidget::close-button:hover\s*\{[^}]*?image:\s*url\("data:image/svg\+xml.*?"\);', 
     r'QDockWidget::close-button:hover {\n    image: url("images/dock_close_dark_hover.svg");')
]

fix('src/frontEnd/style_light.qss', light_reps)
fix('src/frontEnd/style_dark.qss', dark_reps)

