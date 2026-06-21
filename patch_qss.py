import re

def patch_light():
    with open('src/frontEnd/style_light.qss', 'r') as f:
        c = f.read()
    
    # dockPopButton
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"] {\n    \1\n    qproperty-icon: url("images/dock_fullscreen_light.svg");', c, flags=re.DOTALL)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]:hover\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"]:hover {\n    \1\n    qproperty-icon: url("images/dock_fullscreen_light_hover.svg");', c, flags=re.DOTALL)
               
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isPoppedOut="true"\]\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isPoppedOut="true"] {\n    qproperty-icon: url("images/dock_pop_light.svg");', c)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isPoppedOut="true"\]:hover\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isPoppedOut="true"]:hover {\n    qproperty-icon: url("images/dock_pop_light_hover.svg");', c)

    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isCloseBtn="true"\]\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isCloseBtn="true"] {\n    qproperty-icon: url("images/dock_close_light.svg");', c)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isCloseBtn="true"\]:hover\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isCloseBtn="true"]:hover {\n    \1\n    qproperty-icon: url("images/dock_close_light_hover.svg");', c, flags=re.DOTALL)

    c = re.sub(r'QTabBar::close-button\s*\{\s*image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QTabBar::close-button {\n    image: url("images/dock_close_light.svg");', c)
    c = re.sub(r'QTabBar::close-button:hover\s*\{\s*image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QTabBar::close-button:hover {\n    image: url("images/dock_close_light_hover.svg");', c)
               
    with open('src/frontEnd/style_light.qss', 'w') as f:
        f.write(c)

def patch_dark():
    with open('src/frontEnd/style_dark.qss', 'r') as f:
        c = f.read()
    
    # dockPopButton
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"] {\n    \1\n    qproperty-icon: url("images/dock_fullscreen_dark.svg");', c, flags=re.DOTALL)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]:hover\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"]:hover {\n    \1\n    qproperty-icon: url("images/dock_fullscreen_dark_hover.svg");', c, flags=re.DOTALL)
               
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isPoppedOut="true"\]\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isPoppedOut="true"] {\n    qproperty-icon: url("images/dock_pop_dark.svg");', c)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isPoppedOut="true"\]:hover\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isPoppedOut="true"]:hover {\n    qproperty-icon: url("images/dock_pop_dark_hover.svg");', c)

    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isCloseBtn="true"\]\s*\{\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isCloseBtn="true"] {\n    qproperty-icon: url("images/dock_close_dark.svg");', c)
    c = re.sub(r'QPushButton\[dockPopButton="true"\]\[isCloseBtn="true"\]:hover\s*\{\s*(.*?)\s*qproperty-icon:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QPushButton[dockPopButton="true"][isCloseBtn="true"]:hover {\n    \1\n    qproperty-icon: url("images/dock_close_dark_hover.svg");', c, flags=re.DOTALL)

    c = re.sub(r'QTabBar::close-button\s*\{\s*margin-right:.*?image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QTabBar::close-button {\n    margin-right: 4px;\n    image: url("images/dock_close_dark.svg");', c, flags=re.DOTALL)
    c = re.sub(r'QTabBar::close-button:hover\s*\{\s*image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QTabBar::close-button:hover {\n    image: url("images/dock_close_dark_hover.svg");', c)

    c = re.sub(r'QDockWidget::close-button\s*\{\s*margin-right:.*?image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QDockWidget::close-button {\n    margin-right: 6px;\n    image: url("images/dock_close_dark.svg");', c, flags=re.DOTALL)
    c = re.sub(r'QDockWidget::close-button:hover\s*\{\s*image:\s*url\("data:image/svg\+xml;utf8,.*?"\);',
               r'QDockWidget::close-button:hover {\n    image: url("images/dock_close_dark_hover.svg");', c)
               
    with open('src/frontEnd/style_dark.qss', 'w') as f:
        f.write(c)

patch_light()
patch_dark()
print("Done")
