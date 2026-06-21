import re
with open("src/frontEnd/DockArea.py", "r") as f:
    c = f.read()

c = c.replace("        self.enable_tab_close_buttons()\n", "")

override = """
    def tabifyDockWidget(self, first, second):
        super().tabifyDockWidget(first, second)
        self.enable_tab_close_buttons()

    def enable_tab_close_buttons(self):"""

c = c.replace("    def enable_tab_close_buttons(self):", override)

with open("src/frontEnd/DockArea.py", "w") as f:
    f.write(c)

