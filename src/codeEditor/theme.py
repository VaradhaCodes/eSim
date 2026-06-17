"""Visual theme for the eSim code editor.

One light, high-contrast scheme ("eSim Light") plus a single
``apply()`` that styles *any* QScintilla lexer by classifying each of
its styles from the human-readable ``description()`` string -- so the
SPICE, Verilog and VHDL lexers all get consistent, deliberate colours
instead of the near-black defaults.
"""

from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.Qsci import QsciScintilla


# ── palette ──────────────────────────────────────────────────────────
PAPER = "#FFFFFF"
TEXT = "#24292E"
COMMENT = "#6A9955"
KEYWORD = "#0033B3"
NUMBER = "#098658"
STRING = "#A31515"
VALUE = "#9A3FB6"
PARAMETER = "#0070C1"
FUNCTION = "#795E26"
PREPROC = "#AF00DB"
OPERATOR = "#6E7781"
INSTANCE = "#C2410C"
EXPRESSION = "#9A3FB6"
NODE = "#0E7490"

# chrome
MARGIN_FG = "#9DA5B4"
MARGIN_BG = "#F3F4F6"
MARGIN_SEP = "#DFE1E6"    # slightly darker stripe between numbers and fold
CARET_LINE = "#F5F8FF"
SELECTION = "#CFE3FB"
BRACE_FG = "#0033B3"
BRACE_BG = "#C8E6C9"
GUIDE = "#E6E8EB"
CARET = "#24292E"
SEARCH_HL = "#FBD56A"        # all-matches highlight
CURRENT_HL = "#FF9632"       # current match highlight

#: indicator numbers used for find highlighting (clear of lexer use)
SEARCH_INDICATOR = 8
CURRENT_INDICATOR = 9

_FONT_PREFS = [
    "Cascadia Code", "Cascadia Mono", "JetBrains Mono", "Fira Code",
    "Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono",
    "Noto Sans Mono", "Monospace",
]


def editor_font(size=11):
    """Best available monospace font."""
    available = set(QFontDatabase.families())
    family = next((f for f in _FONT_PREFS if f in available), "Monospace")
    font = QFont(family, size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def _classify(desc):
    """Map a lexer style description to (colour, bold, italic)."""
    d = desc.lower()
    if "comment" in d:
        return COMMENT, False, True
    if "instance" in d or "device" in d:
        return INSTANCE, True, False
    if "node" in d or "net" in d:
        return NODE, False, False
    if any(k in d for k in (
            "keyword", "command", "directive", "system task")):
        return KEYWORD, True, False
    if "preprocessor" in d or "macro" in d:
        return PREPROC, False, False
    if "expression" in d:
        return EXPRESSION, False, False
    if "number" in d:
        return NUMBER, False, False
    if "string" in d or "character" in d:
        return STRING, False, False
    if "value" in d:
        return VALUE, False, False
    if "parameter" in d:
        return PARAMETER, False, False
    if "function" in d or "task" in d:
        return FUNCTION, False, False
    if "operator" in d or "delimiter" in d:
        return OPERATOR, False, False
    return TEXT, False, False


def apply(editor, lexer, font=None):
    """Theme *editor* (a QsciScintilla) and its *lexer* (may be None)."""
    font = font or editor_font()
    paper = QColor(PAPER)

    if lexer is not None:
        lexer.setDefaultPaper(paper)
        lexer.setDefaultColor(QColor(TEXT))
        lexer.setDefaultFont(font)
        for style in range(128):
            desc = lexer.description(style)
            if not desc:
                continue
            colour, bold, italic = _classify(desc)
            lexer.setColor(QColor(colour), style)
            lexer.setPaper(paper, style)
            styled = QFont(font)
            styled.setBold(bold)
            styled.setItalic(italic)
            lexer.setFont(styled, style)

    _apply_chrome(editor, font)


def _apply_chrome(editor, font):
    editor.setMarginsFont(font)
    editor.setMarginsForegroundColor(QColor(MARGIN_FG))
    editor.setMarginsBackgroundColor(QColor(MARGIN_BG))
    # Margin 1 is the spacer between line numbers and the fold column.
    # Paint it slightly darker so it reads as a subtle separator stripe.
    # SCI_SETMARGINBACKN (2190): wParam=margin index, lParam=BBGGRR colour.
    sep = QColor(MARGIN_SEP)
    sep_packed = (sep.blue() << 16) | (sep.green() << 8) | sep.red()
    editor.SendScintilla(2190, 1, sep_packed)
    editor.setCaretLineBackgroundColor(QColor(CARET_LINE))
    editor.setCaretForegroundColor(QColor(CARET))
    editor.setCaretWidth(2)
    editor.setSelectionBackgroundColor(QColor(SELECTION))
    editor.resetSelectionForegroundColor()
    editor.setMatchedBraceForegroundColor(QColor(BRACE_FG))
    editor.setMatchedBraceBackgroundColor(QColor(BRACE_BG))
    editor.setIndentationGuidesForegroundColor(QColor(GUIDE))
    editor.setIndentationGuidesBackgroundColor(QColor(PAPER))
    editor.setFoldMarginColors(QColor(MARGIN_BG), QColor(MARGIN_BG))
    editor.setPaper(QColor(PAPER))
    editor.setColor(QColor(TEXT))
    # a touch of line spacing for readability
    editor.setExtraAscent(2)
    editor.setExtraDescent(2)
    # Search highlights are translucent boxes (StraightBox = fill +
    # outline) with a low fill alpha, so the matched glyphs stay
    # readable underneath -- a solid fill made single letters like "a"
    # vanish into the highlight.  The current match gets a stronger
    # fill and a solid outline so it still stands out from the rest.
    editor.indicatorDefine(
        QsciScintilla.IndicatorStyle.StraightBoxIndicator, SEARCH_INDICATOR)
    editor.setIndicatorForegroundColor(QColor(SEARCH_HL), SEARCH_INDICATOR)
    editor.SendScintilla(
        QsciScintilla.SCI_INDICSETALPHA, SEARCH_INDICATOR, 90)
    editor.SendScintilla(
        QsciScintilla.SCI_INDICSETOUTLINEALPHA, SEARCH_INDICATOR, 150)
    editor.indicatorDefine(
        QsciScintilla.IndicatorStyle.StraightBoxIndicator, CURRENT_INDICATOR)
    editor.setIndicatorForegroundColor(QColor(CURRENT_HL), CURRENT_INDICATOR)
    editor.SendScintilla(
        QsciScintilla.SCI_INDICSETALPHA, CURRENT_INDICATOR, 120)
    editor.SendScintilla(
        QsciScintilla.SCI_INDICSETOUTLINEALPHA, CURRENT_INDICATOR, 255)
