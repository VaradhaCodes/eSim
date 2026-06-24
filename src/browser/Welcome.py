from PyQt6 import QtCore, QtWidgets
import os


class Welcome(QtWidgets.QWidget):
    """
    It contains class responsible for content of dock area part of initial esim Window.
    It creates Welcome page of eSim as shown below in image. The library/browser/welcome.html file is used for html content.
    """

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setObjectName("welcomeCard")
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.browser = QtWidgets.QTextBrowser()
        # Tagged + frameless so the Aurora theme palette drives the card
        # surface/text instead of a hard-coded light page.
        self.browser.setObjectName("welcomeScroll")
        self.browser.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        self.browser.setSource(QtCore.QUrl(
            init_path + "library/browser/welcome.html")
        )
        self.browser.setOpenExternalLinks(True)
        self.browser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.vlayout.addWidget(self.browser)
        self.setLayout(self.vlayout)
        self.show()
