import sys
import os
import PySide2
# 解决PySide2和其他QT环境的冲突
qt_path = os.path.dirname(PySide2.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(qt_path, "plugins")

from PySide2.QtWidgets import QApplication
# from app import myGUI
from app_gui import myGUI


def main():
    app = QApplication(sys.argv)
    GUI = myGUI()  
    GUI.ui.show()
    sys.exit(app.exec_())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

