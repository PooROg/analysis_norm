# Файл: main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from gui.interface import NormsAnalyzerGUI

def main():
    root = tk.Tk()
    app = NormsAnalyzerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()