# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from gui.interface import NormsAnalyzerGUI

def main():
    r = tk.Tk()
    a = NormsAnalyzerGUI(r)
    r.protocol("WM_DELETE_WINDOW", a.on_closing)
    r.mainloop()

if __name__ == "__main__":
    main()