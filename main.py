# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from gui.interface import NormsAnalyzerGUI

def main():
    r = tk.Tk()
    a = NormsAnalyzerGUI(r)
    r.mainloop()

if __name__ == "__main__":
    main()