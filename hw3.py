import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from som import SOM
import seaborn as sns

# 定義全域變數
cwd = os.getcwd()
file_base = ''

def converttype(content): # 轉換.txt檔案資料
    for c in range(len(content)):
        content[c] = float(content[c])
    return content

def draw_weight(weights, classifier, x, e): # 畫出拓樸與輸入資料的關係
    fig1 = Figure(figsize=(4,4))
    fig2 = Figure(figsize=(4,4))
    if weights.shape[2] > 2:
        if file_base == 'Number.txt':
            convertw = []
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    convertw.append(np.reshape(weights[i][j], (-1, 5)))
            convertw = np.array(convertw)
            for i in range(1,weights.shape[0]*weights.shape[1]+1):
                ax1 = fig1.add_subplot(weights.shape[0],weights.shape[1],i)
                sns.heatmap(convertw[i-1],
                            cmap='OrRd',
                            cbar=False,
                            vmax=1,
                            ax = ax1)
        else:
            ax1 = fig1.add_subplot(111,projection='3d')
            ax2 = fig2.add_subplot(111,projection='3d')
            ax1.title.set_text('3D Topology With Data')
            ax2.title.set_text('3D Topology')
            ax1.scatter(x[:,0], x[:,1], x[:,2], s=50, c=e[:,:], marker="*")
            ax1.scatter(weights[:,:,0], weights[:,:,1],weights[:,:,2], s=5, c=classifier, cmap='rainbow')
            ax2.scatter(weights[:,:,0], weights[:,:,1],weights[:,:,2], s=5, c=classifier, cmap='rainbow')
    else:
        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        ax1.title.set_text('2D Topology With Data')
        ax2.title.set_text('2D Topology')
        ax1.scatter(x[:,0], x[:,1], s=50, c=e[:,:], marker="*")
        ax1.scatter(weights[:,:,0], weights[:,:,1], s=5, c=classifier, cmap='rainbow')
        ax2.scatter(weights[:,:,0], weights[:,:,1], s=5, c=classifier, cmap='rainbow')
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=5,column=1,columnspan=5)  
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=5,column=7,columnspan=5)  

def _readfile(): # 讀取檔案
    global data,file_base
    filename = filedialog.askopenfilename()
    f = open(filename,mode='r')
    file = f.read().split('\n')
    if type(file_base) == str:
        fileentry.delete(0, 'end')
    file_base=os.path.basename(filename)
    fileentry.insert(0, file_base)
    data = np.array([converttype(fc.split(' ')) for fc in file if fc != ''])
    f.close()

def _training(): # 開始訓練模型
    ep = epoch.get()
    som = SOM(data, ep)
    classifier = som.train()
    x = som.inputdata
    e = som.eoutputdata
    weights = som.w
    draw_weight(weights, classifier, x, e)
    
def _quit(): # 結束程式
    root.quit()
    root.destroy()

# UI interface
root = tk.Tk()
root.title("Test")
root.geometry("800x600")
file_base = tk.StringVar()
epoch = tk.IntVar()

f = tkFont.Font(family='Ink Free')
tk.Label(master=root, text="Dataset: ",width=10,height=1,font=f).grid(row=0,column=0,columnspan=2)
fileentry = tk.Entry(master=root, textvariable=file_base)
fileentry.grid(row=0,column=2,columnspan=2)

tk.Label(master=root, text="Epoch: ",width=10,height=1,font=f).grid(row=1,column=0,columnspan=2)
epochentry = tk.Spinbox(master=root, from_=1, to=1000000, increment=10, textvariable=epoch)
epochentry.grid(row=1,column=2,columnspan=2)

readfile_btn = tk.Button(master=root, text="Choose File", command=_readfile,width=10,height=1,font=f)
readfile_btn.grid(row=0,column=4,columnspan=2)

train_btn = tk.Button(master=root, text="TRAINING!", command=_training,width=10,height=1,font=f)
train_btn.grid(row=1,column=4,columnspan=2)

tk.Label(master=root).grid(row=3,column=0)
tk.Label(master=root).grid(row=4,column=0)
tk.Label(master=root).grid(row=6,column=0,columnspan=2)
tk.Label(master=root).grid(row=7,column=2,columnspan=2)
quit_btn = tk.Button(master=root, text="QUIT", command=_quit,width=10,height=1,font=f)
quit_btn.grid(row=8,column=12)

root.mainloop()