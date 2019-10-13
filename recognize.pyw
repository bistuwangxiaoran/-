import tkinter as tk
import numpy as np
import cv2
from tkinter import filedialog,messagebox
from PIL import Image, ImageTk

def on_mouse(event, x, y, flags, param):
    global img,point1,point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):   #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5) # 图像，矩形顶点，相对顶点，颜色，粗细
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])     
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        resize_img = cv2.resize(cut_img, (28,28)) # 调整图像尺寸为28*28
        ret,thresh_img = cv2.threshold(resize_img,127,255,cv2.THRESH_BINARY) # 二值化
        cv2.imshow('result', thresh_img)
        cv2.imwrite('cut.png', thresh_img) 
        
def Open(lp):
    global fn
    fn= filedialog.askopenfilename(filetypes=[('.png', '*.png'), ('.jpg', '*.jpg')])
    if(fn!=''):
      bg_o = Image.open(fn).resize((100,100))
      bg3= ImageTk.PhotoImage(bg_o)
      lp.configure(image=bg3)
      lp.image = bg3
    else:
      del fn

def Cut(lp2):
    global fn,img
    try:
      fn
    except NameError:
      fn_exists = False
      img=cv2.imread('example.png')
    else:
      fn_exists = True
      img=cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换图像为单通道(灰度图)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    bg_o= Image.open('cut.png').resize((100, 100))
    bg3 = ImageTk.PhotoImage(bg_o)
    lp2.configure(image=bg3)
    lp2.image = bg3


def Recongnize(lt):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()#清空之前值
    im = Image.open('cut.png')
    data = list(im.getdata())
    result = [(255-x)*1.0/255.0 for x in data]     
    
    # 为输入图像和目标输出类别创建节点
    x = tf.compat.v1.placeholder("float", shape=[None, 784]) # 训练所需数据  占位符

    # *************** 构建多层卷积网络 *************** #
    def weight_variable(shape):
      initial = tf.random.truncated_normal(shape, stddev=0.1) # 取随机值，符合均值为0，标准差stddev为0.1
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1,28,28,1]) # -1表示任意数量的样本数,大小为28x28，深度为1的张量

    W_conv1 = weight_variable([5, 5, 1, 32]) # 卷积在每个5x5的patch中算出32个特征。
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 在输出层之前加入dropout以减少过拟合
    keep_prob = tf.compat.v1.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,rate=1-keep_prob)

    # 全连接层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # 输出层
    # tf.nn.softmax()将神经网络的输层变成一个概率分布  
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    
    # *************** 开始识别 *************** #
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver() # 定义saver
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, 'model/model.ckpt')#这里使用了之前保存的模型参数
        prediction = tf.argmax(y_conv,1)
        predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0},session=sess)
      
    
    lt.set("识别结果：%d" %predint[0])
    tk.messagebox.showinfo("完成","识别结果为：%d" %predint[0])
    
win = tk.Tk() # 创建窗口
sw = win.winfo_screenwidth()
sh = win.winfo_screenheight()
ww, wh = 400, 450
x, y = (sw-ww)/2, (sh-wh)/2
win.geometry("%dx%d+%d+%d"%(ww, wh, x, y-40)) # 居中放置窗口
win.title('手写体识别') # 窗口命名

img= Image.open("example.png").resize((100, 100))
bg= ImageTk.PhotoImage(img)
lp = tk.Label(win, image=bg,text='原始图片',compound='right')
lp.pack(expand=True)
tk.Button(win, text='选择图片', width=20, height=2,command=lambda:Open(lp)).pack()
tk.Button(win, text='剪裁', width=20, height=2,command=lambda:Cut(lp2)).pack()

lt = tk.StringVar() # 变量文字
lt.set('识别结果：')
img2= Image.open("cut.png").resize((100, 100))
bg2= ImageTk.PhotoImage(img2)
lp2 = tk.Label(win, image=bg2,text='剪裁图片',compound='right')
lp2.pack(expand=True)

tk.Button(win, text='识别', width=20, height=2,command=lambda:Recongnize(lt)).pack()
tk.Label(win, textvariable=lt, width=20, height=2,anchor='w').pack(expand=True)



win.mainloop()
