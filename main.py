from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import QtCore,QtGui
import sys

import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox

from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device, load_classifier
from utils.capnums import Camera
from dialog.rtsp_win import Window


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)

    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov7.pt'
        self.current_weight = './yolov7.pt'    # 当前权重
        self.source = 'val/images/6gtYIxSFJUqj024TCvXph5cQMd3EyfOLsbPio8a7.jpg'                       # 视频源
        self.conf_thres = 0.37                  # 置信度
        self.iou_thres = 0.4                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = False                 # 继续/暂停
        self.percent_length = 1000              # 进度条  计算百分比的公式为：(value() - minimum()) / (maximum() - minimum())
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=4,  # 框线粗细
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA


            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0    ###########
            for param in model.parameters():   ###########
                num_params += param.numel()      ###########
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            # Get names and colors  改动
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize  做分类的网络
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                    device).eval()
            vid_path, vid_writer = None, None #改动
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once  热身 来一次前馈传播
            #更改
            old_img_w = old_img_h = imgsz
            old_img_b = 1
            count = 0
            # 跳帧检测
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:

                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)  #进度条显示为0
                    self.send_msg.emit('停止')  #label显示停止
                    break
                # 临时更换模型
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

                    self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset) #加载路径，改变后的图像，原图，cap

                    count += 1
                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:#计算进度进行输出
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}

                    # 计类别
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0  归一化处理上
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]  # torch.Size([1, 18900, 85])  预测了18900候选框，85=80+5  augment图像增强

                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms)

                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    for i, det in enumerate(pred):
                        im0 = im0s.copy()

                        if len(det):

                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class  类别
                                statistic_dic[names[c]] += 1

                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),  #画框
                                             line_thickness=line_thickness)


                    if self.rate_check:
                        time.sleep(1/self.rate)

                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('检测结束')
                        # 正常跳出循环
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)#缩小
        self.maxButton.clicked.connect(self.max_or_restore)#放大
        self.closeButton.clicked.connect(self.close)#关闭
        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov7线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))

        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file) #文件按钮

        self.runButton.clicked.connect(self.run_or_continue)#暂停
        self.stopButton.clicked.connect(self.stop)#停止

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def checkrate(self):  # 是否启用帧延时
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'#配置文件地址
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"iou": 0.26,
                          "conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            rate = config['rate']
            check = config['check']

        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    def change_val(self, x, flag):
        if   flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
            print(self.det_thread.rate)
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)


    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        _translate = QtCore.QCoreApplication.translate

        self.runButton.setText(_translate("mainWindow", "开始检测"))


        self.statistic_msg(msg)#进行展示

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()#获取当前选择的模型名字
        self.det_thread.weights = "./pt/%s" % self.model_type#改变线程中weights名字
        self.statistic_msg('模型切换为%s' % x)#对切换模型进行输出

    def open_file(self):

        config_file = 'config/fold.json'

        config = json.load(open(config_file, 'r', encoding='utf-8'))#从json文件中加载路径，json.loads将字符串转为Python对象

        open_fold = config['open_fold']

        if not os.path.exists(open_fold):#判断是否存在该地址
            open_fold = os.getcwd()

        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)

            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

            self.stop()

    def max_or_restore(self):#界面放大
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):

        _translate = QtCore.QCoreApplication.translate

        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            if source.isnumeric():            #source.isnumeric()：如果字符串中只包含数字字符，则返回 True，否则返回 False
                source = '请选择文件后检测'
                self.det_thread.is_continue = False
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
            self.runButton.setText(_translate("mainWindow", "暂停检测"))
        else:
            self.runButton.setText(_translate("mainWindow", "开始检测"))
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')





    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='请稍等，正在关闭程序。。。', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
