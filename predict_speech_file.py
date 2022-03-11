#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
用于通过ASRT语音识别系统预测一次语音文件的程序
"""
import atexit
import os
import threading
import time
import wave

import numpy as np
import pyaudio

from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251
from speech_features import Spectrogram

# ffmpeg -i 10.m4a -acodec pcm_s16le -ac 1 -ar 16000 10.wav


CHUNK = 1000  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 单声道
RATE = 16000  # 采样频率


""" 录音功能 """
print("录音开始")
p = pyaudio.PyAudio()  # 实例化对象
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)  # 打开流，传入响应参数


def recoreding(threshold_pause=0.3,threshold_norespon=5,maxOutputTime=3)->wave:
    chunkContinue = []
    stat = True
    NoRespon = False
    timeNoRespon = time.time()
    timePause = timeNoRespon # =time.time()
    while stat:
        data = stream.read(CHUNK,exception_on_overflow = False)
        audio_data = np.frombuffer(data, dtype=np.short)
        temp = np.max(audio_data)
        if temp > 1000:
            timeNoRespon = time.time()
            NoRespon = True
        else:
            NoRespon = False
            if timeNoRespon - time.time() > threshold_norespon:
                stat=False
        if timePause - time.time() > threshold_pause or (maxOutputTime < timeNoRespon and not NoRespon):
            print(chunkContinue)
            yield chunkContinue
            chunkContinue = []
        else:
            print(len(chunkContinue))
            chunkContinue.extend(data)

atexit.register(stream.stop_stream)  # 关闭流
atexit.register(stream.close)
atexit.register(p.terminate)




AUDIO_LENGTH = 27991
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428
sm251 = SpeechModel251(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)
feat = Spectrogram()
ms = ModelSpeech(sm251, feat, max_label_length=64)

ms.load_model('save_models/' + sm251.get_model_name() + '.model.h5')
for wav_data in recoreding():
    start_time = time.time()
    # res = ms.recognize_speech_from_file('10.wav')
    res = ms.recognize_speech(wav_data, 16000)
    print('声学模型语音识别结果：\n', res, f"Cycle time: {time.time() - start_time}")
