#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-08
# @Author  : Luo Chuwei
# @Link    : https://github.com/luochuwei/GuidedFilter
# @Version : 0

import numpy as np
import cv2

class GuidedFilter(object):
    """
    Author:Luo Chuwei
    Email:luochuwei@gmail.com
    GuidedFilter python
    """
    def __init__(self, I, p, r, eps):
        """
        I: input guidence map, ndarray
        p: filtering input image p
        r: radius, int
        eps: regularization
        """
        self.I = self.Float(I)
        self.p = self.Float(p)
        self.radius = r
        self.eps = eps

        assert I.shape == p.shape, "Input images have different shapes!"

    def filter(self):
        if len(self.I.shape) == 2:
            self.channels = 1
            self.Image_Filterd = self.GuidedFilterGray()
        elif len(self.I.shape) > 2:
            self.height = self.p.shape[0]
            self.width = self.p.shape[1]
            self.channels = self.p.shape[2]
            img = np.zeros_like(self.p, dtype=np.float32)
            for c in range(self.channels):
                img[:, :, c] = self.GuidedFilterMultiDim(self.p[:, :, c])
            self.Image_Filterd = img
        else:
            print("Error dim!")
            return -1
        return self.Unit8(self.Image_Filterd)

    def GuidedFilterGray(self):
        # 1
        I_mean = self.boxfilter(self.I, self.radius)
        p_mean = self.boxfilter(self.p, self.radius)
        I_corr = self.boxfilter(self.I * self.I, self.radius)
        Ip_corr = self.boxfilter(self.I * self.p, self.radius)
        # 2
        I_var = I_corr - I_mean * I_mean
        Ip_cov = Ip_corr - I_mean * p_mean
        # 3
        a = covIp / (I_var + self.eps)
        b = p_mean - a * I_mean
        # 4
        a_mean = self.boxfilter(a, self.radius)
        b_mean = self.boxfilter(b, self.radius)
        # 5
        q = a_mean * self.I + b_mean

        return q

    def GuidedFilterMultiDim(self, input_pc):
        input_p = np.expand_dims(input_pc, axis=2)

        I_mean = self.boxfilter(self.I, self.radius)
        p_mean = self.boxfilter(input_p, self.radius)
        print p_mean.shape

        I_ = self.I.reshape((self.height*self.width, self.channels, 1))
        I_mean_ = I_mean.reshape((self.height*self.width, self.channels, 1))

        I_corr_ = np.matmul(I_, I_.transpose(0, 2, 1))
        I_corr_ = I_corr_.reshape((self.height, self.width, self.channels*self.channels))
        I_corr_ = self.boxfilter(I_corr_, self.radius)
        I_corr = I_corr_.reshape((self.height*self.width, self.channels, self.channels))

        U = np.expand_dims(np.eye(self.channels, dtype=np.float32), axis=0)

        left = np.linalg.inv(I_corr + self.eps * U)

        Ip_corr = self.boxfilter(self.I*input_p, self.radius)

        Ip_cov = Ip_corr - I_mean * p_mean
        right = Ip_cov.reshape((self.height*self.width, self.channels, 1))

        a = np.matmul(left, right)
        axmeanI = np.matmul(a.transpose((0, 2, 1)), I_mean_)
        axmeanI = axmeanI.reshape((self.height, self.width, 1))
        b = p_mean - axmeanI
        a = a.reshape((self.height, self.width, self.channels))

        a_mean = self.boxfilter(a, self.radius)
        b_mean = self.boxfilter(b, self.radius)

        a_mean = a_mean.reshape((self.height*self.width, 1, self.channels))
        b_mean = b_mean.reshape((self.height*self.width, 1, 1))
        I_ = self.I.reshape((self.height*self.width, self.channels, 1))

        q = np.matmul(a_mean, I_) + b_mean
        q = q.reshape((self.height, self.width))

        return q

    def boxfilter(self, img, sz):
        return cv2.boxFilter(img, cv2.CV_32F, (sz, sz))

    def Float(self, input_x):
        x = input_x.copy()
        x = np.clip(np.float32(x / 255.), 0.0, 1.0)
        return x

    def Unit8(self, input_x):
        x = input_x.copy()
        x = x * 255.0
        x = np.clip(np.uint8(x), 0, 255)

if __name__ == '__main__':
    img = cv2.imread("Lenna.png")
    GF = GuidedFilter(img, img, 5, 0.005)
    img_GF = GF.filter()
    cv2.imwrite("Lenna_result.png", img_GF)