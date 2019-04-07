#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//随机生成矩阵
void randomGenerate(cv::Mat &mat, int rows, int cols, int randomState=-1);

//激活函数sigmoid
void sigmoid(cv::Mat &H);

//二维数据转换为一维(从AxB到1xAB)
void mat2line(const cv::Mat &mat, cv::Mat &line, const int channels);
void mats2lines(const std::vector<cv::Mat> &mats, cv::Mat &output, const int channels);

//加载mnist数据集
void loadMnistData_csv(const std::string path, const float trainSampleRatio, 
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, 
                   bool shuffle=true);

//转换label列表为Mat类型
void label2target(const std::vector<bool> &label, cv::Mat &target);
void labels2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target);

//归一化
void normalizeImg(cv::Mat &img);

//找最大值
int getMaxId(const cv::Mat &line);

//计分
float calcScore(const cv::Mat &outputData, const cv::Mat &target);

#endif // FUNCTIONS_H
