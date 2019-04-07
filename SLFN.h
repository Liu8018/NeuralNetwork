#ifndef SLFN_H
#define SLFN_H

#include "functions.h"

class SLFN
{
public:
    SLFN(int inodes, int hnodes, int onodes, int channels, float learningRate);
    
    float train(const cv::Mat &img/*h x w*/, const cv::Mat &target/*1 x o*/);
    
    float validate(const std::vector<cv::Mat> &testImgs, const std::vector<std::vector<bool>> &testLabelBins);
    
private:
    int m_inodes;
    int m_hnodes;
    int m_onodes;
    int m_channels;
    float m_lr;
    cv::Mat m_wih;
    cv::Mat m_who;
};

#endif // SLFN_H
