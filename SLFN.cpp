#include "SLFN.h"

SLFN::SLFN(int inodes, int hnodes, int onodes, int channels, float learningRate)
{
    //参数初始化
    m_inodes = inodes;
    m_hnodes = hnodes;
    m_onodes = onodes;
    m_lr = learningRate;
    m_channels = channels;
    
    //随机生成初始权重矩阵
    randomGenerate(m_wih,m_inodes,m_hnodes);
    randomGenerate(m_who,m_hnodes,m_onodes);
}

float SLFN::train(const cv::Mat &img, const cv::Mat &target)
{
    //输入数据展开
    cv::Mat input;
    mat2line(img,input,m_channels);
    
    //归一化
    normalizeImg(input);
    
    //正向传播
    cv::Mat H = input*m_wih;
    sigmoid(H);
    cv::Mat output = H*m_who;
    sigmoid(output);
    
    //计算误差
    cv::Mat EO = target - output;
    cv::Mat EH = EO*m_who.t();
    
    //更新权重矩阵
    m_who += m_lr*H.t()*( EO.mul(output.mul(1-output)) );
    m_wih += m_lr*input.t()*( EH.mul(H.mul(1-H)) );
    
    //返回误差
    cv::Scalar s = cv::sum(EO.mul(EO));
    return s[0];
}

float SLFN::validate(const std::vector<cv::Mat> &testImgs, const std::vector<std::vector<bool> > &testLabelBins)
{
    cv::Mat testInput;
    mats2lines(testImgs,testInput,m_channels);
    
    normalizeImg(testInput);
    
    cv::Mat t = testInput*m_wih;
    sigmoid(t);
    
    t *= m_who;
    sigmoid(t);
    
    cv::Mat testTarget;
    labels2target(testLabelBins,testTarget);
    float score = calcScore(t,testTarget);
    
    return score;
}

void SLFN::ELM_IniWeight(const std::vector<cv::Mat> &imgs, const std::vector<std::vector<bool> > &trainLabelBins)
{
    cv::Mat input;
    mats2lines(imgs,input,m_channels);
    normalizeImg(input);
    cv::Mat target;
    labels2target(trainLabelBins,target);
    
    cv::Mat H = input*m_wih;
    sigmoid(H);
    
    m_who = (H.t()*H).inv(1)*H.t()*target;
}
