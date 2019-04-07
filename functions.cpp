#include "functions.h"

void sigmoid(cv::Mat &H)
{
    for(int i=0;i<H.rows;i++)
        for(int j=0;j<H.cols;j++)
            H.at<float>(i,j) = 1 / ( 1 + std::exp(-H.at<float>(i,j)) );
}

void randomGenerate(cv::Mat &mat, int rows, int cols, int randomState)
{
    mat.create(rows,cols,CV_32F);
    
    cv::RNG rng;
    if(randomState != -1)
        rng.state = randomState;
    else
        rng.state = (unsigned)time(NULL);
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) = rng.uniform(-1.0,1.0);
}

void mat2line(const cv::Mat &mat, cv::Mat &line, const int channels)
{
    line.create(cv::Size(mat.rows*mat.cols*channels,1),CV_32F);
    
    if(channels==1)
    {
        for(int r=0;r<mat.rows;r++)
            for(int c=0;c<mat.cols;c++)
                line.at<float>(0,r*mat.cols+c) = float(mat.at<uchar>(r,c));
    }
    if(channels==3)
    {
        std::vector<cv::Mat> channels;
        cv::split(mat,channels);
        int j=0;
        for(int i=0;i<3;i++)
            for(int r=0;r<channels[i].rows;r++)
                for(int c=0;c<channels[i].cols;c++)
                {
                    line.at<float>(0,j) = float(channels[i].at<uchar>(r,c));
                    j++;
                }
    }
}

void mats2lines(const std::vector<cv::Mat> &mats, cv::Mat &output, const int channels)
{
    if(mats.empty())
        return;
    
    output.create(cv::Size(mats[0].rows*mats[0].cols*channels,mats.size()),CV_32F);
    
    for(int i=0;i<mats.size();i++)
    {
        cv::Mat lineROI = output(cv::Range(i,i+1),cv::Range(0,output.cols));
        mat2line(mats[i],lineROI, channels);
    }
}

void loadMnistData_csv(const std::string path, const float trainSampleRatio,
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, bool shuffle)
{
    std::ifstream fin(path);
    
    std::string tmpLine;
    std::vector<std::string> lines;
    
    while(std::getline(fin,tmpLine))
        lines.push_back(tmpLine);
    
    srand(time(NULL));
    if(shuffle)
        std::random_shuffle(lines.begin(),lines.end());
    
    int trainSize = lines.size()*trainSampleRatio;
    
    for(int j=0;j<trainSize;j++)
    {
        std::string line;
        line.assign(lines[j]);
        
        std::vector<bool> label_bin(10,0);
        label_bin[line[0]-48] = 1;
        trainLabelBins.push_back(label_bin);

        cv::Mat img(28,28,CV_8U);
        int pixNum=0;
        for(int i=2;i<line.size();i++)
        {
            int value=0;
            
            if(line[i] == ',')
                continue;
            
            while(i<line.size() && line[i] != ',')
            {
                value = value*10 + line[i] - 48;
                i++;
            }
            i--;
            
            int y = pixNum/28;
            int x = pixNum%28;
            img.at<uchar>(y,x) = value;
            
            pixNum++;
        }
        
        trainImgs.push_back(img);
    }
    
    for(int j=trainSize;j<lines.size();j++)
    {
        std::string line;
        line.assign(lines[j]);
        
        std::vector<bool> label_bin(10,0);
        label_bin[line[0]-48] = 1;
        testLabelBins.push_back(label_bin);

        cv::Mat img(28,28,CV_8U);
        int pixNum=0;
        for(int i=2;i<line.size();i++)
        {
            int value=0;
            
            if(line[i] == ',')
                continue;
            
            while(i<line.size() && line[i] != ',')
            {
                value = value*10 + line[i] - 48;
                i++;
            }
            i--;
            
            int y = pixNum/28;
            int x = pixNum%28;
            img.at<uchar>(y,x) = value;
            
            pixNum++;
        }
        
        testImgs.push_back(img);
    }
}

void label2target(const std::vector<bool> &label, cv::Mat &target)
{
    target.create(cv::Size(label.size(),1),CV_32F);
    for(int j=0;j<label.size();j++)
        target.at<float>(0,j) = float(label[j]);
}

void labels2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target)
{
    if(labels.empty())
        return;
    
    int labelLength = labels[0].size();
    target.create(cv::Size(labelLength,labels.size()),CV_32F);
    for(int i=0;i<labels.size();i++)
    {
        for(int j=0;j<labelLength;j++)
            target.at<float>(i,j) = float(labels[i][j]);
    }
}

void normalizeImg(cv::Mat &img)
{
    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
            img.at<float>(i,j) = (img.at<float>(i,j)-127) / 127.0;
}

int getMaxId(const cv::Mat &line)
{
    double minVal,maxVal;
    int minIdx[2],maxIdx[2];
    
    cv::minMaxIdx(line,&minVal,&maxVal,minIdx,maxIdx);
    
    return maxIdx[1];
}

float calcScore(const cv::Mat &outputData, const cv::Mat &target)
{
    int score = 0;
    for(int i=0;i<outputData.rows;i++)
    {
        cv::Mat ROI_o = outputData(cv::Range(i,i+1),cv::Range(0,outputData.cols));
        int maxId_O = getMaxId(ROI_o);
        
        cv::Mat ROI_t = target(cv::Range(i,i+1),cv::Range(0,target.cols));
        int maxId_T = getMaxId(ROI_t);
        
        if(maxId_O == maxId_T)
            score++;
    }

    float finalScore = score/(float)outputData.rows;
    
    return finalScore;
}
