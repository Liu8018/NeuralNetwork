#include "SLFN.h"

int main()
{
    //载入MNIST数据集
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    loadMnistData_csv("/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",
                      0.8,trainImgs,testImgs,trainLabelBins,testLabelBins);
    
    //建立模型
    SLFN slfnModel(784, 300, 10, 1, 0.03);
    
    //训练
    int epochs = 6;
    for(int ep=0;ep<epochs;ep++)
        for(int i=0;i<trainImgs.size();i++)
        {
            cv::Mat target;
            label2target(trainLabelBins[i],target);
            
            float e = slfnModel.train(trainImgs[i],target);
            
            if(i%100==0)
                std::cout<<"e:"<<e<<std::endl;
        }
    
    //测试
    float score0 = slfnModel.validate(trainImgs,trainLabelBins);
    float score1 = slfnModel.validate(testImgs,testLabelBins);
    std::cout<<"score on training data:"<<score0<<std::endl;
    std::cout<<"score on test data:"<<score1<<std::endl;
    
    return 0;
}
