//
// Created by zjs on 20-12-23.
//

#ifndef DBNET_C_DB_DETECTOR_H
#define DBNET_C_DB_DETECTOR_H

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "math.h"
#include <algorithm>

#include <cuda_runtime_api.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "NvInfer.h"
#include "NvUtils.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"

#include "common.h"
#include "filesystem.h"
#include "python_like_func.h"
#include "clipper/clipper.hpp"

#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvonnxparser;

namespace BRFD
{
  class DBDetector {
    public:
      DBDetector(std::string model_dir,
                 std::string model_name_1,
                 std::string model_name_2,
                 int _max_batch_size=1,
                 float _thresh=0.3,
                 float _box_thresh=0.7,
                 float _unclip_ratio=1.5);
      ~DBDetector();

      int inference(cv::Mat image, cv::Mat& prob);
      int inference(cv::cuda::GpuMat image, cv::cuda::GpuMat &prob);
      int inference(string image_path, cv::Mat& prob);

      int visualize(cv::Mat ori_img, cv::Mat prob, string save_or_show="save");
      int visualize(cv::Mat ori_img, std::vector<std::vector<cv::Point2f>> order_points, string save_or_show="save");
      cv::Mat visualize(cv::Mat ori_img, std::vector<cv::Point2f> order_points, cv::Scalar color, string text="");

      int map2ordered_points(cv::Mat &prob, std::vector<std::vector<cv::Point2f>>& order_points, bool unclip= true);

      int crop(cv::Mat& ori_img, std::vector<std::vector<cv::Point2f>> order_points, std::vector<cv::Mat>& crops);
      int crop(cv::cuda::GpuMat& g_ori_img, std::vector<std::vector<cv::Point2f>> order_points, std::vector<cv::cuda::GpuMat>& crops);

   private:
    float thresh=0.3;
    float box_thresh=0.7;
    float unclip_ratio=1.5;
    int max_batch_size=1;
    // 加载1x1 和2x1 两个模型 应对尺度变化,实现伪 动态尺寸输入
    IExecutionContext* context_1x1;
    ICudaEngine* engine_1x1;
    int inputH_1x1 = 640;
    int inputW_1x1 = 640;
    std::vector<int64_t> bufferSize_1x1;
    void* buffers_1x1[2];  // 0 for input, 1 for output


    IExecutionContext* context_2x1;
    ICudaEngine* engine_2x1;
    int inputH_2x1 = 960;
    int inputW_2x1 = 480;
    std::vector<int64_t> bufferSize_2x1;
    void* buffers_2x1[2]; // 0 for input, 1 for output

    cudaStream_t stream;

    int get_engine(const std::string onnx_file,
                   const std::string engine_file);

    int preprocess(const cv::Mat &img,
                   void *tensorRTBuffer,
                   int input_h,
                   int input_w);

    int preprocess(const cv::cuda::GpuMat& gpu_img,
                   void* tensorRTBuffer,
                   int input_h,
                   int input_w);

    std::vector<cv::Point> expandBox(std::vector<cv::Point>& inBox, float ratio = 1.0);

    int64_t volume(const nvinfer1::Dims& d)
    {
      return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    unsigned int getElementSize(nvinfer1::DataType t)
    {
      switch (t)
      {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
      }
      throw std::runtime_error("Invalid DataType.");
      return 0;
    }

    inline std::string cv_type2str(int type) {
      std::string r;

      uchar depth = type & CV_MAT_DEPTH_MASK;
      uchar chans = 1 + (type >> CV_CN_SHIFT);

      switch (depth) {
        case CV_8U:
          r = "8U";
          break;
        case CV_8S:
          r = "8S";
          break;
        case CV_16U:
          r = "16U";
          break;
        case CV_16S:
          r = "16S";
          break;
        case CV_32S:
          r = "32S";
          break;
        case CV_32F:
          r = "32F";
          break;
        case CV_64F:
          r = "64F";
          break;
        default:
          r = "User";
          break;
      }

      r += "C";
      r += (chans + '0');

      return r;
    }
  };
}


#endif //DBNET_C_DB_DETECTOR_H
