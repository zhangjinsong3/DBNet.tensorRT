//
// Created by zjs on 20-12-23.
//

#include "db_detector.h"

int main()
{
  BRFD::DBDetector text_detector = BRFD::DBDetector(
                                                    "../runtime_data/dbnet/",
                                                    "dbnet_res18_640_640.onnx",
                                                    "dbnet_res18_960_480.onnx", 1, 0.3, 0.7, 1.8);
  cv::Mat bill_image = cv::imread("../runtime_data/images/20200730154533_4194304_1_bill.jpg");
  if (bill_image.empty())
  {
    std::cerr << "path wrong!" << endl;
    return -1;
  }
  cv::cuda::GpuMat g_bill_img;
  g_bill_img.upload(bill_image);

  for (int i = 0; i < 100; ++i) {
    cv::Mat prob;

    // 输入 cv::Mat
//    double tic_h = cv::getTickCount();
//    text_detector.inference(bill_image, prob);
//    cout << "Mat inference done! cost time: " << (cv::getTickCount() - tic_h) * 1000 /cv::getTickFrequency() << " ms" << endl;
//    text_detector.visualize(bill_image, prob);

    // 输入cv::cuda::GpuMat
    cv::cuda::GpuMat g_prob;

    double tic = cv::getTickCount();
    text_detector.inference(g_bill_img, g_prob);
    cout << "GpuMat inference done! cost time: " << (cv::getTickCount() - tic) * 1000 /cv::getTickFrequency() << " ms" << endl;

    tic = cv::getTickCount();
    g_prob.download(prob);
    cout << "GpuMat download! cost time: " << (cv::getTickCount() - tic) * 1000 /cv::getTickFrequency() << " ms" << endl;
    cv::imwrite("prob.jpg", prob);

    // convert prob map to ordered polygon
    tic = cv::getTickCount();
    std::vector<std::vector<cv::Point2f>> order_points;
    text_detector.map2ordered_points(prob, order_points, /*unclip*/true);
    cout << "map2ordered_points done! cost time: " << (cv::getTickCount() - tic) * 1000 /cv::getTickFrequency() << " ms" << endl;

    // Crop
    tic = cv::getTickCount();
    std::vector<cv::cuda::GpuMat> d_crops;
    text_detector.crop(g_bill_img, order_points, d_crops);
    cout << "GpuMat croped! cost time: " << (cv::getTickCount() - tic) * 1000 /cv::getTickFrequency() << " ms" << endl;

    for(auto &d_crop: d_crops)
    {
      cv::Mat h_crop;
      d_crop.download(h_crop);
      cv::imwrite("crop.jpg", h_crop);
      break;
    }

    // Visualize polygon
    cv::Mat viz = text_detector.visualize(bill_image, order_points[1], cv::Scalar(128, 250, 0), "seesee");
    cv::imwrite("seesee.jpg", viz);

  }
  return 0;
}