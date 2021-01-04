//
// Created by zjs on 20-12-23.
//

//#include <hwloc.h>
#include "db_detector.h"
using namespace BRFD;
using namespace std;
using namespace cv;
static Logger gLogger;

// TODO: do not take use of _box_thresh, if need, reference tensorrtx
DBDetector::DBDetector(std::string model_dir,
                       std::string model_name_1,
                       std::string model_name_2,
                       int _max_batch_size,
                       float _thresh,
                       float _box_thresh,
                       float _unclip_ratio)
{
  this->thresh = _thresh;
  this->box_thresh = _box_thresh;
  this->unclip_ratio = _unclip_ratio;
  this->max_batch_size = _max_batch_size;

  // GET ENGINE 1X1
  get_engine(model_dir + model_name_1,
             model_dir + filenameWithoutExtension(model_name_1) + ".engine",
             context_1x1,
             engine_1x1,
             buffers_1x1,
             bufferSize_1x1,
             inputH_1x1,
             inputW_1x1);
  // GET ENGINE 2X1
  get_engine(model_dir + model_name_2,
             model_dir + filenameWithoutExtension(model_name_2) + ".engine",
             context_2x1,
             engine_2x1,
             buffers_2x1,
             bufferSize_2x1,
             inputH_2x1,
             inputW_2x1);
}

DBDetector::~DBDetector()
{
  CHECK(cudaFree(buffers_1x1[0]));
  CHECK(cudaFree(buffers_1x1[1]));

  CHECK(cudaFree(buffers_2x1[0]));
  CHECK(cudaFree(buffers_2x1[1]));

  cudaStreamDestroy(stream);
  context_1x1->destroy();
  engine_1x1->destroy();
  context_2x1->destroy();
  engine_2x1->destroy();
}

int DBDetector::get_engine(const std::string onnx_file,
                           const std::string engine_file,
                           IExecutionContext* &context,
                           ICudaEngine* &engine,
                           void* buffers[],
                           std::vector<int64_t >& bufferSize,
                           int inputH,
                           int inputW)
{
  if (!fileExists(engine_file.c_str()))
  {
    // CONVERT ONNX MODEL TO TENSORRT ENGINE AND SAVE IT!
    // create the builder
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
//    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnx_file.c_str(), verbosity)){
      string msg("dbnet: failed to parse onnx file");
      gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
      exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(max_batch_size);
//    auto config = builder->createBuilderConfig();
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setFp16Mode(1);
    //samplesCommon::enableDLA(builder, 1); // use DLA, ps: DLA only support for mobile GPU device (Xavier)
    cout << "dbnet: start building engine for dbnet" << endl;
    engine = builder->buildCudaEngine(*network);
    cout << "dbnet: build engine for dbnet done" << endl;
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // save engine
    nvinfer1::IHostMemory* data = engine->serialize();
    std::ofstream file;
    file.open(engine_file, std::ios::binary | std::ios::out);
    cout << "writing engine file for dbnet..." << endl;
    file.write((const char*)data->data(), data->size());
    cout << "dbnet: save engine file for dbnet done" << endl;
    file.close();

    // then close everything down
    engine->destroy();
    network->destroy();
    builder->destroy();
  }
  // Load engine
  fstream file;
  cout << "dbnet: loading filename from:" << engine_file << endl;
  file.open(engine_file, ios::binary | ios::in);
  file.seekg(0, ios::end);
  int length = file.tellg();
  file.seekg(0, ios::beg);
  std::unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();
  cout << "dbnet: load engine done" << endl;

  // deserializing
  std::cout << "dbnet: deserializing" << endl;
//  nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger);
  IRuntime* trtRuntime = createInferRuntime(gLogger);
  engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
  cout << "dbnet: deserialize done" << endl;

  // get context
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);

  //get buffers
  assert(engine->getNbBindings() == 2);
  int nbBindings = engine->getNbBindings();
  bufferSize.clear();
  bufferSize.resize(nbBindings);

  for (int i = 0; i < nbBindings; ++i)
  {
    nvinfer1::Dims dims = engine->getBindingDimensions(i);
    nvinfer1::DataType dtype = engine->getBindingDataType(i);
    int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
    bufferSize[i] = totalSize;
    CHECK(cudaMalloc(&buffers[i], totalSize));
  }

  assert(max_batch_size * 3 * inputH * inputW * sizeof(float) == bufferSize[0]); // input image
  assert(max_batch_size * 2 * inputH * inputW * sizeof(float) == bufferSize[1]); // output 2 feature maps, one for prob, one for thresh

  CHECK(cudaStreamCreate(&stream));
  trtRuntime->destroy();
}

int DBDetector::inference(string image_path, cv::Mat& prob)
{
  cv::Mat image;
  image = cv::imread(image_path);
  if (image.empty())
  {
    std::cerr << image_path << " maybe wrong!!!" << endl;
    return -1;
  }
  return inference(image, prob);
}

int DBDetector::inference(cv::Mat image, cv::Mat& prob)
{
  int w = image.cols;
  int h = image.rows;
  IExecutionContext* context;
  int inputH;
  int inputW;
  std::vector<int64_t> bufferSize;
  void* buffers[2];

  if ((float(h) / float(w)) < 1.2)
  {
    context = context_1x1;
    inputH = inputH_1x1;
    inputW = inputW_1x1;
    bufferSize = bufferSize_1x1;
    buffers[0] = buffers_1x1[0];
    buffers[1] = buffers_1x1[1];
  }
  else
  {
    context = context_2x1;
    inputH = inputH_2x1;
    inputW = inputW_2x1;
    bufferSize = bufferSize_2x1;
    buffers[0] = buffers_2x1[0];
    buffers[1] = buffers_2x1[1];
  }

  void* input_host = (float*)malloc(inputH * inputW * 3 * sizeof(float));
  preprocess(image, input_host, inputH, inputW);
//  cout << "dbnet: preprcess done!" << endl;
  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  CHECK(cudaMemcpyAsync(buffers[0], input_host, bufferSize[0], cudaMemcpyHostToDevice, stream));
//  cout << "dbnet: cudaMemcpyHostToDevice done!" << endl;

  // do inference
  context->enqueue(max_batch_size, buffers, stream, nullptr);
//  cout << "dbnet: enqueue done!" << endl;

  // get output
  float* output = new float[bufferSize[1]];

  CHECK(cudaMemcpyAsync(output, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
//  cout << "dbnet: cudaMemcpyDeviceToHost done!" << endl;
  cudaStreamSynchronize(stream);

  // output 为 2* inputH* inputW  拿出第一个prob
  cv::Mat map = cv::Mat(inputH, inputW, CV_32FC1, output);
  cv::threshold(map, prob, thresh, 255, cv::THRESH_BINARY);
  prob.convertTo(prob, CV_8UC1);

//  cv::imwrite("prob.jpg", prob);
  cout << "dbnet: binary prob done!" << endl;

  return 0;
}

int DBDetector::inference(cv::cuda::GpuMat image, cv::cuda::GpuMat& prob)
{
  int w = image.cols;
  int h = image.rows;
  IExecutionContext* context;
  int inputH;
  int inputW;
  std::vector<int64_t> bufferSize;
  void* buffers[2];

  if ((float(h) / float(w)) < 1.2)
  {
    context = context_1x1;
    inputH = inputH_1x1;
    inputW = inputW_1x1;
    bufferSize = bufferSize_1x1;
    buffers[0] = buffers_1x1[0];
    buffers[1] = buffers_1x1[1];
  }
  else
  {
    context = context_2x1;
    inputH = inputH_2x1;
    inputW = inputW_2x1;
    bufferSize = bufferSize_2x1;
    buffers[0] = buffers_2x1[0];
    buffers[1] = buffers_2x1[1];
  }

  preprocess(image, buffers[0], inputH, inputW);

  // do inference
//  context->execute(max_batch_size, buffers);
  context->enqueue(max_batch_size, buffers, stream, nullptr);


  cv::cuda::GpuMat g_prob = cv::cuda::GpuMat(inputH, inputW, CV_32FC1, buffers[1]);
  cv::cuda::threshold(g_prob, g_prob, thresh, 255, cv::THRESH_BINARY);
  g_prob.convertTo(prob, CV_8UC1);
//  uint_prob.download(prob);

  // get output
//  float* output = new float[bufferSize[1]];
//
//  CHECK(cudaMemcpyAsync(output, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
//  cudaStreamSynchronize(stream);
//
//  // output 为 2* inputH* inputW  拿出第一个prob
//  cv::Mat map = cv::Mat(inputH, inputW, CV_32FC1, output);
//  cv::threshold(map, prob, thresh, 255, cv::THRESH_BINARY);
//  prob.convertTo(prob, CV_8UC1);
//  cv::imwrite("prob.jpg", prob);
  cout << "dbnet: binary prob done!" << endl;

  return 0;
}

int DBDetector::visualize(cv::Mat ori_img, cv::Mat prob, string save_or_show)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarcy;
//  cout << cv_type2str(prob.type()) << endl;
  cv::findContours(prob, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

  std::vector<cv::Rect> boundRect(contours.size());
  std::vector<cv::RotatedRect> box(contours.size());
  cv::Point2f rect[4];
  cout << "dbnet: Find contours: " << contours.size() << endl;
  for (int i = 0; i < contours.size(); i++) {
    if (cv::contourArea(contours[i]) < 0.0005 * (prob.rows * prob.cols))
    {
      cout << "dbnet: ignore an small contour contourArea: " << cv::contourArea(contours[i]) << "vs" << prob.cols*prob.rows << endl;
      continue;
    }

    auto expanded_contour = expandBox(contours[i], unclip_ratio);;
    box[i] = cv::minAreaRect(cv::Mat(expanded_contour));
//    cout << "expanded box:"<< box[i].center << endl;
    //boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
    //绘制外接矩形和    最小外接矩形（for循环）
    box[i].points(rect);//把最小外接矩形四个端点复制给rect数组
    for (int j = 0; j < 4; j++) {
      cv::Point2f p1, p2;
      p1.x = round(rect[j].x / prob.cols * ori_img.cols);
      p1.y = round(rect[j].y / prob.rows * ori_img.rows);
      p2.x = round(rect[(j + 1) % 4].x / prob.cols * ori_img.cols);
      p2.y = round(rect[(j + 1) % 4].y / prob.rows * ori_img.rows);
//      cout << "line " << p1.x << ", " << p1.y << " to " << p2.x << ", " << p2.y << endl;
      cv::line(ori_img, p1, p2, cv::Scalar(0, 80*j, 255), 2, 8);
    }
  }
  if (save_or_show == "save")
    cv::imwrite("./db_result_mat.jpg", ori_img);
  else
    {
      cv::namedWindow("db_result", 0);
      cv::imshow("db_result", ori_img);
      if (27 == cv::waitKey(1))
        return 0;
    }
  return 0;
}

int DBDetector::visualize(cv::Mat ori_img, std::vector<std::vector<cv::Point2f>> order_points, string save_or_show)
{
  for (auto& or_points: order_points) {
    for (int j = 0; j < 4; j++) {
      cv::Point2f p1, p2;
      p1.x = round(or_points[j].x * ori_img.cols);
      p1.y = round(or_points[j].y * ori_img.rows);
      p2.x = round(or_points[(j + 1) % 4].x * ori_img.cols);
      p2.y = round(or_points[(j + 1) % 4].y * ori_img.rows);
//      cout << "line " << p1.x << ", " << p1.y << " to " << p2.x << ", " << p2.y << endl;
      cv::line(ori_img, p1, p2, cv::Scalar(0, 80*j, 255), 2, 8);
    }
  }
  if (save_or_show == "save")
    cv::imwrite("./db_result_gpumat.jpg", ori_img);
  else
  {
    cv::namedWindow("db_result", 0);
    cv::imshow("db_result", ori_img);
    if (27 == cv::waitKey(1))
      return 0;
  }
  return 0;
}

cv::Mat DBDetector::visualize(cv::Mat ori_img, std::vector<cv::Point2f> order_points, cv::Scalar color, string text)
{
  cv::Mat canvas;
  ori_img.copyTo(canvas);
  assert(order_points.size() > 3);

  for (int j = 0; j < 4; j++) {
    cv::Point2f p1, p2;
    p1.x = round(order_points[j].x * canvas.cols);
    p1.y = round(order_points[j].y * canvas.rows);
    p2.x = round(order_points[(j + 1) % 4].x * canvas.cols);
    p2.y = round(order_points[(j + 1) % 4].y * canvas.rows);
//      cout << "line " << p1.x << ", " << p1.y << " to " << p2.x << ", " << p2.y << endl;
    if (j==0)
    {
      cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 2, 8);
      if (text!="")
      {
        cv::putText(canvas, text, (p1 + p2)/2, cv::FONT_HERSHEY_PLAIN, 2, color);
      }
    }else
      {
        cv::line(canvas, p1, p2, color, 2, 8);
      }
  }

  return canvas;
}

int DBDetector::crop(cv::Mat &ori_img, std::vector<std::vector<cv::Point2f>> order_points, std::vector<cv::Mat>& crops)
{
  for (auto& or_points: order_points) {
    for (auto& point: or_points){
      point.x *= ori_img.cols;
      point.y *= ori_img.rows;
    }
    int w = static_cast<int>(sqrt(pow(or_points[3].x - or_points[2].x, 2)  + pow(or_points[3].y - or_points[2].y, 2)));
    int h = static_cast<int>(sqrt(pow(or_points[3].x - or_points[0].x, 2)  + pow(or_points[3].y - or_points[0].y, 2)));
//    cout << "order_points: " << or_points << endl;
//    cout << "w: " << w << ", h:" << h << endl;
    cv::Point2f src[] = {
        or_points[0],
        or_points[1],
        or_points[2],
        or_points[3]
    };
    cv::Point2f dst[]= {
        cv::Point2f(0., 0.),
        cv::Point2f(w+1., 0),
        cv::Point2f(w+1., h+1.),
        cv::Point2f(0., h+1)
    };
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
//    cout << "M" << M << endl;
    cv::Mat crop;
    cv::warpPerspective(ori_img, crop, M, cv::Size(w+3, h+1));
    crops.emplace_back(crop);
  }
  return 0;
}

int DBDetector::crop(cv::cuda::GpuMat &g_ori_img,
                     std::vector<std::vector<cv::Point2f>> order_points,
                     std::vector<cv::cuda::GpuMat> &crops)
{
  for (auto& or_points: order_points) {
    for (auto& point: or_points){
      point.x *= g_ori_img.cols;
      point.y *= g_ori_img.rows;
    }
    int w = static_cast<int>(sqrt(pow(or_points[3].x - or_points[2].x, 2)  + pow(or_points[3].y - or_points[2].y, 2)));
    int h = static_cast<int>(sqrt(pow(or_points[3].x - or_points[0].x, 2)  + pow(or_points[3].y - or_points[0].y, 2)));
//    cout << "order_points: " << or_points << endl;
//    cout << "w: " << w << ", h:" << h << endl;
    cv::Point2f src[] = {
        or_points[0],
        or_points[1],
        or_points[2],
        or_points[3]
    };
    cv::Point2f dst[]= {
        cv::Point2f(0., 0.),
        cv::Point2f(w+1., 0),
        cv::Point2f(w+1., h+1.),
        cv::Point2f(0., h+1)
    };
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
//    cout << "M" << M << endl;
    cv::cuda::GpuMat crop;
    cv::cuda::warpPerspective(g_ori_img, crop, M, cv::Size(w+3, h+1));
//    cv::warpPerspective(g_ori_img, crop, M, cv::Size(w+3, h+1));
    crops.emplace_back(crop);
  }
  return 0;
}

int DBDetector::map2ordered_points(cv::Mat &prob, std::vector<std::vector<cv::Point2f>> &order_points, bool unclip)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarcy;
  cv::findContours(prob, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

//  std::vector<cv::Rect> boundRect(contours.size());
  std::vector<cv::RotatedRect> box(contours.size());
  cv::Point2f rect[4];
  cout << "dbnet: Find contours: " << contours.size() << endl;
  for (int i = 0; i < contours.size(); i++) {
    std::vector<cv::Point> contour;
    if (cv::contourArea(contours[i]) < 0.0005 * (prob.rows * prob.cols))
    {
      cout << "dbnet: ignore an small contour contourArea: " << cv::contourArea(contours[i]) << "vs" << prob.cols*prob.rows << endl;
      continue;
    }

    if (unclip)
      contour = expandBox(contours[i], unclip_ratio);
    else
      contour = contours[i];
    box[i] = cv::minAreaRect(cv::Mat(contour));
    //boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
    box[i].points(rect);//把最小外接矩形四个端点复制给rect数组
    for (int j = 0; j < 4; j++) {
      rect[j].x = max(min(static_cast<int>(round(rect[j].x)),  prob.cols), 0);
      rect[j].y = max(min(static_cast<int>(round(rect[j].y)),  prob.rows), 0);
    }
    // find out the order
    /*def order_points_clockwise(pts):
          rect = np.zeros((4, 2), dtype="float32")
          s = pts.sum(axis=1)
          rect[0] = pts[np.argmin(s)]
          rect[2] = pts[np.argmax(s)]
          diff = np.diff(pts, axis=1)
          rect[1] = pts[np.argmin(diff)]
          rect[3] = pts[np.argmax(diff)]
          return rect
     * */
    std::vector<cv::Point2f> or_points;
    int id0 = 0; int id1=0; int id2 = 0;int id3 = 0;
    float sum_min = prob.cols + prob.rows; float sum_max = 0.; float dif_min = prob.cols + prob.rows; float dif_max = -(prob.cols + prob.rows);
    for (int k = 0; k < 4; k++) {
      float sum = rect[k].x + rect[k].y;
      float dif = rect[k].y - rect[k].x;
      if (sum < sum_min)
      {
        id0 = k;
        sum_min = sum;
      }
      if (sum > sum_max)
      {
        id2 = k;
        sum_max = sum;
      }
      if (dif < dif_min)
      {
        id1 = k;
        dif_min = dif;
      }
      if (dif > dif_max)
      {
        id3 = k;
        dif_max = dif;
      }
    }
    // TODO: filter wrong order!
    int ids[4] = {id0, id1, id2, id3};
    int n = unique(ids, ids + 4) - ids;
//    cout << "dbnet: order points length: " << n << endl;
    if (n != 4)
    {
      // Probably wrong order, use bbox instead!
      cout << id0 << id1 << id2 << id3 << endl;
      cout << "dbnet: points order probably wrong, use bbox instead!" << endl;
      cv::Rect bbox = cv::boundingRect(cv::Mat(contour));
      or_points.emplace_back(cv::Point2f(bbox.x/prob.cols, bbox.y/prob.rows));
      or_points.emplace_back(cv::Point2f((bbox.x + bbox.width)/prob.cols, bbox.y/prob.rows));
      or_points.emplace_back(cv::Point2f((bbox.x + bbox.width)/prob.cols, (bbox.y + bbox.height)/prob.rows));
      or_points.emplace_back(cv::Point2f(bbox.x/prob.cols, (bbox.y + bbox.height)/prob.rows));
    }else
      {
        or_points.emplace_back(cv::Point2f(rect[id0].x/prob.cols, rect[id0].y/prob.rows));
        or_points.emplace_back(cv::Point2f(rect[id1].x/prob.cols, rect[id1].y/prob.rows));
        or_points.emplace_back(cv::Point2f(rect[id2].x/prob.cols, rect[id2].y/prob.rows));
        or_points.emplace_back(cv::Point2f(rect[id3].x/prob.cols, rect[id3].y/prob.rows));
//    cout << "order points: " << or_points << endl;
      }
    order_points.emplace_back(or_points);
  }
  return 0;
}

std::vector<cv::Point> DBDetector::expandBox(std::vector<cv::Point> &inBox, float ratio)
{
  /*    def unclip(self, box, unclip_ratio=1.5):
            poly = Polygon(box)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            return expanded
   * */
  ClipperLib::Paths polys;
  ClipperLib::Path poly;
  for (auto& point : inBox)
  {
    poly.push_back(ClipperLib::IntPoint(point.x, point.y));
  }
  auto area = cv::contourArea(inBox);
  auto length =cv::arcLength(inBox, true);
  double distance = area * ratio / length;
  auto offset = ClipperLib::ClipperOffset();
  offset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
  polys.push_back(poly);
  offset.Execute(polys, distance);
  std::vector<cv::Point> expanded_points;
  for (auto point: polys[0])
  {
    expanded_points.emplace_back(cv::Point(point.X, point.Y));
  }
  return expanded_points;
}

int DBDetector::preprocess(const cv::Mat &img, void *tensorRTBuffer, int input_h, int input_w)
{
  int c = 3;
  int h = input_h;   //net h
  int w = input_w;   //net w

  auto scaleSize = cv::Size(w, h);

  cv::Mat resized_img, float_img;
  cv::resize(img, resized_img, scaleSize);
  resized_img.convertTo(float_img, CV_32FC3, 1.0f/255.0);

  //HWC TO CHW
  vector<cv::Mat> bgr(c);
  cv::split(float_img, bgr);

  void *dst_b = tensorRTBuffer + sizeof(float) * h * w * 0;
  void *dst_g = tensorRTBuffer + sizeof(float) * h * w * 1;
  void *dst_r = tensorRTBuffer + sizeof(float) * h * w * 2;

  cv::Mat bDst(h, w, CV_32FC1, dst_b);
  cv::subtract(bgr[0], 0.485, bDst);
  cv::multiply(bDst, 1 / 0.229, bDst);

  cv::Mat gDst(h, w, CV_32FC1, dst_g);
  cv::subtract(bgr[1], 0.456, gDst);
  cv::multiply(gDst, 1 / 0.224, gDst);

  cv::Mat rDst(h, w, CV_32FC1, dst_r);
  cv::subtract(bgr[2], 0.406, rDst);
  cv::multiply(rDst, 1 / 0.225, rDst);

//    std::cout << input_channels[0].at<float>(304, 304) << input_channels[0].at<float>(304, 305) << input_channels[0].at<float>(304, 306)<< std::endl;
//    float* recvCPU = static_cast<float*>(tensorRTBuffer);
//    std::cout << recvCPU[185136] << ", " << recvCPU[185137] << ", " <<recvCPU[185138] << endl;
//    std::cout << recvCPU[1108989] << ", " << recvCPU[1108990] << ", " <<recvCPU[1108991] << endl;
  return 0;
}

int DBDetector::preprocess(const cv::cuda::GpuMat& gpu_img, void* tensorRTBuffer, int input_h, int input_w)
{
  cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

  // keep aspect ratio resize
  cv::cuda::GpuMat resize_img, float_img;
  int img_w = gpu_img.cols;
  int img_h = gpu_img.rows;
  int h = input_h;
  int w = input_w;

  cv::cuda::resize(gpu_img, resize_img, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cv_stream);
  resize_img.convertTo(float_img, CV_32FC3, 1./ 255., cv_stream);

  std::vector<cv::cuda::GpuMat> bgr_GpuMat;
  cv::cuda::split(float_img, bgr_GpuMat);

  void *dst_b = tensorRTBuffer + sizeof(float) * h * w * 0;
  void *dst_g = tensorRTBuffer + sizeof(float) * h * w * 1;
  void *dst_r = tensorRTBuffer + sizeof(float) * h * w * 2;

  cv::cuda::GpuMat bDst(h, w, CV_32FC1, dst_b);
  cv::cuda::subtract(bgr_GpuMat[0], 0.485, bDst, cv::cuda::GpuMat(), -1, cv_stream);
  cv::cuda::multiply(bDst, 1 / 0.229, bDst, 1, -1, cv_stream);

  cv::cuda::GpuMat gDst(h, w, CV_32FC1, dst_g);
  cv::cuda::subtract(bgr_GpuMat[1], 0.456, gDst, cv::cuda::GpuMat(), -1, cv_stream);
  cv::cuda::multiply(gDst, 1 / 0.224, gDst, 1, -1, cv_stream);

  cv::cuda::GpuMat rDst(h, w, CV_32FC1, dst_r);
  cv::cuda::subtract(bgr_GpuMat[2], 0.406, rDst, cv::cuda::GpuMat(), -1, cv_stream);
  cv::cuda::multiply(rDst, 1 / 0.225, rDst, 1, -1, cv_stream);

//    cudaSetDevice(0);
//    float* recvCPU=(float*)malloc(DETECT_WIDTH * DETECT_HEIGHT * 3 *sizeof(float));  //将数据从cuda 拷贝到cpu
//    cudaMemcpy(recvCPU, tensorRTBuffer, sizeof(float) * DETECT_WIDTH * DETECT_HEIGHT * 3, cudaMemcpyDeviceToHost);
//    std::cout << recvCPU[185136] << ", " << recvCPU[185137] << ", " <<recvCPU[185138] << endl;
//    std::cout << recvCPU[1108989] << ", " << recvCPU[1108990] << ", " <<recvCPU[1108991] << endl;

  return 0;
}