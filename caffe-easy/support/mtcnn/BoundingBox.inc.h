// This file includes functions dealing with bounding box in image processing.
//
// Code exploited by Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
//
// calcRect, modernPosit are copied from AFLW toolkit, which is available for non-commercial research purposes only.
#pragma once
#include <opencv2\opencv.hpp>
#include "mtcnn.h"

using namespace cv;
#undef assert
#define assert(_Expression) if(!(_Expression)) printf("error: %s %d, %s\n", __FILE__, __LINE__, #_Expression)
  struct FaceAndPoints {
    Mat image;
    std::vector<Point2d> points;
  };

  std::vector<cv::Point3f> worldPts = {
    cv::Point3f(-0.361, 0.309, -0.521),
    cv::Point3f(0.235, 0.296, -0.46),
    cv::Point3f(-0.071, -0.023, -0.13),
    cv::Point3f(-0.296, -0.348, -0.472),
    cv::Point3f(0.156, -0.366, -0.417)
  };

  inline double IoU(Rect2d rect1, Rect2d rect2) {
    double left = std::max<double>(rect1.x, rect2.x);
    double top = std::max<double>(rect1.y, rect2.y);
    double right = std::min<double>(rect1.x + rect1.width, rect2.x + rect2.width);
    double bottom = std::min<double>(rect1.y + rect1.height, rect2.y + rect2.height);
    double overlap = max<double>(right - left, 0) * max<double>(bottom - top, 0);
    return overlap / (rect1.width * rect1.height + rect2.width * rect2.height - overlap);
  }

  inline Rect2d BoundingBoxRegressionTarget(Rect2d data_rect, Rect2d ground_truth) {
    return Rect2d((ground_truth.x - data_rect.x) / data_rect.width,
                  (ground_truth.y - data_rect.y) / data_rect.height,
                  log(ground_truth.width / data_rect.width),
                  log(ground_truth.height / data_rect.height));
  }

  void modernPosit(cv::Mat &rot, cv::Point3f &trans, std::vector<cv::Point2d> imagePts, std::vector<cv::Point3f> worldPts, double focalLength, cv::Point2d center = cv::Point2d(0.0, 0.0), int maxIterations = 100) {
    int nbPoints = imagePts.size();

    std::vector<cv::Point2d>::iterator imagePtsIt;
    std::vector<cv::Point2d> centeredImage;
    for (imagePtsIt = imagePts.begin(); imagePtsIt != imagePts.end(); ++imagePtsIt)
      centeredImage.push_back(*imagePtsIt - center);
    for (imagePtsIt = centeredImage.begin(); imagePtsIt != centeredImage.end(); ++imagePtsIt) {
      imagePtsIt->x /= focalLength;
      imagePtsIt->y /= focalLength;
    }

    std::vector<double> ui(nbPoints);
    std::vector<double> vi(nbPoints);
    std::vector<double> oldUi(nbPoints);
    std::vector<double> oldVi(nbPoints);
    std::vector<double> deltaUi(nbPoints);
    std::vector<double> deltaVi(nbPoints);

    /*double ui[nbPoints];
    double vi[nbPoints];
    double oldUi[nbPoints];
    double oldVi[nbPoints];
    double deltaUi[nbPoints];
    double deltaVi[nbPoints];*/
    for (int i = 0; i < nbPoints; ++i) {
      ui[i] = centeredImage[i].x;
      vi[i] = centeredImage[i].y;
    }

    cv::Mat homogeneousWorldPts(nbPoints, 4, CV_32F);
    for (int i = 0; i < nbPoints; ++i) {
      cv::Point3f worldPoint = worldPts[i];
      homogeneousWorldPts.at<float>(i, 0) = worldPoint.x;
      homogeneousWorldPts.at<float>(i, 1) = worldPoint.y;
      homogeneousWorldPts.at<float>(i, 2) = worldPoint.z;
      homogeneousWorldPts.at<float>(i, 3) = 1; // homogeneous
    }

    cv::Mat objectMat;
    cv::invert(homogeneousWorldPts, objectMat, cv::DECOMP_SVD);

    /*
    std::cout << "objectmat: " << std::endl;
    for (int j = 0; j < 4; ++j)
    {
    for (int i = 0; i < nbPoints; ++i)
    std::cout << objectMat.at<float>(j,i) << " ";
    std::cout << std::endl;
    }
    */

    bool converged = false;
    int iterationCount = 0;

    double Tx = 0.0;
    double Ty = 0.0;
    double Tz = 0.0;
    double r1T[4];
    double r2T[4];
    double r1N[4];
    double r2N[4];
    double r3[4];

    while ((!converged) && ((maxIterations < 0) || (iterationCount < maxIterations))) {
      // r1T= objectMat * ui; % pose vectors
      // r2T = objectMat * vi;
      for (int j = 0; j < 4; ++j) {
        r1T[j] = 0;
        r2T[j] = 0;
        for (int i = 0; i < nbPoints; ++i) {
          r1T[j] += ui[i] * objectMat.at<float>(j, i);
          r2T[j] += vi[i] * objectMat.at<float>(j, i);
        }
      }

      // Tz1 = 1/sqrt(r1T(1)*r1T(1) + r1T(2)*r1T(2)+ r1T(3)*r1T(3)); % 1/Tz1 is norm of r1T
      // Tz2 = 1/sqrt(r2T(1)*r2T(1) + r2T(2)*r2T(2)+ r2T(3)*r2T(3)); % 1/Tz2 is norm of r2T
      double Tz1, Tz2;
      Tz1 = 1 / sqrt(r1T[0] * r1T[0] + r1T[1] * r1T[1] + r1T[2] * r1T[2]);
      Tz2 = 1 / sqrt(r2T[0] * r2T[0] + r2T[1] * r2T[1] + r2T[2] * r2T[2]);

      // Tz = sqrt(Tz1*Tz2); % geometric average instead of arithmetic average of classicPosit.m
      Tz = sqrt(Tz1*Tz2);

      for (int j = 0; j < 4; ++j) {
        r1N[j] = r1T[j] * Tz;
        r2N[j] = r2T[j] * Tz;
      }

      // DEBUG
      for (int j = 0; j < 3; ++j) {
        if ((r1N[j] > 1.0) || (r1N[j] < -1.0)) {
          //std::cout << "WARNING: r1N[" << j << "] == " << r1N[j] << std::endl;
          r1N[j] = std::max<double>(-1.0, std::min<double>(1.0, r1N[j]));
        }
        if ((r2N[j] > 1.0) || (r2N[j] < -1.0)) {
          //std::cout << "WARNING: r2N[" << j << "] == " << r2N[j] << std::endl;
          r2N[j] = std::max<double>(-1.0, std::min<double>(1.0, r2N[j]));
        }
      }

      // r1 = r1N(1:3);
      // r2 = r2N(1:3);
      // r3 = cross(r1,r2);
      // r3T= [r3; Tz];
      r3[0] = r1N[1] * r2N[2] - r1N[2] * r2N[1];
      r3[1] = r1N[2] * r2N[0] - r1N[0] * r2N[2];
      r3[2] = r1N[0] * r2N[1] - r1N[1] * r2N[0];
      r3[3] = Tz;

      Tx = r1N[3];
      Ty = r2N[3];

      // wi= homogeneousWorldPts * r3T /Tz;

      std::vector<double> wi(nbPoints);
      //double wi[nbPoints];

      for (int i = 0; i < nbPoints; ++i) {
        wi[i] = 0;
        for (int j = 0; j < 4; ++j)
          wi[i] += homogeneousWorldPts.at<float>(i, j) * r3[j] / Tz;
      }

      // oldUi = ui;
      // oldVi = vi;
      // ui = wi .* centeredImage(:,1)
      // vi = wi .* centeredImage(:,2)
      // deltaUi = ui - oldUi;
      // deltaVi = vi - oldVi;
      for (int i = 0; i < nbPoints; ++i) {
        oldUi[i] = ui[i];
        oldVi[i] = vi[i];
        ui[i] = wi[i] * centeredImage[i].x;
        vi[i] = wi[i] * centeredImage[i].y;
        deltaUi[i] = ui[i] - oldUi[i];
        deltaVi[i] = vi[i] - oldVi[i];
      }

      // delta = focalLength * focalLength * (deltaUi' * deltaUi + deltaVi' * deltaVi)
      double delta = 0.0;
      for (int i = 0; i < nbPoints; ++i)
        delta += deltaUi[i] * deltaUi[i] + deltaVi[i] * deltaVi[i];
      delta = delta*focalLength * focalLength;

      /*
      std::cout << "delta: " << delta << std::endl ;
      std::cout << "r1N: " << r1N[0] << " " << r1N[1] << " " << r1N[2] << " " << r1N[3] << std::endl;
      std::cout << "r2N: " << r2N[0] << " " << r2N[1] << " " << r2N[2] << " " << r2N[3] << std::endl;
      std::cout << "r1T: " << r1T[0] << " " << r1T[1] << " " << r1T[2] << " " << r1T[3] << std::endl;
      std::cout << "r2T: " << r2T[0] << " " << r2T[1] << " " << r2T[2] << " " << r2T[3] << std::endl;
      std::cout << "r3: " << r3[0] << " " << r3[1] << " " << r3[2] << " " << r3[3] << std::endl;
      */

      // converged = (count>0 & delta < 1)
      converged = (iterationCount > 0) && (delta < 0.01);
      ++iterationCount;

      //std::cout << "delta " << delta << std::endl;
    }
    // trans = [Tx; Ty; Tz];
    // rot = [r1'; r2'; r3'];
    //std::cout << "iter count " << iterationCount << std::endl;

    trans.x = Tx;
    trans.y = Ty;
    trans.z = Tz;

    rot.create(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i) {
      rot.at<float>(0, i) = r1N[i];
      rot.at<float>(1, i) = r2N[i];
      rot.at<float>(2, i) = r3[i];
    }
  }

  cv::Point2d project(cv::Point3f pt, double focalLength, cv::Point2d imgCenter) {
    cv::Point2d res;
    res.x = (pt.x / pt.z) * focalLength + imgCenter.x;
    res.y = (pt.y / pt.z) * focalLength + imgCenter.y;
    return res;
  }

  cv::Point3f transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans) {
    cv::Point3f res;
    res.x = rot.at<float>(0, 0)*pt.x + rot.at<float>(0, 1)*pt.y + rot.at<float>(0, 2)*pt.z + trans.x;
    res.y = rot.at<float>(1, 0)*pt.x + rot.at<float>(1, 1)*pt.y + rot.at<float>(1, 2)*pt.z + trans.y;
    res.z = rot.at<float>(2, 0)*pt.x + rot.at<float>(2, 1)*pt.y + rot.at<float>(2, 2)*pt.z + trans.z;
    return res;
  }

  bool calcCenterScaleAndUp(cv::Mat faceData, std::vector<cv::Point2d> imagePts, double normEyeDist, cv::Point2d &center, double &scale, cv::Point2d &upv) {

    double focalLength = static_cast<double>(faceData.cols)* 1.5;
    cv::Point2d imgCenter = cv::Point2d(static_cast<float>(faceData.cols) / 2.0f, static_cast<float>(faceData.rows) / 2.0f);

    cv::Mat rot;
    cv::Point3f trans;
    modernPosit(rot, trans, imagePts, worldPts, focalLength, imgCenter);

    // project center of the model to the image
    //cv::Point3f modelCenter = _meanFace3DModel.center_between_eyes();
    cv::Point3f modelCenter = cv::Point3f(-0.056, 0.3, -0.530);
    cv::Point3f rotatedCenter = transform(modelCenter, rot, trans);
    center = project(rotatedCenter, focalLength, imgCenter);

    // calc x and y extents of the projection of a sphere
    // centered at the model center (between eyes) and
    // with a diameter of the left to right eye distance
    double modelCenterDist = sqrt(rotatedCenter.x*rotatedCenter.x + rotatedCenter.y*rotatedCenter.y + rotatedCenter.z*rotatedCenter.z);
    double cameraModel3dYAngle = atan(rotatedCenter.y / sqrt(rotatedCenter.z*rotatedCenter.z + rotatedCenter.x*rotatedCenter.x));
    double cameraModel3dXAngle = atan(rotatedCenter.x / sqrt(rotatedCenter.z*rotatedCenter.z + rotatedCenter.y*rotatedCenter.y));
    double sphereCenterBorderAngle = asin(0.63 / 2.0 / modelCenterDist);
    double sphereProjTop = tan(cameraModel3dYAngle - sphereCenterBorderAngle) * focalLength;
    double sphereProjBottom = tan(cameraModel3dYAngle + sphereCenterBorderAngle) * focalLength;
    double sphereProjLeft = tan(cameraModel3dXAngle - sphereCenterBorderAngle) * focalLength;
    double sphereProjRight = tan(cameraModel3dXAngle + sphereCenterBorderAngle) * focalLength;

    scale = std::max<double>(abs(sphereProjRight - sphereProjLeft), abs(sphereProjBottom - sphereProjTop)) / normEyeDist;

    // up vector
    //if (!calcUpvFromEyes(upv)) {
    //  // cout << "upv from pose" << endl;
    //  //cv::Point3f modelCenterUp = _meanFace3DModel.center_between_eyes();
    //  cv::Point3f modelCenterUp = modelCenter;
    //  modelCenterUp.y += 0.5;
    //  cv::Point3f rotatedCenterUp = transform(modelCenterUp, rot, trans);
    //  cv::Point2d centerUp = project(rotatedCenterUp, focalLength, imgCenter);
    //  upv = centerUp - center;
    //  double upvlen = sqrt(upv.x*upv.x + upv.y*upv.y);
    //  upv.x /= upvlen;
    //  upv.y /= upvlen;
    //}
    cv::Point2d lrV = imagePts[1] - imagePts[0];
    double vlen = sqrt(lrV.x*lrV.x + lrV.y*lrV.y);
    upv.x = lrV.y / vlen;
    upv.y = -lrV.x / vlen;
    return true;
  }

  Rect2d calcRect(cv::Mat faceData, std::vector<cv::Point2d> imagePts) {
    //cout << "process face " << faceData->ID << endl;

    // normalized rect
    double faceRectWidth = 128;
    double faceRectHeight = 128;
    double normEyeDist = 50.0;              // distance between eyes in normalized rectangle
    //	cv::Point2d targetCenter(64.0f, 37.0f); // center within normalized rectangle
    cv::Point2d centerOffset(0.0f, 25.0f); // shift from CENTER_BETWEEN_EYES to rectangle center

    cv::Point2d center;
    cv::Point2d upv;
    double scale = 0.0;
    calcCenterScaleAndUp(faceData, imagePts, normEyeDist, center, scale, upv);

    /*
    int x = floor((center.x - targetCenter.x*scale) + 0.5);
    int y = floor((center.y - targetCenter.y*scale) + 0.5);
    */
    double w = faceRectWidth*scale;
    double h = faceRectHeight*scale;

    cv::Point2d rectCenter = center;
    rectCenter -= cv::Point2d(upv.x*centerOffset.y*scale, upv.y*centerOffset.y*scale);
    rectCenter -= cv::Point2d(upv.y*centerOffset.x*scale, -upv.x*centerOffset.x*scale);

    double x = rectCenter.x - faceRectWidth*scale / 2;
    double y = rectCenter.y - faceRectHeight*scale / 2;

    return Rect2d(x, y, w, h);
  }

  bool strict_weak_ordering(const std::pair<Rect2d, float> a, const std::pair<Rect2d, float> b) {
    return a.second < b.second;
  }

  void NMS(std::vector<std::pair<Rect2d, float>>& rects, double nms_threshold) {
    sort(rects.begin(), rects.end(), strict_weak_ordering);
    int remain = rects.size();
    do {
      auto best_rect = rects.end() - 1;
      remain--;
      for (auto rect = best_rect - 1; rect<rects.end(); rect++) {
        if (IoU(rect->first, best_rect->first) > nms_threshold) {
          rects.erase(rect);
          remain--;
        }
      }
    } while (remain > 0);
  }

  enum IoU_TYPE {
    IoU_UNION,
    IoU_MIN,
    IoU_MAX
  };

  std::vector<int> nms_max(std::vector<std::pair<Rect2d, float>>& rects, double overlap, IoU_TYPE type = IoU_UNION) {
    const int n = rects.size();
    std::vector<double> areas(n);

    typedef std::multimap<double, int> ScoreMapper;
    ScoreMapper map;
    for (int i = 0; i < n; i++) {
      map.insert(ScoreMapper::value_type(rects[i].second, i));
      areas[i] = rects[i].first.width*rects[i].first.height;
    }

    int picked_n = 0;
    std::vector<int> picked(n);
    while (map.size() != 0) {
      auto last_item = map.rbegin();
      int last = map.rbegin()->second; // get the index of maximum score value
      //std::cout << map.rbegin()->first << " " << last << std::endl;
      picked[picked_n] = last;
      picked_n++;

      for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
        int idx = it->second;
        if (idx == last) {
          ScoreMapper::iterator tmp = it;
          tmp++;
          map.erase(it);
          it = tmp;
          continue;
        }
        double x1 = std::max<double>(rects[idx].first.x, rects[last].first.x);
        double x2 = std::min<double>(rects[idx].first.x + rects[idx].first.width, rects[last].first.x + rects[last].first.width);
        double w = x2 - x1;
        if (w <= 0) {
          it++; continue;
        }
        double y1 = std::max<double>(rects[idx].first.y, rects[last].first.y);
        double y2 = std::min<double>(rects[idx].first.y + rects[idx].first.height, rects[last].first.y + rects[last].first.height);
        double h = y2 - y1;
        if (h <= 0) {
          it++; continue;
        }
        double ov;
        switch (type) {
        case IoU_MAX:
          ov = w*h / max(areas[idx], areas[last]);
          break;
        case IoU_MIN:
          ov = w*h / min(areas[idx], areas[last]);
          break;
        case IoU_UNION:
        default:
          ov = w*h / (areas[idx] + areas[last] - w*h);
          break;
        }
        
        if (ov > overlap) {
          ScoreMapper::iterator tmp = it;
          tmp++;
          map.erase(it);
          it = tmp;
        }
        else {
          it++;
        }
      }
    }

    picked.resize(picked_n);
    return picked;
  }

  std::vector<int> nms_avg(std::vector<std::pair<Rect2d, float>>& rects, double overlap) {
    const int n = rects.size();
    std::vector<double> areas(n);

    typedef std::multimap<double, int> ScoreMapper;
    ScoreMapper map;
    for (int i = 0; i < n; i++) {
      map.insert(ScoreMapper::value_type(rects[i].second, i));
      areas[i] = rects[i].first.width*rects[i].first.height;
    }

    int picked_n = 0;
    std::vector<int> picked(n);
    while (map.size() != 0) {
      int last = map.rbegin()->second; // get the index of maximum score value
      picked[picked_n] = last;
      picked_n++;

      int overlap_count = 1;
      double mean_x = rects[last].first.x, mean_y = rects[last].first.y, mean_w = log(rects[last].first.width), mean_h = log(rects[last].first.height);
      //double mean_x = rects[last].first.x, mean_y = rects[last].first.y, mean_w = rects[last].first.width, mean_h = rects[last].first.height;

      for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
        int idx = it->second;
        double x1 = std::max<double>(rects[idx].first.x, rects[last].first.x);
        double y1 = std::max<double>(rects[idx].first.y, rects[last].first.y);
        double x2 = std::min<double>(rects[idx].first.x + rects[idx].first.width, rects[last].first.x + rects[last].first.width);
        double y2 = std::min<double>(rects[idx].first.y + rects[idx].first.height, rects[last].first.y + rects[last].first.height);
        double w = std::max<double>(0., x2 - x1);
        double h = std::max<double>(0., y2 - y1);
        double ov = w*h / (areas[idx] + areas[last] - w*h);
        if (ov > overlap) {
          ScoreMapper::iterator tmp = it;
          tmp++;
          map.erase(it);
          it = tmp;
          if (rects[idx].second > rects[last].second * 0.9) {
            mean_x += rects[idx].first.x;
            mean_y += rects[idx].first.y;
            mean_w += log(rects[idx].first.width);
            mean_h += log(rects[idx].first.height);
            //mean_w += rects[idx].first.width;
            //mean_h += rects[idx].first.height;
            overlap_count++;
          }
        }
        else {
          it++;
        }
      }
      rects[last].first.x = mean_x / overlap_count;
      rects[last].first.y = mean_y / overlap_count;
      rects[last].first.width = exp(mean_w / overlap_count);
      rects[last].first.height = exp(mean_h / overlap_count);
      //rects[last].first.width = mean_w / overlap_count;
      //rects[last].first.height = mean_h / overlap_count;
    }

    picked.resize(picked_n);
    return picked;
  }

  inline bool checkRect(Rect2d rect, Size image_size) {
    if (rect.x < 0 || rect.y < 0 || rect.width <= 0 || rect.height <= 0 ||
        (rect.x + rect.width) > (image_size.width - 1) || (rect.y + rect.height) > (image_size.height - 1))
      return false;
    else return true;
  }
  inline bool fixRect(Rect2d& rect, Size image_size, bool only_center = false) {
    if (rect.width <= 0 || rect.height <= 0) return false;
    if (only_center) {
      Point2d center = Point2d(rect.x + rect.width / 2, rect.y + rect.height / 2);
      center.x = max<double>(center.x, 0.0);
      center.y = max<double>(center.y, 0.0);
      center.x = min<double>(center.x, image_size.width - 1);
      center.y = min<double>(center.y, image_size.height - 1);
      rect.x = center.x - rect.width / 2;
      rect.y = center.y - rect.height / 2;
    }
    else {
      rect.x = max<double>(rect.x, 0.0);
      rect.y = max<double>(rect.y, 0.0);
      rect.width = min<double>(rect.width, image_size.width - 1 - rect.x);
      rect.height = min<double>(rect.height, image_size.height - 1 - rect.y);
    }
    return true;
  }

  inline void make_rect_square(Rect2d& rect) {
    double max_len = max(rect.width, rect.height);
    rect.x += rect.width / 2 - max_len / 2;
    rect.y += rect.height / 2 - max_len / 2;
    rect.width = rect.height = max_len;
  }

  double IoU(Rect2d rect, RotatedRect ellip, Size image_size = Size(0,0)) {
    Rect2d baseRect,ellip_br = ellip.boundingRect();
    baseRect.x = min(rect.x, ellip_br.x);
    baseRect.y = min(rect.y, ellip_br.y);
    baseRect.width = max(rect.x + rect.width - baseRect.x, ellip_br.x + ellip_br.width - baseRect.x);
    baseRect.height = max(rect.y + rect.height - baseRect.y, ellip_br.y + ellip_br.height - baseRect.y);
    baseRect.x -= 10;
    baseRect.y -= 10;
    baseRect.width += 20;
    baseRect.height += 20;
    if (image_size.width != 0) fixRect(baseRect,image_size);
    rect.x -= baseRect.x; rect.y -= baseRect.y;
    ellip.center.x -= baseRect.x; ellip.center.y -= baseRect.y;
    Mat rect_image = Mat::zeros(Size(baseRect.width, baseRect.height), CV_8UC1);
    Mat ellipse_image = Mat::zeros(Size(baseRect.width, baseRect.height), CV_8UC1);
    rectangle(rect_image, rect, Scalar(255), -1);
    ellipse(ellipse_image, ellip, Scalar(255), -1);
    Mat Overlap = rect_image.mul(ellipse_image);
    //imshow("rect", rect_image);
    //imshow("ellipse", ellipse_image);
    //imshow("overlap", Overlap);
    //waitKey(0);
    double rect_size = countNonZero(rect_image);//rect.width * rect.height;
    double ellipse_size = countNonZero(ellipse_image);
    double overlap = countNonZero(Overlap);
    return overlap / (rect_size + ellipse_size - overlap);
  }

  /** @brief Applies an affine transformation to an image.

  The function warpAffine transforms the source image using the specified matrix:

  \f[\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\f]

  when the flag WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
  with cv::invertAffineTransform and then put in the formula above instead of M. The function cannot
  operate in-place.

  @param src input image.
  @param dst output image that has the size dsize and the same type as src .
  @param M \f$2\times 3\f$ transformation matrix.
  @param dsize size of the output image.
  @param flags combination of interpolation methods (see cv::InterpolationFlags) and the optional
  flag WARP_INVERSE_MAP that means that M is the inverse transformation (
  \f$\texttt{dst}\rightarrow\texttt{src}\f$ ).
  @param borderMode pixel extrapolation method (see cv::BorderTypes); when
  borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
  the "outliers" in the source image are not modified by the function.
  @param borderValue value used in case of a constant border; by default, it is 0.

  @sa  warpPerspective, resize, remap, getRectSubPix, transform
  */
  Mat cropImage(const Mat& input_image, Rect2d roi, Size2d target_size, int flags = 1, int borderMode = 0, Scalar& borderValue = Scalar(0)) {
    //Point2f srcPoints[4] = { Point2f(roi.x, roi.y), Point2f(roi.x + roi.width,roi.y), Point2f(roi.x,roi.y + roi.height),Point2f(roi.x + roi.width,roi.y + roi.height) };
    //Point2f dstPoints[4] = { Point2f(0,0), Point2f(target_size.width,0),Point2f(0, target_size.height), Point2f(target_size.width, target_size.height) };
    //Mat M2 = getAffineTransform(srcPoints, dstPoints);
    //std::cout << M2 << std::endl;
    Mat M = (Mat_<float>(2, 3) << target_size.width / roi.width, 0, -roi.x*target_size.width / roi.width, 0, target_size.height / roi.height, -roi.y*target_size.height / roi.height);
    //std::cout << M << std::endl;
    Mat result;
    warpAffine(input_image, result, M, target_size, flags, borderMode, borderValue);
    return result;
  }

  //Only support 2 scaling
  Mat getPyramidStitchingImage2(Mat& input_image, std::vector<std::pair<Rect, double>>& location_and_scale, double scaling = 0.707,
                                Scalar background_color = Scalar(0,0,0), int min_side = 12, int interval = 2) {
    using namespace std;
    bool stitch_x = input_image.cols < input_image.rows;
    Size current_size = input_image.size();
    Point left_top = Point(0, 0);
    int width, height;
    if (stitch_x) {
      width = ceil(input_image.cols * (1 + scaling)) + interval * 2;
      height = 0;
      map<int, pair<int, int>> height_index; // (width_start, width, height)
      height_index[0] = pair<int, int>(width, 0);
      do {
        int min_h = INT_MAX, min_start = 0;
        for (auto h : height_index) {
          if (h.second.first > current_size.width + interval && h.second.second < min_h) {
            min_h =  h.second.second;
            min_start = h.first;
          }
        }
        location_and_scale.push_back(make_pair(Rect(min_start, height_index[min_start].second, current_size.width, current_size.height),
          (double)current_size.height / (double)input_image.rows));
        height_index[min_start + current_size.width + interval] = height_index[min_start];
        height_index[min_start + current_size.width + interval].first -= current_size.width + interval;
        height_index[min_start].first = current_size.width;
        height_index[min_start].second += current_size.height + interval;
        if (height_index[min_start].second > height) height = height_index[min_start].second;
        //for (auto h : height_index) {
        //  cout << h.first << " " << h.second.first << " " << h.second.second << endl;
        //}
        //cout << "===================="<<endl;
        current_size.width *= scaling;
        current_size.height *= scaling;
        } while (current_size.width > min_side);
        height += interval;
    }
    else {
      height = ceil(input_image.rows * (1 + scaling)) + interval * 2;
      width = 0;
      map<int, pair<int, int>> width_index; // (height_start, height, width)
      width_index[0] = pair<int, int>(height, 0);
      do {
        int min_w = INT_MAX, min_start = 0;
        for (auto w : width_index) {
          if (w.second.first > current_size.height + interval && w.second.second < min_w) {
            min_w = w.second.second;
            min_start = w.first;
          }
        }
        location_and_scale.push_back(make_pair(Rect(width_index[min_start].second, min_start, current_size.width, current_size.height),
          (double)current_size.width / (double)input_image.cols));
        width_index[min_start + current_size.height + interval] = width_index[min_start];
        width_index[min_start + current_size.height + interval].first -= current_size.height + interval;
        width_index[min_start].first = current_size.height;
        width_index[min_start].second += current_size.width + interval;
        if (width_index[min_start].second > width) width = width_index[min_start].second;
        //for (auto h : width_index) {
        //  cout << h.first << " " << h.second.first << " " << h.second.second << endl;
        //}
        //cout << "====================" << endl;
        current_size.width *= scaling;
        current_size.height *= scaling;
      } while (current_size.height > min_side);
      width += interval;
    }

    Mat big_image = Mat::zeros(height, width, input_image.type());
    big_image = background_color;
    Mat resized_image = input_image;
    //std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
//#pragma omp parallel for // not very fast.
    for (auto ls : location_and_scale) {
      resize(resized_image, resized_image, Size(ls.first.width, ls.first.height));
      resized_image.copyTo(big_image(ls.first));
    }
    //cout << "resize time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t0).count() / 1000 << "ms" << endl;
    return big_image;
  }
