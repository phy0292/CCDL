#include <opencv2/opencv.hpp>

namespace FaceInception {
  std::vector<cv::Point3f> worldPts = {
    cv::Point3f(-0.361, 0.309, -0.521),
    cv::Point3f(0.235, 0.296, -0.46),
    cv::Point3f(-0.071, -0.023, -0.13),
    cv::Point3f(-0.296, -0.348, -0.472),
    cv::Point3f(0.156, -0.366, -0.417)
  };

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

  cv::Rect2d calcRect(cv::Mat faceData, std::vector<cv::Point2d> imagePts) {
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

    return cv::Rect2d(x, y, w, h);
  }
}