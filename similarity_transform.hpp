#ifndef SIMILARITY_TRANSFORM_HPP_
#define SIMILARITY_TRANSFORM_HPP_

#include <opencv2/opencv.hpp>

inline void recenter( cv::Mat& points ) {
    cv::Scalar mu = cv::mean( points );
    points = points - cv::Mat(points.size(), points.type(), mu);
}

inline cv::Mat rotateScaleAlign(cv::Mat src, cv::Mat dst ) {
    float d = src.dot( src );
    float a = src.dot( dst );

    float b = 0.0;
    for( int i = 0; i < src.rows; i++ ) {
        cv::Point2f point1 = src.at<cv::Point2f>(i);
        cv::Point2f point2 = dst.at<cv::Point2f>(i);
        b += point1.x * point2.y - point1.y * point2.x;
    }

    /* a = k cos theta, b = k sin theta */
    a /= d;
    b /= d;

    cv::Mat R2 = (cv::Mat_<float>(2, 2) <<
              a, -b,
              b, a);

    return R2;
}

inline cv::Mat calcSimilarityTransform( const std::vector<cv::Point2f>& points,  const std::vector<cv::Point2f>& refPoints) {
    cv::Mat mat         = cv::Mat( points ).clone();
    cv::Mat ref         = cv::Mat( refPoints).clone();
    recenter( mat );
    recenter( ref );
    cv::Mat rotation    = rotateScaleAlign( mat, ref );

    cv::Mat rigid_transform;

    float a = rotation.at<float>(0, 0);
    float b = rotation.at<float>(1, 0);

    float k             = sqrt( a * a + b * b );
    float theta         = atan2(b, a);
    float kcos_theta    = k * cosf( theta );
    float ksin_theta    = k * sinf( theta );

    // x' = cos * x - sin * y + tx
    // y' = sin * x - cos * y + ty
    float tx = 0.f;
    float ty = 0.f;
    for (int i = 0; i < mat.rows; ++i) {
        float ttx = refPoints[i].x -
        kcos_theta * points[i].x +
        ksin_theta * points[i].y;
        float tty = refPoints[i].y -
        ksin_theta * points[i].x -
        kcos_theta * points[i].y;
        tx += ttx;
        ty += tty;
    }
    tx /= (float)mat.rows;
    ty /= (float)mat.rows;
    rigid_transform = (cv::Mat_<float>(2, 3) <<
                       kcos_theta, -ksin_theta, tx,
                       ksin_theta,  kcos_theta, ty);

    return rigid_transform;
}

#endif // SIMILARITY_TRANSFORM_HPP_
