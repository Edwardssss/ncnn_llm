#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

ncnn::Mat load_image_to_ncnn_mat(const std::string& image_path) {
    int w, h, c;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!data) {
        return ncnn::Mat();
    }

    ncnn::Mat bgr(w, h, 3, (size_t)1u);
    unsigned char* dst = (unsigned char*)bgr.data;
    
    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = data[i * 3 + 2];
        dst[i * 3 + 1] = data[i * 3 + 1];
        dst[i * 3 + 2] = data[i * 3 + 0];
    }

    stbi_image_free(data);
    return bgr;
}

ncnn::Mat ncnn_mat_resize(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    
    ncnn::resize_bilinear_c3((const unsigned char*)src.data, src.w, src.h, src.w * 3,
                              (unsigned char*)dst.data, target_w, target_h, target_w * 3);

    return dst;
}

bool ncnn_mat_empty(const ncnn::Mat& mat) {
    return mat.empty() || mat.w <= 0 || mat.h <= 0;
}