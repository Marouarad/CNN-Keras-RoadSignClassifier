#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 30
#define N_INPUT_2_1 30
#define N_INPUT_3_1 3
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 32
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 32
#define OUT_HEIGHT_4 22
#define OUT_WIDTH_4 22
#define N_FILT_4 32
#define OUT_HEIGHT_4 22
#define OUT_WIDTH_4 22
#define N_FILT_4 32
#define OUT_HEIGHT_6 11
#define OUT_WIDTH_6 11
#define N_FILT_6 32
#define OUT_HEIGHT_7 9
#define OUT_WIDTH_7 9
#define N_FILT_7 64
#define OUT_HEIGHT_7 9
#define OUT_WIDTH_7 9
#define N_FILT_7 64
#define OUT_HEIGHT_9 7
#define OUT_WIDTH_9 7
#define N_FILT_9 64
#define OUT_HEIGHT_9 7
#define OUT_WIDTH_9 7
#define N_FILT_9 64
#define OUT_HEIGHT_11 3
#define OUT_WIDTH_11 3
#define N_FILT_11 64
#define N_SIZE_0_12 576
#define N_LAYER_13 256
#define N_LAYER_13 256
#define N_LAYER_15 43
#define N_LAYER_15 43

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> conv2d_weight_t;
typedef ap_fixed<16,6> conv2d_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> conv2d_relu_table_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> conv2d_1_weight_t;
typedef ap_fixed<16,6> conv2d_1_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<18,8> conv2d_1_relu_table_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> conv2d_2_weight_t;
typedef ap_fixed<16,6> conv2d_2_bias_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<18,8> conv2d_2_relu_table_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> conv2d_3_weight_t;
typedef ap_fixed<16,6> conv2d_3_bias_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<18,8> conv2d_3_relu_table_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<18,8> dense_relu_table_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> dense_1_weight_t;
typedef ap_fixed<16,6> dense_1_bias_t;
typedef ap_uint<1> layer15_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> dense_1_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> dense_1_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> dense_1_softmax_inv_table_t;

#endif
