#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer16_out[N_LAYER_15]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer16_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv2d_weight_t, 2400>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv2d_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2d_1_weight_t, 25600>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv2d_1_bias_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv2d_2_weight_t, 18432>(w7, "w7.txt");
        nnet::load_weights_from_txt<conv2d_2_bias_t, 64>(b7, "b7.txt");
        nnet::load_weights_from_txt<conv2d_3_weight_t, 36864>(w9, "w9.txt");
        nnet::load_weights_from_txt<conv2d_3_bias_t, 64>(b9, "b9.txt");
        nnet::load_weights_from_txt<dense_weight_t, 147456>(w13, "w13.txt");
        nnet::load_weights_from_txt<dense_bias_t, 256>(b13, "b13.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 11008>(w15, "w15.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 43>(b15, "b15.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, layer2_t, config2>(input_1, layer2_out, w2, b2); // conv2d

    layer3_t layer3_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv2d_relu

    layer4_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::conv_2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // conv2d_1

    layer5_t layer5_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // conv2d_1_relu

    layer6_t layer6_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::pooling2d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out); // max_pooling2d

    layer7_t layer7_out[OUT_HEIGHT_7*OUT_WIDTH_7*N_FILT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::conv_2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // conv2d_2

    layer8_t layer8_out[OUT_HEIGHT_7*OUT_WIDTH_7*N_FILT_7];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out); // conv2d_2_relu

    layer9_t layer9_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::conv_2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9); // conv2d_3

    layer10_t layer10_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<layer9_t, layer10_t, relu_config10>(layer9_out, layer10_out); // conv2d_3_relu

    layer11_t layer11_out[OUT_HEIGHT_11*OUT_WIDTH_11*N_FILT_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::pooling2d_cl<layer10_t, layer11_t, config11>(layer10_out, layer11_out); // max_pooling2d_1

    auto& layer12_out = layer11_out;
    layer13_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer11_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // dense

    layer14_t layer14_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::relu<layer13_t, layer14_t, relu_config14>(layer13_out, layer14_out); // dense_relu

    layer15_t layer15_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::dense<layer14_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // dense_1

    nnet::softmax<layer15_t, result_t, softmax_config16>(layer15_out, layer16_out); // dense_1_softmax

}
