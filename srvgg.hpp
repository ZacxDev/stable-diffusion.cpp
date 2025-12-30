#ifndef __SRVGG_HPP__
#define __SRVGG_HPP__

#include "ggml_extend.hpp"

/*
    ===================================    SRVGGNetCompact  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
    https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan

    SRVGGNetCompact is a compact VGG-style super-resolution network used by:
    - realesr-animevideov3 (num_conv=16, ~1.5M params, fast)
    - realesr-general-x4v3 (num_conv=32, ~3M params, balanced)

    Architecture:
        conv_first (3->64) + PReLU
        body: num_conv * (Conv2d(64->64) + PReLU)
        conv_last (64->48 for 4x)
        PixelShuffle(4)
        + upscaled input (residual)
*/

class SRVGGNetCompact : public GGMLBlock {
protected:
    int num_in_ch  = 3;
    int num_out_ch = 3;
    int num_feat   = 64;
    int num_conv   = 16;  // 16 for animevideov3, 32 for general-x4v3
    int upscale    = 4;

public:
    SRVGGNetCompact(int scale     = 4,
                    int num_conv  = 16,
                    int num_in_ch = 3,
                    int num_out_ch = 3,
                    int num_feat  = 64)
        : upscale(scale),
          num_conv(num_conv),
          num_in_ch(num_in_ch),
          num_out_ch(num_out_ch),
          num_feat(num_feat) {
        // conv_first + prelu_first
        blocks["conv_first"]  = std::shared_ptr<GGMLBlock>(new Conv2d(num_in_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["prelu_first"] = std::shared_ptr<GGMLBlock>(new PReLU(num_feat));

        // body: num_conv pairs of (conv + prelu)
        for (int i = 0; i < num_conv; i++) {
            std::string conv_name  = "body_conv." + std::to_string(i);
            std::string prelu_name = "body_prelu." + std::to_string(i);
            blocks[conv_name]      = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
            blocks[prelu_name]     = std::shared_ptr<GGMLBlock>(new PReLU(num_feat));
        }

        // conv_last: output channels = num_out_ch * upscale^2 for PixelShuffle
        int out_ch         = num_out_ch * upscale * upscale;
        blocks["conv_last"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, out_ch, {3, 3}, {1, 1}, {1, 1}));
    }

    int get_scale() { return upscale; }
    int get_num_conv() { return num_conv; }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, num_in_ch, H, W] in PyTorch order -> [W, H, num_in_ch, N] in ggml
        // return: [W*scale, H*scale, num_out_ch, N] in ggml order

        // Save input for residual connection
        auto input = x;

        // conv_first + prelu_first
        auto conv_first  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_first"]);
        auto prelu_first = std::dynamic_pointer_cast<PReLU>(blocks["prelu_first"]);
        x                = conv_first->forward(ctx, x);
        x                = prelu_first->forward(ctx, x);

        // body convolutions with PReLU activations
        for (int i = 0; i < num_conv; i++) {
            auto conv  = std::dynamic_pointer_cast<Conv2d>(blocks["body_conv." + std::to_string(i)]);
            auto prelu = std::dynamic_pointer_cast<PReLU>(blocks["body_prelu." + std::to_string(i)]);
            x          = conv->forward(ctx, x);
            x          = prelu->forward(ctx, x);
        }

        // conv_last (no activation)
        auto conv_last = std::dynamic_pointer_cast<Conv2d>(blocks["conv_last"]);
        x              = conv_last->forward(ctx, x);

        // PixelShuffle upscaling: [W, H, C*r*r, N] -> [W*r, H*r, C, N]
        x = ggml_ext_pixel_shuffle(ctx->ggml_ctx, x, upscale);

        // Residual: add nearest-neighbor upscaled input
        auto upscaled_input = ggml_upscale(ctx->ggml_ctx, input, upscale, GGML_SCALE_MODE_NEAREST);
        x                   = ggml_add(ctx->ggml_ctx, x, upscaled_input);

        return x;
    }
};

#endif  // __SRVGG_HPP__
