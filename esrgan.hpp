#ifndef __ESRGAN_HPP__
#define __ESRGAN_HPP__

#include "ggml_extend.hpp"
#include "model.h"
#include "srvgg.hpp"

// Upscaler architecture types
enum class UpscalerArchitecture {
    RRDB_NET,   // RealESRGAN_x4plus, anime_6B (existing, ~16M params)
    SRVGG_NET,  // realesr-animevideov3, general-x4v3 (fast, ~1.5-3M params)
    UNKNOWN
};

// Detect upscaler architecture from tensor names
// RRDBNet: Contains "RDB", "rdb", or "model.1.sub." patterns
// SRVGGNet: Sequential body.N.weight pattern without RDB, no rdb blocks
inline UpscalerArchitecture detect_upscaler_architecture(const std::vector<std::string>& tensor_names) {
    bool has_rdb       = false;
    bool has_body      = false;
    int max_body_idx   = -1;

    for (const auto& name : tensor_names) {
        // RRDBNet indicators: contains RDB block references
        if (name.find("RDB") != std::string::npos ||
            name.find("rdb") != std::string::npos ||
            name.find("model.1.sub.") != std::string::npos) {
            has_rdb = true;
        }

        // SRVGGNet pattern: body.N.weight without RDB
        // body.0 = conv_first, body.1 = prelu_first, body.2 = body_conv.0, etc.
        if (name.find("body.") == 0) {
            has_body       = true;
            size_t dot_pos = name.find('.', 5);
            if (dot_pos != std::string::npos) {
                try {
                    int idx      = std::stoi(name.substr(5, dot_pos - 5));
                    max_body_idx = std::max(max_body_idx, idx);
                } catch (...) {
                }
            }
        }
    }

    // RRDBNet takes priority if RDB patterns found
    if (has_rdb) {
        return UpscalerArchitecture::RRDB_NET;
    }

    // SRVGGNet: has body.N pattern without RDB, and body indices > 2
    // (body.0=conv_first, body.1=prelu, body.2+=body convs)
    if (has_body && max_body_idx >= 2 && !has_rdb) {
        return UpscalerArchitecture::SRVGG_NET;
    }

    return UpscalerArchitecture::UNKNOWN;
}

// Detect SRVGGNet parameters from tensor names
// Returns: {num_conv, upscale}
// num_conv = (max_body_idx - 1) / 2  (body.0=conv_first, body.1=prelu, then pairs)
// upscale derived from conv_last output channels: channels = 3 * r * r
inline std::pair<int, int> detect_srvgg_params(const std::vector<std::string>& tensor_names,
                                                const std::function<std::vector<int64_t>(const std::string&)>& get_shape) {
    int max_body_idx = -1;
    int upscale      = 4;  // default

    for (const auto& name : tensor_names) {
        if (name.find("body.") == 0) {
            size_t dot_pos = name.find('.', 5);
            if (dot_pos != std::string::npos) {
                try {
                    int idx      = std::stoi(name.substr(5, dot_pos - 5));
                    max_body_idx = std::max(max_body_idx, idx);
                } catch (...) {
                }
            }
        }
    }

    // num_conv: subtract conv_first(0) and prelu_first(1), then divide by 2 (conv+prelu pairs)
    // max_body_idx = 1 + num_conv*2 + 1 (for conv_last)
    // So: num_conv = (max_body_idx - 2) / 2
    int num_conv = (max_body_idx >= 2) ? (max_body_idx - 2) / 2 : 16;

    // Detect upscale from conv_last output channels
    // conv_last.weight has shape [out_ch, in_ch, kH, kW]
    // out_ch = num_out_ch * upscale^2 = 3 * r^2
    std::string conv_last_name = "body." + std::to_string(max_body_idx) + ".weight";
    if (get_shape) {
        auto shape = get_shape(conv_last_name);
        if (shape.size() >= 1) {
            int out_ch = static_cast<int>(shape[0]);
            // out_ch = 3 * r^2, solve for r
            if (out_ch == 12)
                upscale = 2;  // 3*2*2
            else if (out_ch == 27)
                upscale = 3;  // 3*3*3
            else if (out_ch == 48)
                upscale = 4;  // 3*4*4
        }
    }

    return {num_conv, upscale};
}

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

class ResidualDenseBlock : public GGMLBlock {
protected:
    int num_feat;
    int num_grow_ch;

public:
    ResidualDenseBlock(int num_feat = 64, int num_grow_ch = 32)
        : num_feat(num_feat), num_grow_ch(num_grow_ch) {
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv3"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv4"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv5"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 4 * num_grow_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* lrelu(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        return ggml_leaky_relu(ctx->ggml_ctx, x, 0.2f, true);
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
        auto conv3 = std::dynamic_pointer_cast<Conv2d>(blocks["conv3"]);
        auto conv4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv4"]);
        auto conv5 = std::dynamic_pointer_cast<Conv2d>(blocks["conv5"]);

        auto x1    = lrelu(ctx, conv1->forward(ctx, x));
        auto x_cat = ggml_concat(ctx->ggml_ctx, x, x1, 2);
        auto x2    = lrelu(ctx, conv2->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x2, 2);
        auto x3    = lrelu(ctx, conv3->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x3, 2);
        auto x4    = lrelu(ctx, conv4->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x4, 2);
        auto x5    = conv5->forward(ctx, x_cat);

        x5 = ggml_add(ctx->ggml_ctx, ggml_scale(ctx->ggml_ctx, x5, 0.2f), x);
        return x5;
    }
};

class RRDB : public GGMLBlock {
public:
    RRDB(int num_feat, int num_grow_ch = 32) {
        blocks["rdb1"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb2"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb3"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto rdb1 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb1"]);
        auto rdb2 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb2"]);
        auto rdb3 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb3"]);

        auto out = rdb1->forward(ctx, x);
        out      = rdb2->forward(ctx, out);
        out      = rdb3->forward(ctx, out);

        out = ggml_add(ctx->ggml_ctx, ggml_scale(ctx->ggml_ctx, out, 0.2f), x);
        return out;
    }
};

class RRDBNet : public GGMLBlock {
protected:
    int scale       = 4;
    int num_block   = 23;
    int num_in_ch   = 3;
    int num_out_ch  = 3;
    int num_feat    = 64;
    int num_grow_ch = 32;

public:
    RRDBNet(int scale, int num_block, int num_in_ch, int num_out_ch, int num_feat, int num_grow_ch)
        : scale(scale), num_block(num_block), num_in_ch(num_in_ch), num_out_ch(num_out_ch), num_feat(num_feat), num_grow_ch(num_grow_ch) {
        blocks["conv_first"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_in_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
        for (int i = 0; i < num_block; i++) {
            std::string name = "body." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new RRDB(num_feat, num_grow_ch));
        }
        blocks["conv_body"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        if (scale >= 2) {
            blocks["conv_up1"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        }
        if (scale == 4) {
            blocks["conv_up2"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        }
        blocks["conv_hr"]   = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_last"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_out_ch, {3, 3}, {1, 1}, {1, 1}));
    }

    int get_scale() { return scale; }
    int get_num_block() { return num_block; }

    struct ggml_tensor* lrelu(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        return ggml_leaky_relu(ctx->ggml_ctx, x, 0.2f, true);
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [n, num_in_ch, h, w]
        // return: [n, num_out_ch, h*scale, w*scale]
        auto conv_first = std::dynamic_pointer_cast<Conv2d>(blocks["conv_first"]);
        auto conv_body  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_body"]);
        auto conv_hr    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_hr"]);
        auto conv_last  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_last"]);

        auto feat      = conv_first->forward(ctx, x);
        auto body_feat = feat;
        for (int i = 0; i < num_block; i++) {
            std::string name = "body." + std::to_string(i);
            auto block       = std::dynamic_pointer_cast<RRDB>(blocks[name]);

            body_feat = block->forward(ctx, body_feat);
        }
        body_feat = conv_body->forward(ctx, body_feat);
        feat      = ggml_add(ctx->ggml_ctx, feat, body_feat);
        // upsample
        if (scale >= 2) {
            auto conv_up1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up1"]);
            feat          = lrelu(ctx, conv_up1->forward(ctx, ggml_upscale(ctx->ggml_ctx, feat, 2, GGML_SCALE_MODE_NEAREST)));
            if (scale == 4) {
                auto conv_up2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up2"]);
                feat          = lrelu(ctx, conv_up2->forward(ctx, ggml_upscale(ctx->ggml_ctx, feat, 2, GGML_SCALE_MODE_NEAREST)));
            }
        }
        // for all scales
        auto out = conv_last->forward(ctx, lrelu(ctx, conv_hr->forward(ctx, feat)));
        return out;
    }
};

struct ESRGAN : public GGMLRunner {
    UpscalerArchitecture arch_type = UpscalerArchitecture::UNKNOWN;
    std::unique_ptr<RRDBNet> rrdb_net;
    std::unique_ptr<SRVGGNetCompact> srvgg_net;
    int scale     = 4;
    int tile_size = 128;  // avoid cuda OOM for 4gb VRAM

    ESRGAN(ggml_backend_t backend,
           bool offload_params_to_cpu,
           int tile_size                                  = 128,
           const String2TensorStorage& tensor_storage_map = {})
        : GGMLRunner(backend, offload_params_to_cpu) {
        this->tile_size = tile_size;
    }

    std::string get_desc() override {
        if (arch_type == UpscalerArchitecture::SRVGG_NET) {
            return "srvgg";
        }
        return "esrgan";
    }

    UpscalerArchitecture get_arch_type() const { return arch_type; }

    // Load SRVGGNetCompact model with weight name mapping
    // Weight format: body.0=conv_first, body.1=prelu_first, body.2/3=body_conv/prelu.0, etc.
    bool load_srvgg_from_loader(ModelLoader& model_loader, const std::vector<std::string>& tensor_names, int n_threads) {
        // Get shape helper for parameter detection using tensor storage map
        auto& tensor_storage_map = model_loader.get_tensor_storage_map();
        auto get_shape           = [&tensor_storage_map](const std::string& name) -> std::vector<int64_t> {
            auto it = tensor_storage_map.find(name);
            if (it != tensor_storage_map.end()) {
                std::vector<int64_t> shape;
                for (int i = 0; i < it->second.n_dims; i++) {
                    shape.push_back(it->second.ne[i]);
                }
                return shape;
            }
            return {};
        };

        // Detect parameters
        auto [detected_num_conv, detected_scale] = detect_srvgg_params(tensor_names, get_shape);
        LOG_INFO("detected SRVGGNetCompact: num_conv=%d, scale=%d", detected_num_conv, detected_scale);

        // Create SRVGGNetCompact
        srvgg_net = std::make_unique<SRVGGNetCompact>(detected_scale, detected_num_conv, 3, 3, 64);
        srvgg_net->init(params_ctx, {}, "");

        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> srvgg_tensors;
        srvgg_net->get_param_tensors(srvgg_tensors);

        // Build name mapping: internal name -> file name
        // body.0 = conv_first, body.1 = prelu_first
        // body.2 = body_conv.0, body.3 = body_prelu.0
        // ...
        // body.N = conv_last (N = 2 + num_conv*2)
        std::map<std::string, std::string> name_mapping;
        name_mapping["conv_first.weight"]  = "body.0.weight";
        name_mapping["conv_first.bias"]    = "body.0.bias";
        name_mapping["prelu_first.weight"] = "body.1.weight";

        for (int i = 0; i < detected_num_conv; i++) {
            int conv_idx  = 2 + i * 2;
            int prelu_idx = 3 + i * 2;
            name_mapping["body_conv." + std::to_string(i) + ".weight"]  = "body." + std::to_string(conv_idx) + ".weight";
            name_mapping["body_conv." + std::to_string(i) + ".bias"]    = "body." + std::to_string(conv_idx) + ".bias";
            name_mapping["body_prelu." + std::to_string(i) + ".weight"] = "body." + std::to_string(prelu_idx) + ".weight";
        }

        int conv_last_idx                 = 2 + detected_num_conv * 2;
        name_mapping["conv_last.weight"]  = "body." + std::to_string(conv_last_idx) + ".weight";
        name_mapping["conv_last.bias"]    = "body." + std::to_string(conv_last_idx) + ".bias";

        // Map internal tensor names to file tensor names
        std::map<std::string, ggml_tensor*> file_tensors;
        for (auto& p : srvgg_tensors) {
            auto it = name_mapping.find(p.first);
            if (it != name_mapping.end()) {
                file_tensors[it->second] = p.second;
            } else {
                LOG_WARN("unmapped tensor: %s", p.first.c_str());
            }
        }

        bool success = model_loader.load_tensors(file_tensors, {}, n_threads);
        if (!success) {
            LOG_ERROR("load srvgg tensors from model loader failed");
            return false;
        }

        scale = srvgg_net->get_scale();
        LOG_INFO("SRVGGNetCompact model loaded with scale=%d, num_conv=%d", scale, detected_num_conv);
        return true;
    }

    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading esrgan from '%s'", file_path.c_str());

        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) {
            LOG_ERROR("init esrgan model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        // Get tensor names
        auto tensor_names = model_loader.get_tensor_names();

        // Detect architecture type first
        arch_type = detect_upscaler_architecture(tensor_names);

        // Handle SRVGGNet models
        if (arch_type == UpscalerArchitecture::SRVGG_NET) {
            return load_srvgg_from_loader(model_loader, tensor_names, n_threads);
        }

        // Fall back to RRDBNet loading (original code path)
        arch_type = UpscalerArchitecture::RRDB_NET;

        // Detect if it's ESRGAN format
        bool is_ESRGAN = std::find(tensor_names.begin(), tensor_names.end(), "model.0.weight") != tensor_names.end();

        // Detect parameters from tensor names
        int detected_num_block = 0;
        if (is_ESRGAN) {
            for (const auto& name : tensor_names) {
                if (name.find("model.1.sub.") == 0) {
                    size_t first_dot = name.find('.', 12);
                    if (first_dot != std::string::npos) {
                        size_t second_dot = name.find('.', first_dot + 1);
                        if (second_dot != std::string::npos && name.substr(first_dot + 1, 3) == "RDB") {
                            try {
                                int idx            = std::stoi(name.substr(12, first_dot - 12));
                                detected_num_block = std::max(detected_num_block, idx + 1);
                            } catch (...) {
                            }
                        }
                    }
                }
            }
        } else {
            // Original format
            for (const auto& name : tensor_names) {
                if (name.find("body.") == 0) {
                    size_t pos = name.find('.', 5);
                    if (pos != std::string::npos) {
                        try {
                            int idx            = std::stoi(name.substr(5, pos - 5));
                            detected_num_block = std::max(detected_num_block, idx + 1);
                        } catch (...) {
                        }
                    }
                }
            }
        }

        int detected_scale = 4;  // default
        if (is_ESRGAN) {
            // For ESRGAN format, detect scale by highest model number
            int max_model_num = 0;
            for (const auto& name : tensor_names) {
                if (name.find("model.") == 0) {
                    size_t dot_pos = name.find('.', 6);
                    if (dot_pos != std::string::npos) {
                        try {
                            int num       = std::stoi(name.substr(6, dot_pos - 6));
                            max_model_num = std::max(max_model_num, num);
                        } catch (...) {
                        }
                    }
                }
            }
            if (max_model_num <= 4) {
                detected_scale = 1;
            } else if (max_model_num <= 7) {
                detected_scale = 2;
            } else {
                detected_scale = 4;
            }
        } else {
            // Original format
            bool has_conv_up2 = std::any_of(tensor_names.begin(), tensor_names.end(), [](const std::string& name) {
                return name == "conv_up2.weight";
            });
            bool has_conv_up1 = std::any_of(tensor_names.begin(), tensor_names.end(), [](const std::string& name) {
                return name == "conv_up1.weight";
            });
            if (has_conv_up2) {
                detected_scale = 4;
            } else if (has_conv_up1) {
                detected_scale = 2;
            } else {
                detected_scale = 1;
            }
        }

        int detected_num_in_ch   = 3;
        int detected_num_out_ch  = 3;
        int detected_num_feat    = 64;
        int detected_num_grow_ch = 32;

        // Create RRDBNet with detected parameters
        rrdb_net = std::make_unique<RRDBNet>(detected_scale, detected_num_block, detected_num_in_ch, detected_num_out_ch, detected_num_feat, detected_num_grow_ch);
        rrdb_net->init(params_ctx, {}, "");

        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> esrgan_tensors;
        rrdb_net->get_param_tensors(esrgan_tensors);

        bool success;
        if (is_ESRGAN) {
            // Build name mapping for ESRGAN format
            std::map<std::string, std::string> expected_to_model;
            expected_to_model["conv_first.weight"] = "model.0.weight";
            expected_to_model["conv_first.bias"]   = "model.0.bias";

            for (int i = 0; i < detected_num_block; i++) {
                for (int j = 1; j <= 3; j++) {
                    for (int k = 1; k <= 5; k++) {
                        std::string expected_weight        = "body." + std::to_string(i) + ".rdb" + std::to_string(j) + ".conv" + std::to_string(k) + ".weight";
                        std::string model_weight           = "model.1.sub." + std::to_string(i) + ".RDB" + std::to_string(j) + ".conv" + std::to_string(k) + ".0.weight";
                        expected_to_model[expected_weight] = model_weight;

                        std::string expected_bias        = "body." + std::to_string(i) + ".rdb" + std::to_string(j) + ".conv" + std::to_string(k) + ".bias";
                        std::string model_bias           = "model.1.sub." + std::to_string(i) + ".RDB" + std::to_string(j) + ".conv" + std::to_string(k) + ".0.bias";
                        expected_to_model[expected_bias] = model_bias;
                    }
                }
            }

            if (detected_scale == 1) {
                expected_to_model["conv_body.weight"] = "model.1.sub." + std::to_string(detected_num_block) + ".weight";
                expected_to_model["conv_body.bias"]   = "model.1.sub." + std::to_string(detected_num_block) + ".bias";
                expected_to_model["conv_hr.weight"]   = "model.2.weight";
                expected_to_model["conv_hr.bias"]     = "model.2.bias";
                expected_to_model["conv_last.weight"] = "model.4.weight";
                expected_to_model["conv_last.bias"]   = "model.4.bias";
            } else {
                expected_to_model["conv_body.weight"] = "model.1.sub." + std::to_string(detected_num_block) + ".weight";
                expected_to_model["conv_body.bias"]   = "model.1.sub." + std::to_string(detected_num_block) + ".bias";
                if (detected_scale >= 2) {
                    expected_to_model["conv_up1.weight"] = "model.3.weight";
                    expected_to_model["conv_up1.bias"]   = "model.3.bias";
                }
                if (detected_scale == 4) {
                    expected_to_model["conv_up2.weight"]  = "model.6.weight";
                    expected_to_model["conv_up2.bias"]    = "model.6.bias";
                    expected_to_model["conv_hr.weight"]   = "model.8.weight";
                    expected_to_model["conv_hr.bias"]     = "model.8.bias";
                    expected_to_model["conv_last.weight"] = "model.10.weight";
                    expected_to_model["conv_last.bias"]   = "model.10.bias";
                } else if (detected_scale == 2) {
                    expected_to_model["conv_hr.weight"]   = "model.5.weight";
                    expected_to_model["conv_hr.bias"]     = "model.5.bias";
                    expected_to_model["conv_last.weight"] = "model.7.weight";
                    expected_to_model["conv_last.bias"]   = "model.7.bias";
                }
            }

            std::map<std::string, ggml_tensor*> model_tensors;
            for (auto& p : esrgan_tensors) {
                auto it = expected_to_model.find(p.first);
                if (it != expected_to_model.end()) {
                    model_tensors[it->second] = p.second;
                }
            }

            success = model_loader.load_tensors(model_tensors, {}, n_threads);
        } else {
            success = model_loader.load_tensors(esrgan_tensors, {}, n_threads);
        }

        if (!success) {
            LOG_ERROR("load esrgan tensors from model loader failed");
            return false;
        }

        scale = rrdb_net->get_scale();
        LOG_INFO("esrgan model loaded with scale=%d, num_block=%d", scale, detected_num_block);
        return success;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x) {
        // Check that we have a valid network
        if (arch_type == UpscalerArchitecture::SRVGG_NET) {
            if (!srvgg_net)
                return nullptr;
        } else {
            if (!rrdb_net)
                return nullptr;
        }

        constexpr int kGraphNodes = 1 << 16;  // 65k
        struct ggml_cgraph* gf    = new_graph_custom(kGraphNodes);
        x                         = to_backend(x);

        auto runner_ctx         = get_context();
        struct ggml_tensor* out = nullptr;

        // Dispatch based on architecture type
        if (arch_type == UpscalerArchitecture::SRVGG_NET) {
            out = srvgg_net->forward(&runner_ctx, x);
        } else {
            out = rrdb_net->forward(&runner_ctx, x);
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(const int n_threads,
                 struct ggml_tensor* x,
                 ggml_tensor** output,
                 ggml_context* output_ctx = nullptr) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x);
        };
        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __ESRGAN_HPP__