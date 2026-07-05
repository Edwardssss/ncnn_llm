// Youtu-VL example — text-only or image prefill + generate
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#include "youtu_vl.h"
#include "ncnn_text_runtime.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: youtu_vl_run <model_dir> [prompt] [image_path]\n");
        return 1;
    }
    std::string model_dir = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Describe this image.";
    std::string image_path = argc > 3 ? argv[3] : "";

    fprintf(stderr, "[youtu_vl_run] Loading model from %s\n", model_dir.c_str());
    ncnn_llm_youtu model(model_dir);

    GenerateConfig cfg;
    cfg.max_new_tokens = argc > 3 ? 30 : 30;
    cfg.temperature = 0.0f;
    cfg.do_sample = 0;

    std::string output;
    auto cb = [&output](const std::string& token) {
        output += token;
    };

    if (!image_path.empty()) {
        fprintf(stderr, "[youtu_vl_run] Image: %s\n", image_path.c_str());
        fprintf(stderr, "[youtu_vl_run] Prompt: \"%s\"\n", prompt.c_str());
        output = model.run(prompt, image_path, cfg);
    } else {
        fprintf(stderr, "[youtu_vl_run] Prefill: \"%s\"\n", prompt.c_str());
        auto ctx = model.prefill(prompt);
        fprintf(stderr, "[youtu_vl_run] First token: %d\n", ctx->cur_token);
        fprintf(stderr, "[youtu_vl_run] Generating...\n");
        ctx = model.generate(ctx, cfg, cb);
    }

    std::cout << "--- Output ---\n" << output << "\n---\n";
    return 0;
}
