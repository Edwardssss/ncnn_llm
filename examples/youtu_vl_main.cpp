// Youtu-VL example — text-only or image prefill + generate
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#include "youtu_vl.h"
#include "ncnn_text_runtime.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: youtu_vl_run <model_dir> [prompt] [image_path] [max_new_tokens]\n");
        return 1;
    }
    std::string model_dir = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Describe this image.";
    std::string image_path = argc > 3 ? argv[3] : "";
    int max_tokens = argc > 4 ? std::stoi(argv[4]) : 4096;

    printf("Loading model from %s\n", model_dir.c_str());
    ncnn_llm_youtu model(model_dir);

    GenerateConfig cfg;
    cfg.max_new_tokens = max_tokens;
    cfg.temperature = 0.1f;
    cfg.top_p = 0.001f;
    cfg.repetition_penalty = 1.05f;
    cfg.do_sample = 1;

    std::string output;

    if (!image_path.empty()) {
        printf("Image: %s\n", image_path.c_str());
        printf("Prompt: \"%s\"\n", prompt.c_str());
        output = model.run(prompt, image_path, cfg);
    } else {
        printf("Text prompt: \"%s\"\n", prompt.c_str());
        output = model.run_text(prompt, cfg);
    }

    std::cout << "--- Output ---\n" << output << "\n---\n";
    return 0;
}
