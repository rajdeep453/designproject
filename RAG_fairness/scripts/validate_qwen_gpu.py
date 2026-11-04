"""Load the local Qwen2.5-0.5B-Instruct checkpoint and verify CUDA generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--output",
        default="output/environment_validation/qwen_gpu_generation.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to report GPU inference")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"model weights not found under {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    # The shipped sampling defaults are irrelevant for greedy generation and
    # otherwise produce warnings when do_sample=False.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats(0)

    messages = [
        {
            "role": "user",
            "content": "Answer in one short sentence: What is BM25 used for in a retrieval-augmented system?",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    answer = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    result = {
        "status": "passed",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_path": str(model_path),
        "revision": (
            (model_path / "source_revision.txt").read_text(encoding="utf-8").strip()
            if (model_path / "source_revision.txt").is_file()
            else getattr(model.config, "_commit_hash", None)
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
        "generation_seconds": elapsed,
        "answer": answer,
        "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
        "gpu_reserved_bytes": torch.cuda.memory_reserved(0),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
