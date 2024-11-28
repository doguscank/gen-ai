from gen_ai.configs.defaults.text_gen import qwen_qwq as qwen_qwq_cfg
from gen_ai.tasks.text_gen.qwen_qwq import (
    QwenQwQInput,
    QwenQwQModel,
    QwenQwQModelConfig,
)

model_config = QwenQwQModelConfig(
    repo_id=qwen_qwq_cfg.QWEN_QWQ_Q3_K_S_MODEL_ID,
    filename=qwen_qwq_cfg.QWEN_QWQ_Q3_K_S_MODEL_FILENAME,
    cache_dir=qwen_qwq_cfg.CACHE_DIR,
    n_gpu_layers=40,
    n_ctx=32768,
    flash_attn=True,
    verbose=False,
)

input = QwenQwQInput(
    prompt="Steam enters a long, horizontal pipe with an inlet diameter of D1 = 16 cm at 2 MPa and 300°C with a velocity of 2.5 m/s. Farther downstream, the conditions are 1.8 MPa and 250°C, and the diameter is D2 = 14 cm. Determine (a) the mass flow rate of the steam and (b) the rate of heat transfer",
    stream=True,
    system_prompt="You are a helpful assistant that can answer questions about everything. You should think step-by-step and provide detailed answers.",
)

model = QwenQwQModel(config=model_config)

output = model(input)

print(output.response)

# llama = Llama.from_pretrained(
#     repo_id=qwen_qwq_cfg.QWEN_QWQ_Q3_K_S_MODEL_ID,
#     filename=qwen_qwq_cfg.QWEN_QWQ_Q3_K_S_MODEL_FILENAME,
#     cache_dir=qwen_qwq_cfg.CACHE_DIR,
#     n_gpu_layers=40,
#     n_ctx=32768,
#     flash_attn=True,
#     verbose=True,
# )

# prompt = "Steam enters a long, horizontal pipe with an inlet diameter of D1 = 16 cm at 2 MPa and 300°C with a velocity of 2.5 m/s. Farther downstream, the conditions are 1.8 MPa and 250°C, and the diameter is D2 = 14 cm. Determine (a) the mass flow rate of the steam and (b) the rate of heat transfer"

# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant that can answer questions about everything. You should think step-by-step and provide detailed answers.",
#     },
#     {"role": "user", "content": prompt},
# ]

# response = llama.create_chat_completion(
#     messages=messages,
#     temperature=0.9,
#     min_p=0.1,
#     top_p=0.8,
#     top_k=20,
#     stream=True,
#     seed=-1,
# )

# result_text = ""

# for chunk in response:
#     if "content" in chunk["choices"][0]["delta"]:
#         text_piece = chunk["choices"][0]["delta"]["content"]
#         print(text_piece, end="")
#         result_text += text_piece

# print(result_text)
