from pathlib import Path

CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "transformers_cache"
    / "llama_mesh"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_MESH_MODEL_ID = "Zhengyi/LLaMA-Mesh"

LLAMA_MESH_F16_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_F16_MODEL_FILENAME = "LLaMA-Mesh-f16.gguf"

LLAMA_MESH_Q8_0_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q8_0_MODEL_FILENAME = "LLaMA-Mesh-Q8_0.gguf"

LLAMA_MESH_Q6_K_L_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q6_K_L_MODEL_FILENAME = "LLaMA-Mesh-Q6_K_L.gguf"

LLAMA_MESH_Q6_K_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q6_K_MODEL_FILENAME = "LLaMA-Mesh-Q6_K.gguf"

LLAMA_MESH_Q5_K_L_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q5_K_L_MODEL_FILENAME = "LLaMA-Mesh-Q5_K_L.gguf"

LLAMA_MESH_Q5_K_M_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q5_K_M_MODEL_FILENAME = "LLaMA-Mesh-Q5_K_M.gguf"

LLAMA_MESH_Q5_K_S_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q5_K_S_MODEL_FILENAME = "LLaMA-Mesh-Q5_K_S.gguf"

LLAMA_MESH_Q4_K_L_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_K_L_MODEL_FILENAME = "LLaMA-Mesh-Q4_K_L.gguf"

LLAMA_MESH_Q4_K_M_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_K_M_MODEL_FILENAME = "LLaMA-Mesh-Q4_K_M.gguf"

LLAMA_MESH_Q3_K_XL_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q3_K_XL_MODEL_FILENAME = "LLaMA-Mesh-Q3_K_XL.gguf"

LLAMA_MESH_Q4_K_S_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_K_S_MODEL_FILENAME = "LLaMA-Mesh-Q4_K_S.gguf"

LLAMA_MESH_Q4_0_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_0_MODEL_FILENAME = "LLaMA-Mesh-Q4_0.gguf"

LLAMA_MESH_Q4_0_8_8_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_0_8_8_MODEL_FILENAME = "LLaMA-Mesh-Q4_0_8_8.gguf"

LLAMA_MESH_Q4_0_4_8_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_0_4_8_MODEL_FILENAME = "LLaMA-Mesh-Q4_0_4_8.gguf"

LLAMA_MESH_Q4_0_4_4_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q4_0_4_4_MODEL_FILENAME = "LLaMA-Mesh-Q4_0_4_4.gguf"

LLAMA_MESH_IQ4_XS_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_IQ4_XS_MODEL_FILENAME = "LLaMA-Mesh-IQ4_XS.gguf"

LLAMA_MESH_Q3_K_L_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q3_K_L_MODEL_FILENAME = "LLaMA-Mesh-Q3_K_L.gguf"

LLAMA_MESH_Q3_K_M_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q3_K_M_MODEL_FILENAME = "LLaMA-Mesh-Q3_K_M.gguf"

LLAMA_MESH_IQ3_M_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_IQ3_M_MODEL_FILENAME = "LLaMA-Mesh-IQ3_M.gguf"

LLAMA_MESH_Q2_K_L_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q2_K_L_MODEL_FILENAME = "LLaMA-Mesh-Q2_K_L.gguf"

LLAMA_MESH_Q3_K_S_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q3_K_S_MODEL_FILENAME = "LLaMA-Mesh-Q3_K_S.gguf"

LLAMA_MESH_IQ3_XS_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_IQ3_XS_MODEL_FILENAME = "LLaMA-Mesh-IQ3_XS.gguf"

LLAMA_MESH_Q2_K_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_Q2_K_MODEL_FILENAME = "LLaMA-Mesh-Q2_K.gguf"

LLAMA_MESH_IQ2_M_MODEL_ID = "bartowski/LLaMA-Mesh-GGUF"
LLAMA_MESH_IQ2_M_MODEL_FILENAME = "LLaMA-Mesh-IQ2_M.gguf"

LLAMA_MESH_DESC = """
| Filename                  | Quant Type | Description                                                                                     |
|---------------------------|------------|-------------------------------------------------------------------------------------------------|
| LLaMA-Mesh-f16.gguf       | f16        | Full F16 weights.                                                                              |
| LLaMA-Mesh-Q8_0.gguf      | Q8_0       | Extremely high quality, generally unneeded but max available quant.                           |
| LLaMA-Mesh-Q6_K_L.gguf    | Q6_K_L     | Uses Q8_0 for embed and output weights. Very high quality, near perfect, recommended.         |
| LLaMA-Mesh-Q6_K.gguf      | Q6_K       | Very high quality, near perfect, recommended.                                                 |
| LLaMA-Mesh-Q5_K_L.gguf    | Q5_K_L     | Uses Q8_0 for embed and output weights. High quality, recommended.                            |
| LLaMA-Mesh-Q5_K_M.gguf    | Q5_K_M     | High quality, recommended.                                                                    |
| LLaMA-Mesh-Q5_K_S.gguf    | Q5_K_S     | High quality, recommended.                                                                    |
| LLaMA-Mesh-Q4_K_L.gguf    | Q4_K_L     | Uses Q8_0 for embed and output weights. Good quality, recommended.                            |
| LLaMA-Mesh-Q4_K_M.gguf    | Q4_K_M     | Good quality, default size for most use cases, recommended.                                    |
| LLaMA-Mesh-Q3_K_XL.gguf   | Q3_K_XL    | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability.|
| LLaMA-Mesh-Q4_K_S.gguf    | Q4_K_S     | Slightly lower quality with more space savings, recommended.                                   |
| LLaMA-Mesh-Q4_0.gguf      | Q4_0       | Legacy format, generally not worth using over similarly sized formats.                        |
| LLaMA-Mesh-Q4_0_8_8.gguf  | Q4_0_8_8   | Optimized for ARM and AVX inference. Requires 'sve' support for ARM (see details below).       |
| LLaMA-Mesh-Q4_0_4_8.gguf  | Q4_0_4_8   | Optimized for ARM inference. Requires 'i8mm' support (see details below). Don't use on Mac.    |
| LLaMA-Mesh-Q4_0_4_4.gguf  | Q4_0_4_4   | Optimized for ARM inference. Should work well on all ARM chips, not for use with GPUs.         |
| LLaMA-Mesh-IQ4_XS.gguf    | IQ4_XS     | Decent quality, smaller than Q4_K_S with similar performance, recommended.                    |
| LLaMA-Mesh-Q3_K_L.gguf    | Q3_K_L     | Lower quality but usable, good for low RAM availability.                                       |
| LLaMA-Mesh-Q3_K_M.gguf    | Q3_K_M     | Low quality.                                                                                   |
| LLaMA-Mesh-IQ3_M.gguf     | IQ3_M      | Medium-low quality, new method with decent performance comparable to Q3_K_M.                  |
| LLaMA-Mesh-Q2_K_L.gguf    | Q2_K_L     | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable.              |
| LLaMA-Mesh-Q3_K_S.gguf    | Q3_K_S     | Low quality, not recommended.                                                                 |
| LLaMA-Mesh-IQ3_XS.gguf    | IQ3_XS     | Lower quality, new method with decent performance, slightly better than Q3_K_S.               |
| LLaMA-Mesh-Q2_K.gguf      | Q2_K       | Very low quality but surprisingly usable.                                                     |
| LLaMA-Mesh-IQ2_M.gguf     | IQ2_M      | Relatively low quality, uses SOTA techniques to be surprisingly usable.                       |
"""
