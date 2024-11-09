from enum import Enum


class SchedulerTypes(Enum):
    """Enum class for different diffusion noise scheduler types.

    https://huggingface.co/docs/diffusers/en/api/schedulers/overview

    Maps user-friendly scheduler names to their corresponding Diffusers library implementations:
    - DPM++ 2M: DPMSolverMultistepScheduler
    - DPM++ 2M Karras: DPMSolverMultistepScheduler with use_karras_sigmas=True
    - DPM++ 2M SDE: DPMSolverMultistepScheduler with algorithm_type="sde-dpmsolver++"
    - DPM++ 2M SDE Karras: DPMSolverMultistepScheduler with use_karras_sigmas=True and algorithm_type="sde-dpmsolver++"
    - DPM++ 2S a: Custom implementation similar to DPMSolverSinglestepScheduler
    - DPM++ 2S a Karras: Custom implementation similar to DPMSolverSinglestepScheduler with use_karras_sigmas=True
    - DPM++ SDE: DPMSolverSinglestepScheduler
    - DPM++ SDE Karras: DPMSolverSinglestepScheduler with use_karras_sigmas=True
    - DPM2: KDPM2DiscreteScheduler
    - DPM2 Karras: KDPM2DiscreteScheduler with use_karras_sigmas=True
    - DPM2 a: KDPM2AncestralDiscreteScheduler
    - DPM2 a Karras: KDPM2AncestralDiscreteScheduler with use_karras_sigmas=True
    - DPM adaptive: Custom implementation
    - DPM fast: Custom implementation
    - Euler: EulerDiscreteScheduler
    - Euler a: EulerAncestralDiscreteScheduler
    - Heun: HeunDiscreteScheduler
    - LMS: LMSDiscreteScheduler
    - LMS Karras: LMSDiscreteScheduler with use_karras_sigmas=True
    - DEIS: DEISMultistepScheduler
    - UniPC: UniPCMultistepScheduler
    """

    DPMPP_2M = "DPM++ 2M"
    DPMPP_2M_KARRAS = "DPM++ 2M Karras"
    DPMPP_2M_SDE = "DPM++ 2M SDE"
    DPMPP_2M_SDE_KARRAS = "DPM++ 2M SDE Karras"
    DPMPP_2S_A = "DPM++ 2S a"
    DPMPP_2S_A_KARRAS = "DPM++ 2S a Karras"
    DPMPP_SDE = "DPM++ SDE"
    DPMPP_SDE_KARRAS = "DPM++ SDE Karras"
    DPM2 = "DPM2"
    DPM2_KARRAS = "DPM2 Karras"
    DPM2_A = "DPM2 a"
    DPM2_A_KARRAS = "DPM2 a Karras"
    DPM_ADAPTIVE = "DPM adaptive"
    DPM_FAST = "DPM fast"
    EULER = "Euler"
    EULER_A = "Euler a"
    HEUN = "Heun"
    LMS = "LMS"
    LMS_KARRAS = "LMS Karras"
    DEIS = "DEIS"
    UNIPC = "UniPC"
