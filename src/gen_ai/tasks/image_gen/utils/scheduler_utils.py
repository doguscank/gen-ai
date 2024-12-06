from typing import Dict

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    SchedulerMixin,
    TCDScheduler,
    UniPCMultistepScheduler,
)

from gen_ai.constants.diffusion_noise_scheduler_types import SchedulerTypes

SCHEDULER_TYPE_TO_CLASS: Dict[SchedulerTypes, SchedulerMixin] = {
    SchedulerTypes.DPMPP_2M: DPMSolverMultistepScheduler,
    SchedulerTypes.DPMPP_2M_KARRAS: DPMSolverMultistepScheduler,
    SchedulerTypes.DPMPP_2M_SDE: DPMSolverMultistepScheduler,
    SchedulerTypes.DPMPP_2M_SDE_KARRAS: DPMSolverMultistepScheduler,
    SchedulerTypes.DPMPP_SDE: DPMSolverSinglestepScheduler,
    SchedulerTypes.DPMPP_SDE_KARRAS: DPMSolverSinglestepScheduler,
    SchedulerTypes.DPM2: KDPM2DiscreteScheduler,
    SchedulerTypes.DPM2_KARRAS: KDPM2DiscreteScheduler,
    SchedulerTypes.DPM2_A: KDPM2AncestralDiscreteScheduler,
    SchedulerTypes.DPM2_A_KARRAS: KDPM2AncestralDiscreteScheduler,
    SchedulerTypes.EULER: EulerDiscreteScheduler,
    SchedulerTypes.EULER_A: EulerAncestralDiscreteScheduler,
    SchedulerTypes.HEUN: HeunDiscreteScheduler,
    SchedulerTypes.LMS: LMSDiscreteScheduler,
    SchedulerTypes.LMS_KARRAS: LMSDiscreteScheduler,
    SchedulerTypes.DEIS: DEISMultistepScheduler,
    SchedulerTypes.UNIPC: UniPCMultistepScheduler,
    SchedulerTypes.TCD: TCDScheduler,
}

SCHEDULER_TYPE_TO_ARGS: Dict[SchedulerTypes, Dict] = {
    SchedulerTypes.DPMPP_2M_KARRAS: {"use_karras_sigmas": True},
    SchedulerTypes.DPMPP_2M_SDE: {"algorithm_type": "sde-dpmsolver++"},
    SchedulerTypes.DPMPP_2M_SDE_KARRAS: {
        "use_karras_sigmas": True,
        "algorithm_type": "sde-dpmsolver++",
    },
    SchedulerTypes.DPMPP_SDE_KARRAS: {"use_karras_sigmas": True},
    SchedulerTypes.DPM2_KARRAS: {"use_karras_sigmas": True},
    SchedulerTypes.DPM2_A_KARRAS: {"use_karras_sigmas": True},
    SchedulerTypes.LMS_KARRAS: {"use_karras_sigmas": True},
}


def get_scheduler(
    scheduler_type: SchedulerTypes,
) -> SchedulerMixin:
    """
    Get the scheduler class based on the scheduler type.

    Parameters
    ----------
    scheduler_type : SchedulerTypes
        The scheduler type.

    Returns
    -------
    SchedulerMixin
        The scheduler class.
    """

    if scheduler_type not in SCHEDULER_TYPE_TO_CLASS:
        raise NotImplementedError(
            f"Scheduler type {scheduler_type} is not implemented."
        )

    scheduler_class = SCHEDULER_TYPE_TO_CLASS[scheduler_type]
    scheduler_args = SCHEDULER_TYPE_TO_ARGS.get(scheduler_type, {})
    return scheduler_class(**scheduler_args)
