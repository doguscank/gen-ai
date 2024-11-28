import abc

from pydantic import BaseModel, ConfigDict


class Output(abc.ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
