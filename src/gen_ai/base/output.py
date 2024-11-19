import abc

from pydantic import BaseModel


class Output(abc.ABC, BaseModel):
    pass
