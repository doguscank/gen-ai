import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
