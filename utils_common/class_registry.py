import inspect
import typing
import omegaconf
import dataclasses


class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, arg_keys):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [
            arg
            for arg in args
            if (arg[0] != "self" and arg[0] != "args" and arg[0] != "kwargs")
        ]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.classes.items()
            ],
        )

    def make_dataclass_from_args(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.args.items()
            ],
        )

    def add_to_registry(self, name, arg_keys=None):
        def add_class_by_name(cls):
            self.classes[name] = cls
            if inspect.isfunction(cls):
                self.args[name] = self.make_dataclass_from_init(
                    cls, name, arg_keys
                )
            elif inspect.isclass(cls):
                self.args[name] = self.make_dataclass_from_init(
                    cls.__init__, name, arg_keys
                )

            return cls
        return add_class_by_name

    def add_func_to_registry(self, name):
        def add_class_by_name(func):
            self.classes[name] = func
            self.args[name] = self.make_dataclass_from_init(
                func, name, None
            )

            return func
        return add_class_by_name
