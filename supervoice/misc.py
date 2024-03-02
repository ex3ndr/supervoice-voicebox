def dict_to_object(src):
    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = DictToObject(value)
                self.__dict__[key] = value

        def __repr__(self):
            return f"{self.__dict__}"
    return DictToObject(src)