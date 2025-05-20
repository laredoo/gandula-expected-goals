def parser_params(func):
    def wrapper(*args, **kwargs):
        expected_return = func(*args, **kwargs)
        if expected_return:
            return {key: value.split(" ") for key, value in expected_return.items()}

    return wrapper
