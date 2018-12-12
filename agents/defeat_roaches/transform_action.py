

def transform_action(action):
    """A method to transform an (x, y) target into a unique index, or visa versa."""

    # first, setup the size of the action space
    X = 121
    Y = 101

    # check some types
    assert type(action) in [int, tuple], 'action must be of type int or tuple.'

    # different rules for if integer or tuple
    if type(action) is int:
        x = action // Y
        y = action % Y
        return x, y
    elif type(action) is tuple:
        x, y = action
        return x * Y + y
