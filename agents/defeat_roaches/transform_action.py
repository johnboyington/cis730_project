

def transform_action(action):
    """A method to transform an (x, y) target into a unique index, or visa versa."""

    # first, setup the size of the action space
    X = 120
    Y = 100
    reduction = 4

    # check some types
    assert type(action) in [int, tuple], 'action must be of type int or tuple.'

    # different rules for if integer or tuple
    if type(action) is int:
        x = int(((action // (Y/reduction)) * reduction) + (reduction / 2))
        y = int(((action % (Y/reduction)) * reduction) + (reduction / 2))
        return x, y
    elif type(action) is tuple:
        # reducing action to within space
        x, y = action
        x = x // reduction
        y = y // reduction
        return x * (Y / reduction) + y
