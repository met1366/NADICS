class Color:
    BLACK = "\x1b[1;30m"
    RED = "\x1b[1;31m"
    GREEN = "\x1b[1;32m"
    BLUE = "\x1b[1;34m"
    YELLOW = "\x1b[1;33m"
    ENDC = "\x1b[0m"

def colored(string, color):
    return color + string + Color.ENDC
