import optparse
from app.helpers import build_recognizer,calculate
import os
from app.gui import GUI

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--Train", action="store_true",
                         default=False, help="Train the model")
    opt_parser.add_option("--GUI", action="store_true",
                         default=False, help="Launch the GUI")
    options, args = opt_parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    if options.Train:
        build_recognizer()
    else:
        GUI()

