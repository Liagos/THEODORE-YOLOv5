import argparse
from configparser import ConfigParser


class Arguments:
    def __init__(self):
        self.file = "config.ini"
        self.p = argparse.ArgumentParser()

    def getArgs(self):
        self.p.add_argument("-c", "--configFile", dest='config_file', default='config.ini',
                            type=str,
                            help="path to config file for the annotation parser")  # ,required=True)
        args = self.p.parse_args()

        config_file = args.config_file
        config = ConfigParser()
        config.read(config_file)

        self.p.add_argument("-output_path", "--outputPath", dest="outputPath",
                            default=config["outputPath"]["output_path"],
                            type=str,
                            help="dataset output folder")
        args = self.p.parse_args()

        return args
