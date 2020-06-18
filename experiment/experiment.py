import datetime
import json
import logging
import os
from logging import handlers

logger = logging.getLogger('experiment')


class experiment:
    '''
    Class to create directory and other meta information to store experiment results.
    A directory is created in output_dir/DDMMYYYY/name_0
    In-case there already exists a folder called name, name_1 would be created.
    '''

    def __init__(self, name, args, output_dir="../", commit_changes=False, rank=None, seed=None):
        import sys
        self.command_args = "python " + " ".join(sys.argv)
        if not args is None:
            if rank is not None:
                self.name = name + str(rank) + "/" + str(seed)
            else:
                self.name = name
            self.params = args
            print(self.params)
            self.results = {}
            self.dir = output_dir

            root_folder = datetime.datetime.now().strftime("%d%B%Y")

            if not os.path.exists(output_dir + root_folder):
                try:
                    os.makedirs(output_dir + root_folder)
                except:
                    assert (os.path.exists(output_dir + root_folder))

            self.root_folder = output_dir + root_folder
            full_path = self.root_folder + "/" + self.name

            ver = 0

            while True:
                ver += 1
                if not os.path.exists(full_path + "_" + str(ver)):
                    try:
                        os.makedirs(full_path + "_" + str(ver))
                        break
                    except:
                        pass
            self.path = full_path + "_" + str(ver) + "/"

            fh = logging.FileHandler(self.path + "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(fh)

            ch = logging.handlers.logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            self.store_json()

    def is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False

    def add_result(self, key, value):
        assert (self.is_jsonable(key))
        assert (self.is_jsonable(value))
        self.results[key] = value

    def store_json(self):
        with open(self.path + "metadata.json", 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4, separators=(',', ': '), sort_keys=True)
            outfile.write("")

    def get_json(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)
