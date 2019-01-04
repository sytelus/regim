import os,sys,inspect,logging

#this class simply tries to see if airsim 
class SetupPath:
    @staticmethod
    def getDirLevels(path):
        path_norm = os.path.normpath(path)
        return len(path_norm.split(os.sep))

    @staticmethod
    def getCurrentPath():
        cur_filepath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        return os.path.dirname(cur_filepath)

    @staticmethod
    def getGrandParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 2:
            return os.path.dirname(os.path.dirname(cur_path))
        return ''

    @staticmethod
    def getParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 1:
            return os.path.dirname(cur_path)
        return ''

    @staticmethod
    def addParentPath():
        parent = SetupPath.getParentDir()
        if parent !=  '':
            if sys.path.index(parent) != 0:
                sys.path.insert(0, parent)
        else:
            logging.warning("Cannot get parent folder and won't be added in search path for import")

SetupPath.addParentPath()

