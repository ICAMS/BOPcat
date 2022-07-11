import os
import time
from .variables import bopcat_version
import ase
import numpy as np


def get_bopfox_version():
    temp = os.popen("which bopfox").read().strip()
    temp = temp.split('/')
    path2bopfox = ''
    for i in range(1, len(temp) - 1):
        path2bopfox += ('/' + temp[i])
    if not os.path.isdir(path2bopfox):
        raise ValueError("Provided path; %s does not exist." % path2bopfox)
    cwd = os.getcwd()
    os.chdir(path2bopfox)
    if os.path.isfile('bopmain.fp'):
        l = open('bopmain.fp').readlines()
    else:
        print('Cannot determine BOPfox version.')
        os.chdir(cwd)
        return 'Unknown'
    for i in range(len(l)):
        if 'BOPfox' in l[i] and 'rev' in l[i]:
            ver = l[i].strip()
            ver = ver[ver.index('BOP'):-1]
            break
    os.chdir(cwd)
    return ver


def print_format(message, level=0, stream=False):
    space = ' ' * ((level + 1) * 4)
    out = space + message
    print(out)
    if stream:
        return out


def bopcat_logo():
    catv = bopcat_version()
    tim = time.asctime()
    try:
        asev = ase.__version__ + ' (%s)' % os.path.dirname(ase.__file__)
    except:
        asev = 'unknown'
    bopv = get_bopfox_version()
    user = '%s@%s' % (os.getenv('USER', '???'), os.uname()[1])
    out = """     
    _______________________             ____       
    ___  __ )_  __ \__  __ \___________ __  /_     
    __  __  |  / / /_  /_/ /  ___/  __ `/  __/     
    _  /_/ // /_/ /_  ____// /__ / /_/ // /_       
    /_____/ \____/ /_/     \___/ \__,_/ \__/       
                                             
    Bond-Order Potential construction and testing  
    {0}                                            
    Date: {1}                                      
    User: {2}               
    BOPfox version: {3}     
    ASE version: {4}        
    """.format(catv, tim, user, bopv, asev)
    return out


class cattxt:
    def __init__(self, **kwargs):
        self.txt = ''
        self.filename = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'filename':
                self.filename = kwargs[key]

    #    def formatting(self,txt):
    #        m0, m1 = 0, 0
    #        for i in range(len(txt)):
    #            s = str(txt[i])
    #            if isinstance(txt[i],float):
    #                s = s.split('.')
    #                if len(s[0]) > m0:
    #                    m0 = len(s[0])
    #                if len(s[1]) > m1:
    #                    m1 = len(s[1])
    #            else:
    #                if len(s) > m0:
    #                    m0 = len(s)
    #        return m0+m1, m1

    def add(self, txt, length='default'):
        if type(txt) in [str, int, float]:
            txt = [txt]
        if length == 'default':
            length = [(30, 10)] * len(txt)
        else:
            if isinstance(length, int):
                length = [(length, 0)]
            else:
                length = [(int(i), i - int(i)) for i in length]
        assert (len(length) == len(txt))
        if isinstance(txt, list) or isinstance(txt, np.ndarray):
            if len(np.shape(txt)) > 1:
                print_format('cannot dump array', level=3)
                return
            temp = ''
            for i in range(len(txt)):
                m0 = length[i][0]
                m1 = length[i][1]
                if isinstance(txt[i], float):
                    temp += '%*.*f    ' % (m0, m1, txt[i])
                elif isinstance(txt[i], int):
                    temp += '%*d    ' % (m0, txt[i])
                elif isinstance(txt[i], str):
                    temp += '%*s    ' % (m0, txt[i])
            txt = temp + '\n'
        self.txt += txt

    def write(self):
        f = open(self.filename, 'w')
        f.write(self.txt)
        f.close()


class counter:
    def __init__(self):
        self.i = 0

    def __call__(self, incr=1):
        self.i += incr


if __name__ == "__main__":
    out = cattxt(filename='del.out')
    out.add('This is an example')
    out.add('of the BOPcat file handling capability')
    out.add(['a', 'b', 'c', 'd'], length=[5, 5, 5, 5])
    out.add([1.4, 1.50000, 1.4678, 2])
    out.add(0, length=5)
    out.add('end', length=10)
    out.write()
