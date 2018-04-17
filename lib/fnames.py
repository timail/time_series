# UTF-8, *nix

# name of dir for plots
IMG_DIR = './plots/'

def file_numeration():
    n = 0
    while True:
        yield '{:02d}'.format(n + 1) 
        n += 1

n = file_numeration()

# plot filename format: 01_name.png
def plot(name):
    return IMG_DIR + next(n) + '_' + name
