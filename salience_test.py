from lib.salience.filter import salience_filter, saliency3
from mlib.JsonSerializable import obj

def main():
    salience_filter.main()
    saliency3.main(obj({
        'inputFile': '_figs/salience_filter/input.png', 'intensityOutput': '_figs/salience_filter/tatome/intensity.png',
        'fileList': None,
        'gaborOutput': None,
        'rgOutput': None,
        'byOutput': None,
        'cOutput': None,
        'saliencyOutput': None,
        'markMaxima': None
    }))


