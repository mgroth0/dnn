from lib.salience.filter import salience_filter, saliency3
from mlib.JsonSerializable import obj

small = ''
# small = '_small'

def main():
    # salience_filter.main()
    for small in ['','_small']:
        saliency3.main(obj({
            # 'inputFile': '_figs/salience_filter/input.png',
            # 'intensityOutput': '_figs/salience_filter/tatome/intensity.png',
            'inputFile': f'_figs/salience_filter/input{small}.png',
            'intensityOutput': f'_figs/salience_filter/tatome/intensity{small}.png',
            'fileList': None,
            'gaborOutput': f'_figs/salience_filter/tatome/gabor{small}.png',
            'rgOutput': f'_figs/salience_filter/tatome/rg{small}.png',
            'byOutput': f'_figs/salience_filter/tatome/by{small}.png',
            'cOutput': f'_figs/salience_filter/tatome/c{small}.png',
            'saliencyOutput': f'_figs/salience_filter/tatome/saliency{small}.png',
            'markMaxima': None
        }))


