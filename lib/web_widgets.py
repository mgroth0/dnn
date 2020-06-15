import pdb

from lib.web import *
from lib.wolf.wolfpy import WOLFRAM

def FigureTable(*figs_captions, apiURL=None, exp_id=None, editable=False):
    children = []
    for fig, caption in figs_captions:
        id = exp_id + '.' + '.'.join(
            File(fig).names(keepExtension=False)[-1:])
        log(f'creating figure: {id}')
        children.append(
            TableRow(
                DataCell(
                    HTMLImage(WOLFRAM.push_file(fig)[0])
                )
                , DataCell(
                    P(caption, id=id) if not editable else TextArea(caption, id=id, style='''
                    height: 100%;
                    
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;'''),
                    style='position:relative;width:100%;')
            )
        )
    return Table(*children)
