import numpy as np
from numpy import amax, amin, mean, vectorize

from lib.image.ImageTransformation import ImageTransformation
from mlib.boot.mlog import err
from mlib.boot.stream import zeros

class MattSalienceFilter(ImageTransformation):

    def _transform(self, x: np.ndarray) -> np.ndarray:
        scales = [x]
        for i in range(1, 9):
            import cv2  # 3 SECOND IMPORT
            x = cv2.pyrDown(x)

            scales.append(x)

            self.intermediate_hook(f'scale_{i}', x)

        for center in [2, 3, 4]:
            for surround in [center + x for x in [3, 4]]:
                scale_diff = surround - center
                center_im = scales[center]
                surround_im = scales[center]
                feat_intense = zeros(center_im.shape[0], center_im.shape[1])
                feat_rg = zeros(center_im.shape[0], center_im.shape[1])
                feat_by = zeros(center_im.shape[0], center_im.shape[1])
                for px_row in range(0, center_im.shape[0]):
                    px_row_sur = px_row
                    for i in range(scale_diff):
                        px_row_sur = px_row_sur / 2
                    px_row_sur = round(px_row_sur)
                    for px_col in range(0, center_im.shape[1]):
                        px_col_sur = px_col
                        for i in range(scale_diff):
                            px_col_sur = px_col_sur / 2
                        px_col_sur = round(px_col_sur)

                        center_intense = sum(center_im[px_row, px_col])
                        surround_intense = sum(surround_im[px_row_sur, px_col_sur])
                        feat_intense[px_row, px_col] = abs(center_intense - surround_intense)
                        err('stick with intensity for now, do full learning experiments with it before moving on!')
                        err('DUH! if its black its black... its zero.')
                        center_rg = (center_im[px_row, px_col][0] / center_intense) - (
                                center_im[px_row, px_col][1] / center_intense)

                        surround_gr = (surround_im[px_row_sur, px_col_sur][1] / surround_intense) - (
                                surround_im[px_row_sur, px_col_sur][0] / surround_intense)
                        feat_rg[px_row, px_col] = abs(center_rg - surround_gr)

                        cen_yellow = mean([center_im[px_row, px_col, 0], center_im[px_row, px_col, 1]])
                        sur_yellow = mean(
                            [surround_im[px_row_sur, px_col_sur, 0], surround_im[px_row_sur, px_col_sur, 1]])

                        center_by = (center_im[px_row, px_col][2] / center_intense) - (cen_yellow / center_intense)
                        surround_yb = (sur_yellow / surround_intense) - (
                                surround_im[px_row_sur, px_col_sur][2] / surround_intense)
                        feat_by[px_row, px_col] = abs(center_by - surround_yb)

                self.intermediate_hook(f'feat_intense_{center}_{surround}',
                                       feat_intense - 1)  # minus 1 since plus one above?
                if center == 2 and surround == 5:
                    output = feat_intense - 1  # minus 1 since plus one above?

                mn = amin(feat_rg)
                mx = amax(feat_rg)

                def vis(px):
                    px = px - mn
                    px = px / (mx - mn)
                    px = px * 255
                    return px - 1  # minus 1 since plus one above?


                self.intermediate_hook(f'feat_rg_{center}_{surround}', vectorize(vis)(feat_rg))

                mn = amin(feat_by)
                mx = amax(feat_by)

                def vis(px):
                    px = px - mn
                    px = px / (mx - mn)
                    px = px * 255
                    return px - 1  # minus 1 since plus one above?

                self.intermediate_hook(f'feat_by_{center}_{surround}', vectorize(vis)(feat_by))

        #
        err('step 1. figure out what resolution it is supposed to return')
        return output  # not done?
