from mlib.file import Folder, File
if __name__ == '__main__':
    for shape in Folder('_images_human_copy_3/Contour/30').files:
        for f in shape.files:
            f = File(f)
            n = f.name_pre_ext.split(shape.name)[1]
            f.rename('contour' + str(n) +  '.png')

