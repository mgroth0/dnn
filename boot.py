from mlib.boot import exec_proj
import os
import mlib.file
mlib.file.dummy = 'please stop circular imports'
exec_proj.exec_proj(
    os.path.basename(os.getcwd())
)
