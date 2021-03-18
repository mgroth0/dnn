from mlib.boot import exec_proj
exec_proj.dummy = None
import os
os.dummy = None
import mlib.file
mlib.file.dummy = 'please stop circular imports'
# exec_proj.exec_proj(
#     os.path.basename(os.getcwd())
# )
# DEBUG
import dnn
dnn.dummy = None
