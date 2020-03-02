import os
for root, dirs, files in os.walk('.', topdown=True):
    print(files)
    print(root)
    print(dirs)
    b = os.path.join(root,files[0])
    print(b)
