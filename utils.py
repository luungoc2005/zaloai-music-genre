def removeExt(filename):
    if '.' not in filename:
        return filename
    else:
        return filename[0:filename.index('.')]