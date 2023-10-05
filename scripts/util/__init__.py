import os
from zipfile import ZipFile
from tempfile import NamedTemporaryFile


def zip_dir(folder):
    file = NamedTemporaryFile()
    with ZipFile(file, 'w') as outzip:
        for subdir, _, files in os.walk(folder):
            for f in files:
                # Read file
                srcpath = os.path.join(subdir, f)
                dstpath_in_zip = os.path.relpath(srcpath, start=folder)
                with open(srcpath, 'rb') as infile:
                    # Write to zip
                    outzip.writestr(dstpath_in_zip, infile.read())
    return file
