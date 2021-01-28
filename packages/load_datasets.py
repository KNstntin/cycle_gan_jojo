import urllib.request

def real2jojo():
    url = "https://www.dropbox.com/s/ukeygzg9ywnnapg/real2jojo.zip?dl=1"
    zip_file = urllib.request.urlopen(url)
    data = zip_file.read()
    zip_file.close()
    with open('real2jojo.zip', "wb") as f:
        f.write(data)

def horse2zebra():
    url = "https://www.dropbox.com/s/xm7c5jw2ut389o6/horse2zebra.zip?dl=1"
    zip_file = urllib.request.urlopen(url)
    data = zip_file.read()
    zip_file.close()
    with open('horse2zebra.zip', "wb") as f:
        f.write(data)
