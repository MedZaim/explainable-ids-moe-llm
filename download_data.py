# download_data.py
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", 
    "KDDTrain+.txt"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt", 
    "KDDTest+.txt"
)