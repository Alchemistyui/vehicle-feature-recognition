# USAGE
# python search_bing_api.py --query "charmander" --output dataset/charmander
# python search_bing_api.py --query "pikachu" --output dataset/pikachu
# python search_bing_api.py --query "squirtle" --output dataset/squirtle
# python search_bing_api.py --query "bulbasaur" --output dataset/bulbasaur
# python search_bing_api.py --query "mewtwo" --output dataset/mewtwo

# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
    help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory of images")
args = vars(ap.parse_args())

# set your Microsoft Cognitive Services API key along with (1) the
# maximum number of results for a given search and (2) the group size
# for results (maximum of 50 per request)
API_KEY = "38ced9e874324b45884a1b68d41e837a"
MAX_RESULTS = 250
GROUP_SIZE = 50

# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# when attemping to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])

# store the search term in a convenience variable then set the
# headers and search parameters
# 初始化搜索参数
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# make the search搜索
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# grab the results from the search, including the total number of
# estimated results returned by the Bing API
# 执行搜索并以JSON格式获取结果，计算并向终端打印估计的结果数量，初始化总数。
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
    term))

# initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

    # loop over the results
    # 循环遍历当前批量的图像，并尝试将每个单独的图像下载到我们的输出文件夹。
    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
            # 尝试通过URL获取图像，并为其构建路径+文件名，然后我们尝试打开并将文件写入磁盘
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)

            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)])

            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

        # try to load the image from disk
        # 看看OpenCV是否可以实际加载图像，这意味着（1）图像文件已成功下载，（2）图像路径有效：
        image = cv2.imread(p)

        # if the image is `None` then we could not properly load the
        # image from disk (so it should be ignored)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue

        # update the counter
        total += 1


# mkdir dataset/yui
# python3 search_bing_api.py --query "yui yoshioka" --output dataset/yui