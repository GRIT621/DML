import pandas as pd
import requests
import pandas as pd
import time
import random
import signal
CUDA_VISIBLE_DEVICES="1"

# from timeout_decorator import timeout
# Define your API key
api_key = "sk-GZdJZKgMlIoyW7qx387435C67b554e87Ab86988d15Ae84Fa"

# Define your proxy API
proxy_api_url = "https://api.bbai1.com/v1/chat/completions"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}
# file = open("test1.txt", "a+")

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

prompt = "Classify the follow review into star 1-5 different sentiment levels from negative to positive, only return the number:"
# # yelp = pd.read_csv('/home/lsj0920/MPL/dataset/Yelp/yelp_review_full_csv/test.csv', names=['label','text'])
yelp = pd.read_csv('/home/lsj0920/MPL/dataset/Yelp/yelp_review_full_csv/test.csv', names=['label','text'])
test_yelp = yelp[3000:5000]
# # print(test_yelp['text'][292][:10])
#
#

for i in range(3000,len(test_yelp)+3000):
    new_prompt = prompt +test_yelp['text'][i]
    data = {
    'model': 'gpt-3.5-turbo',  # Specify the model here
    'messages': [{
        'role': 'system',
        'content': new_prompt},]
    }
    count = 2
    while True:
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(5)
            response = requests.post(proxy_api_url, headers=headers, json=data)
            signal.alarm(0)
            resp = response.json()
            a = int(resp["choices"][0]["message"]['content'][0])
            # print("!!",a)
        except Exception as e:
            print("@@", e)
            # print("&&", resp)
            a = -1
            count -= 1
            # time.sleep(60)
        print(a)
        if a != -1 or count <= 0 :
            break
    with open("test3000-5000.txt", "a+") as file:
        file.write(str(a) + ",," + "text:" + test_yelp['text'][i][:10] + '\n')
        file.close()




# response = requests.post(proxy_api_url, headers=headers, json=data)
# resp = response.json()
# print(resp["choices"][0]["message"]['content'])