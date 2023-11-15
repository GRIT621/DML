import pandas as pd
import requests
import pandas as pd
import time
import random
import signal
CUDA_VISIBLE_DEVICES="1"

# from timeout_decorator import timeout
# Define your API key
api_key = ""

# Define your proxy API
proxy_api_url = "https://api.bbai1.com/v1/chat/completions"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}
# file = open("test1.txt", "a+")

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")
# amazon_prompt = "Classify the follow topic into Society & Culture(1), Science & Mathematics(2), Health(3), Education & Reference(4)," \
#          " Computers & Internet(5), Sports(6), Business & Finance(7), Entertainment & Music(8), Family & Relationships(9)," \
#          "and Politics & Government(10). Note only return the number and no other words:"

yelp_prompt = "Classify the follow review into star 1-5 different sentiment levels from negative to positive, only return the number:"
# # yelp = pd.read_csv('/home/lsj0920/MPL/dataset/Yelp/yelp_review_full_csv/test.csv', names=['label','text'])
yelp = pd.read_csv('./dataset/Yelp/yelp_review_full_csv/test.csv', names=['label','text'])
test_yelp = yelp.sample(15000)
# # print(test_yelp['text'][292][:10])
#
#

for i in range(len(test_yelp)):
    new_prompt = yelp_prompt +test_yelp['text'][i]
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
    with open("yelp.txt", "a+") as file:
        file.write(str(a) + ",," + "text:" + test_yelp['text'][i][:10] + '\n')
        file.close()




# response = requests.post(proxy_api_url, headers=headers, json=data)
# resp = response.json()
# print(resp["choices"][0]["message"]['content'])