import openai

openai.api_key = open("openai_key.txt").read()

print(openai.api_key)
