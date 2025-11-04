import google.generativeai as genai

genai.configure(api_key="AIzaSyDUf2o2kOpQXgef621Y8AqphlpLsAn5hHk")

# ✅ Show available models
for model in genai.list_models():
    print(model.name)
