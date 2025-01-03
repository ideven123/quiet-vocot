from model.load_model import load_model, infer
from PIL import Image

# loading the model
model_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangwanlin-240108540162/VoCoT/offline/Volcano'
model, preprocessor = load_model(model_path, precision='fp16')

# perform reasoning, activate VoCoT by passing cot=True
input_image = Image.open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangwanlin-240108540162/VoCoT/figs/sample_input.jpg')
response_1 = infer(model, preprocessor, input_image, 'Is there a event "the cat is below the bed" in this image?', cot=True)
response_2 = infer(model, preprocessor, input_image, 'Why is the cat on the bed?', cot=True)
response_3 = infer(model, preprocessor, input_image, 'Describe the image.', cot=True)
print('response 1: ', response_1[0])
print('response 2: ', response_2[0])
print('response 3: ', response_3[0])