import os
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def combine_imgs(img1, img2):
  new_image = Image.new('RGB',(2*28, 28), (250,250,250))
  new_image.paste(img1,(0,0))
  new_image.paste(img2,(28,0))
  new_image = new_image.resize((28,28))
  return new_image

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)

numbers = np.arange(0,100,1)
train_numbers = random.choices(numbers, k=70)
test_numbers = random.choices(numbers, k=30)

for i in train_numbers:
  try:
    os.mkdir("./images/train/"+str(i)+"/")
  except:
    continue

num_to_idx = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
for i in range(0,len(mnist_dataset)):
    img, lab = mnist_dataset[i]
    num_to_idx[lab].append(i)

last_num_idx = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for i in train_numbers:
  for j in range(0,250):
    if i<10:
      num_idx_arr = num_to_idx[i]
      num_idx = num_idx_arr[last_num_idx[i]]
      
      img, _ = mnist_dataset[num_idx]

      img.save("./images/train/"+str(i)+"/"+str(last_num_idx[i])+".png")

      last_num_idx[i] += 1
    
    else:
      num_str = str(i)
      num_1 = int(num_str[0])
      num_2 = int(num_str[1])

      num_idx_arr1 = num_to_idx[num_1]
      num_idx1 = num_idx_arr1[last_num_idx[num_1]]
      last_num_idx[num_1] += 1

      img1, _ = mnist_dataset[num_idx1]

      num_idx_arr2 = num_to_idx[num_2]
      num_idx2 = num_idx_arr2[last_num_idx[num_2]]
      last_num_idx[num_2] += 1

      img2, _ = mnist_dataset[num_idx2]

      combined_img = combine_imgs(img1, img2)

      combined_img.save("./images/train/"+str(i)+"/"+str(j)+".png")

mnist_dataset = datasets.MNIST(root='./data', train=False, download=True)

for i in test_numbers:
  try:
    os.mkdir("./images/test/"+str(i)+"/")
  except:
    continue

num_to_idx = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
for i in range(0,len(mnist_dataset)):
    img, lab = mnist_dataset[i]
    num_to_idx[lab].append(i)

last_num_idx = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for i in test_numbers:
  for j in range(0,40):
    if i<10:
      num_idx_arr = num_to_idx[i]
      num_idx = num_idx_arr[last_num_idx[i]]
      
      img, _ = mnist_dataset[num_idx]

      img.save("./images/test/"+str(i)+"/"+str(last_num_idx[i])+".png")

      last_num_idx[i] += 1
    
    else:
      num_str = str(i)
      num_1 = int(num_str[0])
      num_2 = int(num_str[1])

      num_idx_arr1 = num_to_idx[num_1]
      num_idx1 = num_idx_arr1[last_num_idx[num_1]]
      last_num_idx[num_1] += 1

      img1, _ = mnist_dataset[num_idx1]

      num_idx_arr2 = num_to_idx[num_2]
      num_idx2 = num_idx_arr2[last_num_idx[num_2]]
      last_num_idx[num_2] += 1

      img2, _ = mnist_dataset[num_idx2]

      combined_img = combine_imgs(img1, img2)

      combined_img.save("./images/test/"+str(i)+"/"+str(j)+".png")