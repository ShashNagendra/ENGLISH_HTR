#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''from google.colab import drive
drive.mount('/content/drive/')
'''


# In[2]:


#!python main.py --source=iam --transform


# In[3]:


#%cd /home/shashank/Desktop/handwritten-text-recognition-master/src


# In[4]:


get_ipython().system('pip install kaldiio')
get_ipython().system('pip install stn')
get_ipython().system('pip install rapidfuzz')
get_ipython().system('pip install seaborn')


# In[5]:


get_ipython().system('pip install tensorflow_addons')
import string
sub = dict.fromkeys(string.printable[:95], 0)
ins = dict.fromkeys(string.printable[:95], 0)
delete = dict.fromkeys(string.printable[:95], 0)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np
array=np.zeros((len(string.printable[:95]), len(string.printable[:95])))
#array =array.astype(np.float)


###########################
import datetime
 
# using now() to get current time
now = str(datetime.datetime.now())





# In[6]:

def alg4(word1, word2):
    M = [[float('inf')] * (len(word2) + 1) for _ in range(len(word1) + 1)]

    # Filling last row
    for i in range(len(word2) + 1):
        M[len(word1)][i] = len(word2) - i

    # Filling last column
    for j in range(len(word1) + 1):
        M[j][len(word2)] = len(word1) - j

    # Filling bottom to up manner
    for i in range(len(word1) - 1, -1, -1):
        for j in range(len(word2) - 1, -1, -1):
            if word1[i] == word2[j]:
                M[i][j] = M[i + 1][j + 1]
            else:
                M[i][j] = 1 + min(M[i + 1][j], M[i][j + 1], M[i + 1][j + 1])

    x, y = 0, 0
    count = 0
    while x < len(M) - 1 and y < len(M[0]) - 1:
        current = M[x][y]
        dia = M[x + 1][y + 1]
        right = M[x][y + 1]
        bottom = M[x + 1][y]
        if dia <= right and dia <= bottom and dia <= current:
            if dia == current - 1:
                index_word1 = string.printable[:95].find(word1[x])
                index_word2 = string.printable[:95].find(word2[y])
                if index_word1 != -1 and index_word2 != -1:
                    print("Substitution -->", word1[x], "replaced by", word2[y])
                    # Update data structures here if needed
                    count += 1
                else:
                    print("Invalid characters encountered:", word1[x], word2[y])
                x += 1
                y += 1
            else:
                print("No operation -->", word1[x])
                x += 1
                y += 1
        elif right <= bottom and right <= current:
            print("Insertion", word2[y])
            # Update data structures here if needed
            count += 1
            y += 1
        else:
            print("Deletion", word1[x])
            # Update data structures here if needed
            x += 1
            count += 1
    print("Total operations:", count)

# In[7]:


import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name != "/device:GPU:0":
    raise SystemError("GPU device not found")

print("Found GPU at: {}".format(device_name))




import os
import datetime
import string

# define parameters
source = "iam"
arch = "flor"
epochs = 1000
batch_size = 16


# define paths
source_path = os.path.join("..", "data", f"{source}.hdf5")
output_path = os.path.join("..", "output", source, arch,now)

target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
os.makedirs(output_path, exist_ok=True)

# define input size, number max of chars per line and list of valid chars
input_size = (1024, 128, 1)
max_text_length = 128
charset_base = string.printable[:95]

print("source:", source_path)
print("output", output_path)
print("target", target_path)
print("charset:", charset_base)



from data.generator import DataGenerator

dtgen = DataGenerator(source=source_path,
                      batch_size=batch_size,
                      charset=charset_base,
                      max_text_length=max_text_length)

print(f"Train images: {dtgen.size['train']}")
print(f"Validation images: {dtgen.size['valid']}")
print(f"Test images: {dtgen.size['test']}")






from network.best4senetgateddropouttwo import HTRModel

# create and compile HTRModel
model = HTRModel(architecture=arch,
                 input_size=input_size,
                 vocab_size=dtgen.tokenizer.vocab_size,
                 beam_width=10,
                 stop_tolerance=20,
                 reduce_tolerance=15)

model.compile(learning_rate=0.001)
model.summary(output_path, "summary.txt")

# get default callbacks and load checkpoint weights file (HDF5) if exists
model.load_checkpoint(target=target_path)

callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)





# to calculate total and average time per epoch
start_time = datetime.datetime.now()

h = model.fit(x=dtgen.next_train_batch(),
              epochs=epochs,
              steps_per_epoch=dtgen.steps['train'],
              validation_data=dtgen.next_valid_batch(),
              validation_steps=dtgen.steps['valid'],
              callbacks=callbacks,
              shuffle=True,
              verbose=1)

total_time = datetime.datetime.now() - start_time

loss = h.history['loss']
val_loss = h.history['val_loss']

min_val_loss = min(val_loss)
min_val_loss_i = val_loss.index(min_val_loss)

time_epoch = (total_time / len(loss))
total_item = (dtgen.size['train'] + dtgen.size['valid'])

t_corpus = "\n".join([
    f"Total train images:      {dtgen.size['train']}",
    f"Total validation images: {dtgen.size['valid']}",
    f"Batch:                   {dtgen.batch_size}\n",
    f"Total time:              {total_time}",
    f"Time per epoch:          {time_epoch}",
    f"Time per item:           {time_epoch / total_item}\n",
    f"Total epochs:            {len(loss)}",
    f"Best epoch               {min_val_loss_i + 1}\n",
    f"Training loss:           {loss[min_val_loss_i]:.8f}",
    f"Validation loss:         {min_val_loss:.8f}"
])

with open(os.path.join(output_path, "train.txt"), "w") as lg:
    lg.write(t_corpus)
    print(t_corpus)
    
    


# In[8]:


from data import preproc as pp
import cv2
#from google.colab.patches import cv2_imshow
start_time = datetime.datetime.now()

# predict() function will return the predicts with the probabilities
predicts, s = model.predict(x=dtgen.next_test_batch(),
                            steps=dtgen.steps['test'],
                            ctc_decode=True,
                            verbose=1)
                            
print(s)
# decode to string
predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

total_time = datetime.datetime.now() - start_time

# mount predict corpus file
with open(os.path.join(output_path, "predict.txt"), "w") as lg:
  for pd, gt in zip(predicts, ground_truth):
        lg.write(f"TE_L {gt}\nTE_P {pd}\n")
   
for i, item in enumerate(dtgen.dataset['test']['dt'][:10]):
  print("=" * 1024, "\n")
  cv2.imshow('',pp.adjust_to_see(item))
  print(ground_truth[i])
  print(predicts[i], "\n")
    
    
   
    
    
from data import evaluation

evaluate = evaluation.ocr_metrics(predicts, ground_truth)

e_corpus = "\n".join([
    f"Total test images:    {dtgen.size['test']}",
    f"Total time:           {total_time}",
    f"Time per item:        {total_time / dtgen.size['test']}\n",
    f"Metrics:",
    f"Character Error Rate: {evaluate[0]:.8f}",
    f"Word Error Rate:      {evaluate[1]:.8f}",
    f"Sequence Error Rate:  {evaluate[2]:.8f}"
])

with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
  lg.write(e_corpus)
    
    

print("\neval=",evaluate)    
    
    
    
    
    
    


# In[9]:


#evaluation metrics added

 
   
# Copyright (c) OpenMMLab. All rights reserved.
import re
from difflib import SequenceMatcher

from rapidfuzz import string_metric


def cal_true_positive_char(pred, gt):
 all_opt = SequenceMatcher(None, pred, gt)
 true_positive_char_num = 0
 for opt, _, _, s2, e2 in all_opt.get_opcodes():
       if opt == 'equal':
         true_positive_char_num += (e2 - s2)
       else:
         pass
 return true_positive_char_num


def count_matches(pred_texts, gt_texts):
   match_res = {
       'gt_char_num': 0,
       'pred_char_num': 0,
       'true_positive_char_num': 0,
       'gt_word_num': 0,
       'match_word_num': 0,
       'match_word_ignore_case': 0,
       'match_word_ignore_case_symbol': 0
   }
   comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
   norm_ed_sum = 0.0
   for pred_text, gt_text in zip(pred_texts, gt_texts):
       if gt_text == pred_text:
           match_res['match_word_num'] += 1
       gt_text_lower = gt_text.lower()
       pred_text_lower = pred_text.lower()
       if gt_text_lower == pred_text_lower:
           match_res['match_word_ignore_case'] += 1
       gt_text_lower_ignore = comp.sub('', gt_text_lower)
       pred_text_lower_ignore = comp.sub('', pred_text_lower)
       if gt_text_lower_ignore == pred_text_lower_ignore:
           match_res['match_word_ignore_case_symbol'] += 1
       match_res['gt_word_num'] += 1

       # normalized edit distance
       edit_dist = string_metric.levenshtein(pred_text_lower_ignore,
                                             gt_text_lower_ignore)
       norm_ed = float(edit_dist) / max(1, len(gt_text_lower_ignore),
                                        len(pred_text_lower_ignore))
       norm_ed_sum += norm_ed

       # number to calculate char level recall & precision
       match_res['gt_char_num'] += len(gt_text_lower_ignore)
       match_res['pred_char_num'] += len(pred_text_lower_ignore)
       true_positive_char_num = cal_true_positive_char(
           pred_text_lower_ignore, gt_text_lower_ignore)
       match_res['true_positive_char_num'] += true_positive_char_num

   normalized_edit_distance = norm_ed_sum / max(1, len(gt_texts))
   match_res['ned'] = normalized_edit_distance

   return match_res


def eval_ocr_metric(pred_texts, gt_texts):
   """Evaluate the text recognition performance with metric: word accuracy and
   1-N.E.D. See https://rrc.cvc.uab.es/?ch=14&com=tasks for details.

   Args:
       pred_texts (list[str]): Text strings of prediction.
       gt_texts (list[str]): Text strings of ground truth.

   Returns:
       eval_res (dict[str: float]): Metric dict for text recognition, include:
           - word_acc: Accuracy in word level.
           - word_acc_ignore_case: Accuracy in word level, ignore letter case.
           - word_acc_ignore_case_symbol: Accuracy in word level, ignore
               letter case and symbol. (default metric for
               academic evaluation)
           - char_recall: Recall in character level, ignore
               letter case and symbol.
           - char_precision: Precision in character level, ignore
               letter case and symbol.
           - 1-N.E.D: 1 - normalized_edit_distance.
   """
   assert isinstance(pred_texts, list)
   assert isinstance(gt_texts, list)
   assert len(pred_texts) == len(gt_texts)

   match_res = count_matches(pred_texts, gt_texts)
   eps = 1e-8
   char_recall = 1.0 * match_res['true_positive_char_num'] / (
       eps + match_res['gt_char_num'])
   char_precision = 1.0 * match_res['true_positive_char_num'] / (
       eps + match_res['pred_char_num'])
   word_acc = 1.0 * match_res['match_word_num'] / (
       eps + match_res['gt_word_num'])
   word_acc_ignore_case = 1.0 * match_res['match_word_ignore_case'] / (
       eps + match_res['gt_word_num'])
   word_acc_ignore_case_symbol = 1.0 * match_res[
       'match_word_ignore_case_symbol'] / (
           eps + match_res['gt_word_num'])

   eval_res = {}
   eval_res['word_acc'] = word_acc
   eval_res['word_acc_ignore_case'] = word_acc_ignore_case
   eval_res['word_acc_ignore_case_symbol'] = word_acc_ignore_case_symbol
   eval_res['char_recall'] = char_recall
   eval_res['char_precision'] = char_precision
   eval_res['1-N.E.D'] = 1.0 - match_res['ned']

   for key, value in eval_res.items():
       eval_res[key] = float('{:.4f}'.format(value))
   print("predicted text:",pred_texts)
   return eval_res
   print(e_corpus)
   
   
   
evaluate1 = count_matches(predicts, ground_truth) 
print("\neval1=",evaluate1)


import re
 
def remove1(string):
   pattern = re.compile(r'\s+')
   return re.sub(pattern, '', string)
   



import string       
for pred_text, gt_text in zip(predicts, ground_truth):        
	seq2 = pred_text
	seq1 = gt_text
	print("pred text:",seq2)
	print("ground truth:",seq1)
	#seq1=seq1.translate({ord(c): None for c in string.whitespace})

	#seq2=seq2.translate({ord(c): None for c in string.whitespace})
	#seq1=remove1(seq1)
	#seq2=remove1(seq2)
	print("pred text2:",seq2)
	print("ground truth2:",seq1)
	alg4(seq1,seq2)

print(ins)
print(delete)
print(sub)


# In[10]:


keymax = max(zip(sub.values(), sub.keys()))[1]
keymax
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.bar(sub.keys(),sub.values())
myList = sub.items()
myList = sorted(myList) 
x, y = zip(*myList) 
#print(x,y)
plt.plot(x, y)
plt.show()


# In[11]:


myList = sub.items()
myList = sorted(myList) 
x, y = zip(*myList) 
#print(x,y)
plt.plot(x, y)
plt.show()


# In[12]:


'''
import string
len(string.printable[:95])
import pandas as pd

df_cm = pd.DataFrame(array, index = [i for i in string.printable[:95]],
                  columns = [i for i in string.printable[:95]])
plt.figure(figsize = (100,100))
sn.heatmap(df_cm, annot=True,fmt="s")
'''


# In[13]:


sub


# In[14]:


new=array[10:62,10:62]


# In[15]:


import pandas as pd
plt.figure(figsize = (100,100))
#new=new.pivot(string.printable[10:62])
df_cm = pd.DataFrame(new, index = [i for i in string.printable[10:62]],
                  columns = [i for i in string.printable[10:62]])
ax = sn.heatmap(new, annot=True, fmt="f",cmap="YlGnBu")


# In[16]:


import numpy
numpy.savetxt("iam puigcerver+SENet ratio2"+now+".txt", new)


# In[17]:


ins


# In[18]:


delete


# In[ ]:




