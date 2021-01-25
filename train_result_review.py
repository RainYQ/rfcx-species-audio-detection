import matplotlib.pyplot as plt
import pickle
import numpy as np

train_history = "./data/result/Train_History_Dict.txt"
file = open(train_history, "rb")
history = pickle.load(file)
print(history.keys())
# summarize history for accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
# using when some loss inf
data = np.clip(history['val_loss'], 0, 1)
plt.plot(data)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()
file.close()
