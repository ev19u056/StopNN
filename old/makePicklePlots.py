
import pickle
import matplotlib.pyplot as plt

acc = pickle.load(open("accuracy.pickle", "rb"))
loss = pickle.load(open("loss.pickle", "rb"))

plt.figure(figsize=(7,6))
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('pickleplots.png')
