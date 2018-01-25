import os
import localConfig as cfg
from keras.models import model_from_json


runNum = 20
model_name = "L1_N5_550_520_run20"
filepath = cfg.lgbk + "Searches/run"+str(runNum)
os.chdir(filepath)

print "Loading Model ..."
with open(model_name+'.json', 'r') as json_file:
  loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")
