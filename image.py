from PIL import Image
import pandas as pd
import numpy as np
for i in range(350):
    file = ("img_%d.jpg" %(i+1))
    img= Image.open(file)

    pixels = np.asarray(img)
    pixels = pixels/255
    csvfile=("testdata%d.csv" %(i+1))
    np.savetxt(csvfile ,pixels.reshape(1,pixels.size),delimiter=",")
list = []
for j in range(350):
    
  name =  ('testdata%d.csv' %(j+1))
  list.append(name)
combined_csv = pd.concat( [ pd.read_csv(f, header = None) for f in list] )    
combined_csv.to_csv( "test_csv.csv", index=False ,header=None)

