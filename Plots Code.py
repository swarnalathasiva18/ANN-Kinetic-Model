plt.figure(figsize=(500,150))
plt.title('Correlation of Attributes with Class variable')
correlation=df.corr()
sns.set(font_scale=1.4)
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()

inpdata1=inpdata.copy()


for num in range(len(inpdata1)):
    inpdata1[num]=inpdata1[num].drop_duplicates(subset=inpdata1[num].columns.difference(['#time','npart']), keep='last')
    #inpdata1[num]=inpdata1[num][inpdata1[num]['N2']<0.9999]
    inpdata1[num]=inpdata1[num].sort_values(["npart","#time"])

for num in range(len(inpdata1)):
    inpdata1[num].insert(41, "T", Ts[num])


datfull=pd.concat(inpdata1)

datinp=datfull.iloc[:,:41]
datinp1=datinp[(datinp<0).any(1)]
datinp1

datinp=datfull.iloc[:,2:41]
datinp1=datinp[(datinp>1).any(1)]
datinp1

import numpy as np
import matplotlib.pyplot as plt

y=datfull.iloc[:,42:].values
dat=[i for i in range(39)]
for i in dat:
    print(datfull.columns[i+42],y[:,i].mean(),y[:,i].std())
