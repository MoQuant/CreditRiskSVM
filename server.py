'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
final_dataset[cat] = {'ClassA':{'x':cx,'y':cy,'z':cz,'color':'red'},
'ClassB':{'x':px,'y':py,'z':pz,'color':'green'},
'Surface':{'x':lx.tolist(),'y':ly.tolist(),'z':lz.tolist()}}

# model = SVC()
# coef = model.coef_[0]
# b0 = clf.intercept_
# b1 = -coef[0]/coef[1]
    
'''

import asyncio
import websockets
import json

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def Plane(model, x, n=50):
    coef = model.coef_
    b = model.intercept_

    x0 = np.min(x[:, 0])
    x1 = np.max(x[:, 0])
    y0 = np.min(x[:, 1])
    y1 = np.max(x[:, 1])
    dx = (x1 - x0)/(n - 1)
    dy = (y1 - y0)/(n - 1)

    ux, uy, uz = [], [], []
    lx, ly, lz = [], [], []

    for i in range(n):
        ux, uy, uz = [], [], []
        qx = x0 + i*dx
        for j in range(n):
            qy = y0 + j*dy
            ux.append(qx)
            uy.append(qy)
            uz.append(float(coef[0][0]*qx + coef[0][1]*qy + b))
        lx.append(ux); ly.append(uy); lz.append(uz)
    return lx, ly, lz
            

def Sample(f):
    def Solve(*a, **b):
        df, y, j0, j1 = f(*a, **b)
        return df[:1000], y[:1000], j0, j1[:1000]
    return Solve

def ZNorm(df):
    x = df.values
    m, n = x.shape
    for i in range(n):
        x[:, i] -= np.mean(x[:, i])
        x[:, i] /= np.std(x[:, i])
    return x

@Sample
def Dataset(filename='Loans.csv'):
    df = pd.read_csv(filename)
    
    y = df['NotFullyPaid'].values.tolist()

    df['NotFullyPaid'] = list(map(lambda x: -1 if x == 1 else 1, y))

    plus = df[(df['NotFullyPaid'] == -1)]
    minus = df[(df['NotFullyPaid'] == 1)]

    minus = minus.iloc[0:len(plus)]

    ds = pd.concat([plus, minus])
    del ds['CreditPolicy']
    purpose_of_debt = ds['Purpose'].values.tolist()
    purpose = list(set(purpose_of_debt))
    ds = ds.sample(frac=1).reset_index(drop=True)
    y = ds['NotFullyPaid'].values.tolist()
    
    del ds['NotFullyPaid']
    del ds['Purpose']

    return ds, y, purpose, purpose_of_debt

async def Serving(ws, path):
    print("Connected to plotter")
    ds, y, categories, full_categories = Dataset()

    dZ = ZNorm(ds)

    data = {cat:{'x':[],'y':[]} for cat in categories}

    for cat, ns, yi in zip(full_categories, dZ, y):
        data[cat]['x'].append(ns)
        data[cat]['y'].append(yi)

    final_dataset = {}
    for cat in categories:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data[cat]['x'])
        I = int(0.7*len(X_pca))

        trainX, trainY = X_pca[:I], data[cat]['y'][:I]
        testX = X_pca[I:]

        svm = SVC(kernel='linear')
        svm.fit(trainX, trainY)

        pred = svm.predict(testX)
        pred = pred.tolist()

        hx, hy, hz = [], [], []
        kx, ky, kz = [], [], []
        for ix, iy, pz in zip(testX[:, 0], testX[:, 1], pred):
            if pz > 0:
                hx.append(ix); hy.append(iy); hz.append(pz+1)
            if pz < 0:
                kx.append(ix); ky.append(iy); kz.append(pz-1)

        lx, ly, lz = Plane(svm, testX)
        final_dataset[cat] = {'ClassA':{'x':kx,'y':ky,'z':kz,'color':'red'},
                              'ClassB':{'x':hx,'y':hy,'z':hz,'color':'green'},
                              'Surface':{'x':lx,'y':ly,'z':lz}}


        await ws.send(json.dumps(final_dataset))

        
print("Server has booted")
loop = asyncio.get_event_loop()
loop.run_until_complete(websockets.serve(Serving, 'localhost', 8080))
loop.run_forever()
