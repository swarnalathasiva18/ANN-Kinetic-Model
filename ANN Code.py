
def init_weights_he(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



import torch
from torch import nn
import torch.nn.functional as f

#creating the model

class NetWithDropout(nn.Module):
    def __init__(self,H1, H2 , H3 ,H4 ,H5, H6,p,input_size =39,output_size =39):
        super(NetWithDropout, self).__init__()


        self.ANN = nn.Sequential(nn.Linear(input_size,H1),nn.ReLU(),nn.BatchNorm1d(H1),nn.Dropout(p),
                                 nn.Linear(H1, H2),nn.ReLU(),nn.BatchNorm1d(H2),nn.Dropout(p),
                                 nn.Linear(H2,H3),nn.ReLU(),nn.BatchNorm1d(H3),nn.Dropout(p),
                                 nn.Linear(H3, H4),nn.ReLU(),nn.BatchNorm1d(H4),nn.Dropout(p),
                                 nn.Linear(H4, H5),nn.ReLU(),nn.BatchNorm1d(H5),nn.Dropout(p),
                                 nn.Linear(H5, H6),nn.ReLU(),nn.BatchNorm1d(H6),nn.Linear(H6, output_size))



            #model.eval() turns off the dropout function and model.train() uses it
    def forward(self, x):
      x = self.ANN(x)
      return x


class NetWithoutDropout(nn.Module):
    def __init__(self, H1 , H2 , H3 ,H4,H5,H6,input_size = 39,output_size = 39):
        super(NetWithoutDropout, self).__init__()


        self.ANN = nn.Sequential(nn.Linear(input_size,H1),nn.ReLU(),nn.BatchNorm1d(H1),
                                 nn.Linear(H1, H2),nn.ReLU(),nn.BatchNorm1d(H2),
                                 nn.Linear(H2,H3),nn.ReLU(),nn.BatchNorm1d(H3),
                                 nn.Linear(H3, H4),nn.ReLU(),nn.BatchNorm1d(H4),
                                 nn.Linear(H4, H5),nn.ReLU(),nn.BatchNorm1d(H5),
                                 nn.Linear(H5, H6),nn.ReLU(),nn.BatchNorm1d(H6),
                                 nn.Linear(H6, output_size))


            #model.eval() turns off the dropout function and model.train() uses it
    def forward(self, x):
      x = self.ANN(x)
      return x

# training the model  FOR SGD

final_loss1, final_loss2, final_loss3 = 0,0,0


def objective(trial):
  params = {'H1':trial.suggest_categorical('H1',[512,1024,2048]),
            'H2':trial.suggest_categorical('H2', [256,512,1024]),
            'H3':trial.suggest_categorical('H3', [128,256,512]),
            'H4':trial.suggest_categorical('H4', [64,128,256]),
            'H5':trial.suggest_categorical('H5', [32,64,128]),
            'H6':trial.suggest_categorical('H6', [0,16,32,64])}
  model = NetWithDropout(**params)
  learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 0.1)
  dropout = trial.suggest_uniform('p',0.1,0.7)
  epochs = trial.suggest_int('epochs',400,600)
  eta_min = trial.suggest_loguniform('eta_min',1e-8,1e-6)
  momentum = trial.suggest_uniform('momentum',0.0,1.0)
  weight_decay = trial.suggest_uniform('weight_decay',0.0001,0.1)
  epochs = epochs
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum ,weight_decay = weight_decay)
  scheduler = CosineAnnealingLR(optimizer,T_max = 100, eta_min = eta_min )

  for epoch in range(epochs):
    for x_tra,y_tra in dataloader_train :
      model.train()
      optimizer.zero_grad()
      torch.set_default_dtype(torch.float64)
      yhat_tra = model(x_tra.view(-1,39))
      loss_train = criterion(y_tra, yhat_tra)


      #L1 regularization
      l1_lambda = 0.00001
      l1_norm = sum(p.abs().sum() for p in model.parameters())
      loss_train = loss_train + l1_lambda*l1_norm
      loss_train.backward()
      optimizer.step()


  #storing the parameter value for testing
  #torch.save(model.state_dict(),'model_params.pth')

#model evaluation
 r2sco = R2Score()
 r = r2sco(y_tra,yhat_tra)


def validation(model):
  for x_val, y_val in dataloader_val:
    model_val =model
    model_val.load_state_dict(torch.load('model_params.pth'))
    model_val.eval()
    yhat_val = model_val(x_val.view(-1, 39))
    loss = criterion(yhat_val, y_val)
    final_loss2 += loss.item()

    #creating the dataframe of predicted values
    pred_rates = yhat_val.detach().numpy()
    pred_df_val = pd.DataFrame(pred_rates, columns = df_y.columns)

#model evaluation
  r2sco = R2Score
  R_squared =r2sco(y_tra,yhat_tra)
  print('R^2 value of validation data:',R_squared)
  print('loss of validation data',final_loss2/len(df_y_val.shape[0])*100, '%')


  return(pred_df_val)


study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=100)

# Access the best hyperparameters
best_params = study.best_params()
best_trial = study.best_trial

#keeping the session alive untill the cell is running

import time

while True:
    time.sleep(600)
    print("Keep-alive")

def objective(trial):
  params = {'H1':512,'H2':256,'H3':128,'H4':64,'H5':32,'H6':16,'p':trial.suggest_uniform('p',0.0, 0.6)}


  model = NetWithDropout(**params)
  model.apply(init_weights_he)
  learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
  epochs = trial.suggest_int('epochs',400,600)
  l1_lambda = trial.suggest_loguniform('l1_lambda',1e-6,1e-4)
  weight_decay = trial.suggest_uniform('weight_decay',0.001,0.1)
  gamma = trial.suggest_uniform('gamma',0.0,1.0)
  step_size = trial.suggest_int('step_size',10,100)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999) ,weight_decay = weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = step_size, gamma = gamma)

  for epoch in range(epochs):
    for x_tra,y_tra in dataloader_train :
      model.train()
      optimizer.zero_grad()
      torch.set_default_dtype(torch.float64)
      yhat_tra = model(x_tra.view(-1,39))
      loss_train = criterion(y_tra, yhat_tra)


      #L1 regularization
      l1_lambda = l1_lambda
      l1_norm = sum(p.abs().sum() for p in model.parameters())
      loss_train = loss_train + l1_lambda*l1_norm
      loss_train.backward()
      optimizer.step()
      scheduler.step()




    for x_val,y_val in dataloader_val:
      yhat_val = model(x_val.view(-1,39))
      loss_val = criterion(yhat_val,y_val)

    y_val = y_val.detach().numpy()
    yhat_val = yhat_val.detach().numpy()
    R_squared = r2_score(y_val,yhat_val)


  #storing the parameter value for testing
  #torch.save(model.state_dict(),'model_params.pth')
  return R_squared

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 30)
best_params = study.best_params
best_trial = study.best_trial





