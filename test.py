import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=1,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')
parser.add_argument('--yrealy',type=int,default=82,help='sensor_id which will be used to produce the real vs. preds output')
parser.add_argument('--ytest_size',type=int,default=3063,help='timesteps based on TEST dataset')
parser.add_argument('--from_seq_length',type=int,default=0,help='') # para acotar el rango de horizontes temporales a predecir
parser.add_argument('--prediction_multi_or_single',type=str,default='multi', help='indicate the problem as either multi or single, to determine whether predictions are done for multiple time steps sequentially (has generality, less precision) or for a unique time step (more precision, no generality)')
parser.add_argument('--single_prediction_time_step',type=int,default=None, help='indicate which single temporal horizon to be used')
parser.add_argument('--splits',type=int,default=10, help='splits for timesteps')

args = parser.parse_args()




def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit , out_dim=args.seq_length, in_dim=args.in_dim)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.splits, args.prediction_multi_or_single, args.single_prediction_time_step)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y, _, _)  in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []


    if args.prediction_multi_or_single=='single':
        i=args.seq_length-1    

        # pred = scaler.inverse_transform(yhat[:,:,i])
        pred = scaler.inverse_transform(yhat) if args.seq_length == 1 else scaler.inverse_transform(yhat[:,:,i])        
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.single_prediction_time_step, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])    

    else:
        
        for i in range(args.from_seq_length,args.seq_length):
    
            # pred = scaler.inverse_transform(yhat[:,:,i])
            pred = scaler.inverse_transform(yhat) if args.seq_length == 1 else scaler.inverse_transform(yhat[:,:,i])        

            real = realy[:,:,i]
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])


    if args.prediction_multi_or_single=='multi':
    
        log = 'On average over {:.4f} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(args.seq_length,np.mean(amae),np.mean(amape),np.mean(armse)))
    
    if args.addaptadj == True:
        addaptadj_text = "Adapt"
    else:
        addaptadj_text = "NoAdapt"
    
    variant = args.adjdata
    variant = str(str(str(variant.split("/")[2]).split(".")[0]).split("_")[3])

    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./heatmap" + "_" + variant + "_" + addaptadj_text + '.pdf')
        
    
    y_real = np.array([])
    y_hat = np.array([])
    sensor_id = np.array([])
    temporal_horizon = np.array([])


    if args.seq_length==1:

        j = args.seq_length-1
        
        for i in range(args.yrealy):
            
            y_real = np.append(y_real , realy[:, i , j ].cpu().detach().numpy() ) 
            y_hat = np.append(y_hat , scaler.inverse_transform(yhat).cpu().detach().numpy() )
            y_seq_length = np.repeat( j+1 , args.ytest_size) #timesteps test dataset
            temporal_horizon = np.append(temporal_horizon , y_seq_length)

            sensor_yrealy = np.repeat( i+1 , args.ytest_size * args.seq_length)
            sensor_id = np.append(sensor_id , sensor_yrealy)
    
    else:

        for i in range(args.yrealy):
            
            for j in range(args.seq_length):
    
                y_real = np.append(y_real , realy[:, i , j ].cpu().detach().numpy() ) 
                y_hat = np.append(y_hat , scaler.inverse_transform(yhat[:, i , j ]).cpu().detach().numpy() )
                y_seq_length = np.repeat( j+1 , args.ytest_size) #timesteps test dataset
                temporal_horizon = np.append(temporal_horizon , y_seq_length)        

            sensor_yrealy = np.repeat( i+1 , args.ytest_size * args.seq_length)
            sensor_id = np.append(sensor_id , sensor_yrealy)
        
    timesteps = np.tile(np.tile(np.arange(args.ytest_size)+1,args.seq_length) ,args.yrealy)
    

    print(f'Shape is {y_real.shape[0]} real values , {y_hat.shape[0]} predictions , {y_seq_length.shape[0]} timesteps , {temporal_horizon.shape[0]} replicated timesteps , {sensor_yrealy.shape[0]} rows per sensor (timesteps * horizons) , {sensor_id.shape[0]} repeated sensors')    
    
    df2 = pd.DataFrame({'sensor id': sensor_id,'temporal horizon': temporal_horizon,'timesteps':timesteps, 'real_values': y_real, 'pred_values': y_hat})
    df2.to_csv('./predictions' + '_' + variant + "_" + addaptadj_text + '.csv',index=False)

###     y12 = realy[:,args.yrealy,11].cpu().detach().numpy()
###     yhat12 = scaler.inverse_transform(yhat[:,args.yrealy,11]).cpu().detach().numpy()

###     y1 = realy[:,args.yrealy,0].cpu().detach().numpy()
###     yhat1 = scaler.inverse_transform(yhat[:,args.yrealy,0]).cpu().detach().numpy()

###     df2 = pd.DataFrame({'real1': y1, 'pred1':yhat1 })




if __name__ == "__main__":
    main()
