#--------------  run the simulations in the paper 
# run setting 1 of the simulation in the paper
# runs 1 replication of the simulation and save model results (on the test set) to an excel file.
# Each file runs one setting, and results from all settings will be stored in one excel file under different tabs.
# implements both Bridge and Bridge Naive version, to compare columns in the excel file for the RMSEs across 4 methods
# can modify "gene_prop" (gene proportions) for different overlapping gene proporions: np.array([[0,0.6], [0.5, 1]]), np.array([[0,0.55], [0.5, 1]]), np.array([[0,0.7], [0.3, 1]])
# can modify "n_sub" for different sample size: 500, 1500
# can modify "id" for different seed


# import functions
from bridge_functions import *
from simulation_functions import * 


# modify the following to save the simulation results
excel_file_name = "res_sim" 
excel_file_path = "sim_results/"

# Create the folder if it doesn't already exist
if not os.path.exists(excel_file_path):
    os.makedirs(excel_file_path)

WZ_form = "linear"

excel_tab = "setting1_n1k5b_ol0d2"  # this is the simulation setting (linear expression, training sample size 1000, test sample size 500)

n_epochs_train = 3000 
n_epochs_test = 100 
penalty_train  = 0.005
penalty_test = 0.005

mask1_percent_train = 0
mask1_percent_test = 0

run_test = True

test_size = 500
train_size = 1000


# id: seed for running the simulation
id = 1

test_sim_seed = 999 + id

### add id to the excel file name
excel_file_name = excel_file_name+"_"+str(id)


##-------- simulation parameters# simulate orthogonal beta & W. beta is the coef
n_gene = 600
n_sub = train_size
n_factor = 5
center_prop = [0.5, 0.5]
gene_prop = np.array([[0,0.7], [0.5, 1]])
px = len(center_prop)
n_contV = 1; 
n_binaryV = 1; 
pv = n_contV + n_binaryV
V_sd = 1

Z_sd = 10


random.seed(16)
np.random.seed(16)
beta = np.random.normal(loc=2, scale=1, size=(n_gene, px))
beta[:,0] = beta[:,0] - 4 
beta[:,1] = beta[:,1] - 7 

# simulate orthogonal W
W = stats.ortho_group.rvs(n_gene)  
W = W[:, 0:n_factor]

# simulate gamma_cont
gamma_cont = np.random.normal(loc = 1, scale=1, size = (n_gene, n_contV))
# simulate gamma_binary
gamma_binary =np.random.normal(loc = -3, scale=1, size = (n_gene, n_contV))

# parameter of Z for outcome O
O_coef = np.array([1, 1, 1, 1, 1]).reshape((-1,1))

# parameter of V for outcome O
O_coef_V = np.array([10, 10]).reshape((-1,1))

# variance of O
O_scale = 5


##------------------- Simulate training data 
dat_s = sim_wCovariates(n_sub=n_sub, beta=copy.copy(beta), W=W, gamma_cont=gamma_cont, gamma_binary=gamma_binary, 
                Z_sd=Z_sd, V_sd=V_sd,
                center_prop=center_prop, gene_prop=gene_prop, seed=id, evaluate_on="all", 
                mask1_seed=id, mask1_percent=mask1_percent_train, 
                O_coef = O_coef, O_coef_V=O_coef_V, O_scale=O_scale, 
                WZ_form = WZ_form)


##################  Modify the followig model parameters  ##################
n_factor_train = n_factor

model_params = {
    'n_gene': n_gene, 
    'n_factor': n_factor_train,          
    'px' : px,
    'pv': pv,
    'W_orthonormal' : True,  
    'beta_W_orth':False, 
    'gamma_W_orth':False,
    'Z_demean_bycenter':True, 
    'Z_scale_bycenter':False,
    'Z_orthogonal': True,   
    'penalty':penalty_train,      
    'nquantile':4,
    'l1_gamma':0, 
    'l1_W':0,
    'l1_beta':0,
}
model_params_test = {
    'n_gene': n_gene, 
    'n_factor': n_factor_train,          
    'px' : px,
    'pv': pv,
    'W_orthonormal' : False,    # needs to be False for test
    'beta_W_orth':False, 
    'gamma_W_orth':False,
    'Z_demean_bycenter':True, 
    'Z_scale_bycenter':False,
    'Z_orthogonal': True,     
    'penalty':penalty_test,  
    'nquantile':4,
    'l1_gamma':False, 
    'l1_W':False, 
    'l1_beta':0, 
}

# moel training parameters
train_model_params = {
    'n_epochs' : n_epochs_train,
    'batch_size':20000, 
    'early_stop':False, 
    'min_delta':0.001, 
    'patience':None,
    'trueZW' : dat_s['WZ'], 'trueBeta' : dat_s['beta'], 'trueGamma' : dat_s['gamma'], 
    'plot_every':20000, 'plot_atEnd':False, 'print_process':True
}


lr = 0.01



#--------------- training model
# create the training dataset
dataset_s = PrepareData(dat_s['Y'], dat_s['X'], V=dat_s['V'], weight=None)
# create the training model 
model_s =  model_ZV(n_sub = n_sub, **model_params)
# train the model
opt_s = optim.Adam(model_s.parameters(), lr=lr)
model_s_train = train_model(model=model_s, dataset=dataset_s, 
mask_ind_object = dat_s['mask_obj'], optimizer=opt_s, **train_model_params)

#-------------- Variance of O explained by true Z and V and fitted Z and V in OLS on Train
# V + true Z 
ols_trueZ = run_ols(dat_s['O'], np.hstack((dat_s['V'], dat_s['Z'])))

# V only 
ols_V = run_ols(dat_s['O'], dat_s['V'])

# V + overlapping Y
ols_VY = run_ols(dat_s['O'], np.hstack((dat_s['V'], dat_s['Y_obs'].dropna(axis = 1))))

# V + estimated Z from Training model 
ols_trainZ = run_ols(dat_s['O'], np.hstack((dat_s['V'], model_s.Z.detach().numpy())))



####-------------- Then apply the model to one independent large test set 
if run_test:
    
    # simulate test data (use above the same parameter unless respecify below)
    dat_test = sim_wCovariates(n_sub=test_size, beta=beta, W=W, gamma_cont=gamma_cont, gamma_binary=gamma_binary, 
                Z_sd=Z_sd, V_sd=V_sd,
                center_prop=center_prop, gene_prop=gene_prop, seed=test_sim_seed, evaluate_on="all", 
                mask1_seed=id, mask1_percent=mask1_percent_test, 
                O_coef = O_coef, O_coef_V=O_coef_V, O_scale=O_scale, 
                WZ_form = WZ_form)

    train_model_params_test = {
        'n_epochs' : n_epochs_test, 
        'batch_size':20000, 
        'early_stop':False, 
        'min_delta':0.001, 
        'patience':None,
        'trueZW' : dat_test['WZ'], 'trueBeta' : dat_test['beta'], 'trueGamma' : dat_test['gamma'], 
        'plot_every':20000, 'plot_atEnd':False, 'print_process':True
    }
    
    # create the model
    if hasattr(model_s, "gamma"):
        model_test = model_ZV(
            n_sub = test_size, 
            **model_params_test, 
            fix_W_at = model_s.W.weight,
            fix_beta_at = model_s.beta, 
            fix_gamma_at = model_s.gamma)
    else:
        model_test = model_ZV(
            n_sub = test_size, 
            **model_params_test, 
            fix_W_at = model_s.W.weight,
            fix_beta_at = model_s.beta)


    # create the Test dataset
    dataset_test = PrepareData(dat_test['Y'], dat_test['X'], V=dat_test['V'], weight=None)
    # train the Test model
    opt_test = optim.Adam(model_test.parameters(), lr=lr)

    model_s_train_test = train_model(model=model_test, dataset=dataset_test, mask_ind_object = dat_test['mask_obj'], optimizer=opt_test, **train_model_params_test)



    ###-------------- Run the linear regression with outcome O and see how much variance explained (Test, under the True Z and learned Z)
    # V + true Z 
    pred_test = ols_trueZ['model'].predict( sm.add_constant(np.hstack((dat_test['V'], dat_test['Z']))))
    ols_trueZ_test = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_trueZ_test_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V only
    pred_test = ols_V['model'].predict( sm.add_constant(dat_test['V']))
    ols_testV = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testV_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V + overlapping Y
    pred_test = ols_VY['model'].predict( sm.add_constant(np.hstack((dat_test['V'], dat_test['Y_obs'].dropna(axis = 1)))))
    ols_testVY = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testVY_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V + estimated Z from Training model 
    pred_test = ols_trainZ['model'].predict( sm.add_constant(np.hstack((dat_test['V'], model_test.Z.detach().numpy()))))
    ols_testZ = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testZ_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 



    ##------------- write Test model results to excel
    append_modelres_excel_s(model=model_test, mask_obj=dat_test['mask_obj'], model_type = "bridge_test",
        model_train=model_s_train_test, eval_model_obj=None, 
        dat = dat_test, obs_Y=dat_test['Y_obs'],
        preYprob_c = None, preY_c = None, plot=False,
        file=excel_file_path+excel_file_name+".xlsx",
        sheetname = excel_tab,
        notes = id,
        rmse_trueZ = ols_trueZ_test_rmse, 
        rmse_estZ = ols_testZ_rmse,
        rmse_V = ols_testV_rmse, 
        rmse_VY = ols_testVY_rmse,
        )




############################################ Run the noX model
model_params_noX = {
    'n_gene': n_gene, 
    'n_factor': n_factor_train,          
    'px' : 1,   
    'pv': 0,     
    'W_orthonormal' : True,  
    'beta_W_orth':False, 
    'gamma_W_orth':False,
    'Z_demean_bycenter':True, 
    'Z_scale_bycenter':False,
    'Z_orthogonal': True,
    'penalty':0,       
    'nquantile':4,
    'l1_gamma':0, 
    'l1_W':0,
    'l1_beta':0,
}
model_params_test_noX = {
    'n_gene': n_gene, 
    'n_factor': n_factor_train,          
    'px' : 1,   
    'pv': 0,    
    'W_orthonormal' : False,    # here needs to be False for test
    'beta_W_orth':False, 
    'gamma_W_orth':False,
    'Z_demean_bycenter':True, 
    'Z_scale_bycenter':False,
    'Z_orthogonal': True,
    'penalty':0,           
    'nquantile':4,
    'l1_gamma':False, 
    'l1_W':False, 
    'l1_beta':0, 
}

# moel training parameters
train_model_params_noX = {
    'n_epochs' : n_epochs_train,
    'batch_size':20000, 
    'early_stop':False, 
    'min_delta':0.001, 
    'patience':None,
    'plot_every':200000, 'plot_atEnd':False, 'print_process':True
}


lr = 0.01



#--------------- training model (NO X)
# create the training dataset
X = pd.DataFrame(np.ones(dat_s['Y'].shape[0]).reshape((-1,1)))  # !!!
dataset_noX = PrepareData(dat_s['Y'], X, V=None, weight=None)

# create the training model 
model_noX =  model_ZV(n_sub = n_sub, **model_params_noX)
# train the model
opt_noX = optim.Adam(model_noX.parameters(), lr=lr)
model_noX_train = train_model(model=model_noX, dataset=dataset_noX, mask_ind_object = dat_s['mask_obj'], optimizer=opt_noX, **train_model_params_noX)

#-------------- Variance of O explained by true Z and V and fitted Z and V in OLS on Train
# V + true Z 
ols_trueZ = run_ols(dat_s['O'], np.hstack((dat_s['V'], dat_s['Z'])))

# V only 
ols_V = run_ols(dat_s['O'], dat_s['V'])

# V + overlapping Y
ols_VY = run_ols(dat_s['O'], np.hstack((dat_s['V'], dat_s['Y_obs'].dropna(axis = 1))))

# V + estimated Z from Training model 
ols_trainZ = run_ols(dat_s['O'], np.hstack((dat_s['V'], model_noX.Z.detach().numpy())))




####-------------- Then apply the model to one independent large test set (No X)
if run_test:

    train_model_params_test_noX = {
        'n_epochs' : n_epochs_test, 
        'batch_size':20000, 
        'early_stop':False, 
        'min_delta':0.001, 
        'patience':None,
        'plot_every':200000, 'plot_atEnd':False, 'print_process':True
    }
    
    # create the model (no X)
    if hasattr(model_noX, "gamma"):
        model_noX_test = model_ZV(
            n_sub = test_size, 
            **model_params_test_noX, 
            fix_W_at = model_noX.W.weight,
            fix_beta_at = model_noX.beta, 
            fix_gamma_at = model_noX.gamma)
    else:
        model_noX_test = model_ZV(
            n_sub = test_size, 
            **model_params_test_noX, 
            fix_W_at = model_noX.W.weight,
            fix_beta_at = model_noX.beta)


    # train the Test model (no X)
    # create the test dataset
    X_test = pd.DataFrame(np.ones(dat_test['Y'].shape[0]).reshape((-1,1)))  # !!!
    dataset_noX_test = PrepareData(dat_test['Y'], X_test, V=None, weight=None)

    opt_noX_test = optim.Adam(model_noX_test.parameters(), lr=lr)

    model_noX_train_test = train_model(model=model_noX_test, dataset=dataset_noX_test, mask_ind_object = dat_test['mask_obj'], optimizer=opt_noX_test, **train_model_params_test_noX)



    ###-------------- Run the linear regression with outcome O and see how much variance explained (Test, under the True Z and learned Z)  (no X)
    # V + true Z 
    pred_test = ols_trueZ['model'].predict( sm.add_constant(np.hstack((dat_test['V'], dat_test['Z']))))
    ols_trueZ_test = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_trueZ_test_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V only
    pred_test = ols_V['model'].predict( sm.add_constant(dat_test['V']))
    ols_testV = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testV_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V + overlapping Y
    pred_test = ols_VY['model'].predict( sm.add_constant(np.hstack((dat_test['V'], dat_test['Y_obs'].dropna(axis = 1)))))
    ols_testVY = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testVY_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 

    # V + estimated Z from Training model 
    pred_test = ols_trainZ['model'].predict( sm.add_constant(np.hstack((dat_test['V'], model_test.Z.detach().numpy()))))
    ols_testZ = 1 - (np.sum((dat_test['O'] - pred_test.reshape(-1,1))**2) / np.sum((dat_test['O'] - np.mean(dat_test['O']))**2))
    ols_testZ_rmse = mean_squared_error(dat_test['O'], pred_test.reshape(-1,1), squared=False) 




    ##------------- write Test model results to excel (no X)
    append_modelres_excel_s(model=model_noX_test, mask_obj=dat_test['mask_obj'], model_type = "naive_bridge_test",
        model_train=model_noX_train_test, eval_model_obj=None, 
        dat = dat_test, obs_Y=dat_test['Y_obs'],
        preYprob_c = None, preY_c = None, plot=False,
        file=excel_file_path+excel_file_name+".xlsx",
        sheetname = excel_tab,
        notes = id, 
        rmse_trueZ = ols_trueZ_test_rmse, 
        rmse_estZ = ols_testZ_rmse,
        rmse_V = ols_testV_rmse, 
        rmse_VY = ols_testVY_rmse,
        )


