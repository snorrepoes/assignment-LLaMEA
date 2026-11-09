import sys,pathlib,numpy as np,torch,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/opt/work/bo_assignment")
from boa.problems import load_config,make_recorder,BudgetExhausted
from boa.initial_design import initial_points
torch.set_num_threads(1)
path,seed,B=sys.argv[1],int(sys.argv[2]),150
cfg=load_config();rec,_=make_recorder("lj1125",B,cfg)
ns={};exec(compile(open(path).read(),path,"exec"),ns)
try: ns["Optimize_with_BO"](budget=B,dim=6,random_seed=seed,initial_points=initial_points(12,6,seed))(rec)
except BudgetExhausted: pass
v=np.array(rec.values);print(path.split("/")[-1],seed,"best",v.min(),"regret",v.min()+1)
