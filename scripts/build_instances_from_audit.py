"""Build instances/uaflp/*.json from the audited sources (Manus audit folder UAFLP_instances_verified + Mendeley AB20). Kept for provenance; paths are local."""
import json,ast,numpy as np
U='/mnt/user-data/uploads/UAFLP_instances_verified/'
def lit(x): return ast.literal_eval(x) if isinstance(x,str) else x
def tri(W):
    n=len(W);return [[i+1,j+1,round(float(W[i][j]),6)] for i in range(n) for j in range(i+1,n) if abs(W[i][j])>0]
def num(x):
    try: return float(x)
    except: return 0.0
OBJ='sum_{i<j} w_ij * (|cx_i-cx_j| + |cy_i-cy_j|), c = rectangle centroid'
out={}
o=json.load(open(U+'O9.json'))['gym_flp_record'];F=np.array([[num(v) for v in r] for r in o['flows']]);F=F+F.T
out['O9']=dict(name='O9',n=9,W=12.0,H=13.0,areas=[float(a) for a in o['areas']],weights=tri(F),shape={'type':'Rmax','value':4},objective=OBJ,
  provenance={'status':'indirectly_verified','data':'gym-flp commit efbd8b0 continuous.json (no public original table found)',
   'verification':'Replaying Konak et al. 2006 Table 3 FBS solution 7-8|4-1-2|3-6-9-5 with bays along the 13-long side gives 241.06, equal to the paper (tests/test_replay.py)',
   'geometry':'Goncalves & Resende 2015 Table 8 p.19','origin':'Meller, Narayanan & Vance 1998'})
d=json.load(open(U+'O7.json'))['authoritative_record'];W=np.zeros((7,7))
for p in d['all_21_unordered_pair_weights']:
    i,j=lit(p['pair']);W[i-1][j-1]=W[j-1][i-1]=float(p['weight'])
out['O7']=dict(name='O7',n=7,W=8.54,H=13.0,areas=[float(a) for a in d['areas']],weights=tri(W),shape={'type':'Rmax','value':4},objective=OBJ,
  provenance={'status':'verified','data':'MINLPLib o7.mod objective (21 unordered pairs)','geometry':'Goncalves & Resende 2015 Table 8','origin':'Meller, Narayanan & Vance 1998'})
d=json.load(open(U+'Ba12.json'))['authoritative_record'];W=np.zeros((12,12))
for k,v in lit(d['objective_coefficient_c_ij']['nonzero_pairs']).items():
    i,j=lit(k);W[i-1][j-1]=W[j-1][i-1]=float(v)
assert int((np.triu(W,1)>0).sum())==59
out['Ba12']=dict(name='Ba12',n=12,W=10.0,H=6.0,areas=[float(a) for a in d['areas']],weights=tri(W),shape={'type':'min_side','value':1},objective=OBJ,
  provenance={'status':'verified','data':'Westerlund, Eronen & Makela 2022, Optimization 71(14), Appendix Tables A1-A3 p.4176','origin':'Bazaraa 1975'})
d=json.load(open(U+'vC10.json'))['authoritative_record'];M=d['flow']['upper_triangle_matrix_with_omitted_cells_as_null'];W=np.zeros((10,10))
for i in range(10):
    for j in range(i+1,10):
        v=M[i][j]
        if v is not None: W[i][j]=W[j][i]=float(v)
out['vC10s']=dict(name='vC10s',n=10,W=25.0,H=51.0,areas=[float(a) for a in d['areas_m2']],weights=tri(W),shape={'type':'min_side','value':5},objective=OBJ,
  provenance={'status':'verified','data':'Masel thesis (Ohio Univ.) Tables A.1/A.2 p.85, transcription of van Camp et al. 1992','geometry':'Goncalves & Resende 2015 Table 8; Jeong & Chae 2026 Table 2','origin':'van Camp, Carter & Vannelli 1992'})
s=json.load(open(U+'SC35.json'));A=np.array([[num(v) for v in r] for r in s['authoritative_record']['flows']['matrix']]);W=A+A.T
out['SC35']=dict(name='SC35',n=35,W=16.0,H=15.0,areas=[float(a) for a in s['gym_flp_record']['areas']],weights=tri(W),shape={'type':'Rmax','value':4},objective=OBJ,
  provenance={'status':'verified_flows_corrected','data':'flows: Qi Liu 2004 PhD thesis (Virginia Tech) Appendix A.3 p.191; areas: gym-flp PRN (sum 192 = 80% of 16x15, matches Liu Table 4.7)',
   'corrections_vs_gym_flp':'(5,6) 8.92 not 8.2; (6,13) 0 not 0.61; (6,17) 0.61 not 0; (15,32) 53.42 not 0; (16,32) 1.83 not 0',
   'geometry':'Goncalves & Resende 2015 Table 8 (16x15, Rmax=4)','origin':'Liu & Meller 2007'})
M=np.loadtxt('instances/sources/ab20_mendeley/AB20_Armour_Buffa.csv',delimiter=',',encoding='utf-8-sig');W=M+M.T
out['AB20']=dict(name='AB20',n=20,W=30.0,H=20.0,areas=[27.,18,27,18,18,18,9,9,9,24,60,42,18,24,27,75,64,41,27,45],weights=tri(W),shape={'type':'Rmax','value':4},
  objective='sum_i sum_j c_ij f_ij d_ij = sum_{i<j} w_ij d_ij, w_ij = M_ij + M_ji, M = Mendeley weighted matrix (c*f)',
  provenance={'status':'verified','data':'Mendeley Data DOI 10.17632/bv3zywjyyj.1 (CC BY 4.0) AB20_Armour_Buffa.csv; areas Chae & Jeong 2026 Table 4',
   'note':'gym-flp AB20 unusable: no cost matrix and row-1 flows shifted one column. Floor 2x3 in some papers is the ratio of 30x20.','geometry':'Goncalves & Resende 2015 Table 8','origin':'Armour & Buffa 1963'})
for k,v in out.items():
    A=sum(v['areas']);assert A<=v['W']*v['H']+1e-6,k
    json.dump(v,open(f'instances/uaflp/{k}.json','w'),indent=1)
    print(k,v['n'],'pairs',len(v['weights']),'area',A,'floor',v['W']*v['H'])
