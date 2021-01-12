import torch
import torch_geometric
import torch_scatter
from mendeleev import element
from pymatgen.core.structure import Structure
from e3nn.point.data_helpers import DataPeriodicNeighbors
import numpy as np
import scipy.constants as const
from mpmath import mp

class ComprehensiveEvaluation:
    
    def __init__(self, cif_namelist, model_kwargs, cif_path='data/', chunk_id=0):
        self.chunk_id = chunk_id
        self.model_kwargs = model_kwargs
        self.cif_strlist = []
        for x in cif_namelist:
            with open(cif_path+x, 'r') as f:
                self.cif_strlist.append(f.read().splitlines())
        self.structures = [Structure.from_str("\n".join(c), "CIF") for c in self.cif_strlist]
        self.encode_structures(self.structures)
        
    def encode_structures(self,structures):
        len_element = 118
        self.data = []
        for i, struct in enumerate(structures):
#             print(f"Encoding sample {i+1:5d}/{len(structures):5d}           for mp-{self.chunk_id:3d}                ", end="\r", flush=True)
            input = torch.zeros(len(struct), len_element)
            for j, site in enumerate(struct):
                input[j, int(element(str(site.specie)).atomic_number)] = element(str(site.specie)).atomic_weight
            self.data.append(DataPeriodicNeighbors(
                x=input, Rs_in=None, 
                pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
                r_max=self.model_kwargs.get('max_radius'), y=torch.zeros(1,50), n_norm=40,
            ))
    
    def predict_phdos(self, data, model, device='cpu'):
        self.phdos = []
        for i in range(len(data)):
#             print(f"Calculating sample {i+1:5d}/{len(data):5d}              for mp-{self.chunk_id:3d}                ", end="\r", flush=True)
            d = torch_geometric.data.Batch.from_data_list([data[i]])
            d.to(device)
            self.phdos.append(model(d.x, d.edge_index, d.edge_attr, n_norm=40, batch=d.batch)[0].cpu().detach().tolist())
    
    def cal_heatcap(self, g, omega, T_lst, structures):
        assert len(g) == len(structures), "Lengths of DOS and structures should be equal"
        omega_hz = np.array(omega[1:])*const.c*100*2*np.pi # wavenumber to circular frequency
        self.C_v_mol = []
        self.C_v_kg = []
        self.phdos_norm = []
        for i, struct in enumerate(structures):
#             print(f"Calculating heat capacity {i+1:5d}/{len(structures):5d} for mp-{self.chunk_id:3d}                ", end="\r", flush=True)
            g_norm = np.array(g[i][1:])/np.trapz(np.array(g[i][1:]),omega_hz)
            self.phdos_norm.append(np.insert(g_norm,0,0).tolist())
            if struct.ntypesp == 1:
                g_norm_xSitesNum = 3*g_norm
            else:
                g_norm_xSitesNum = 3*struct.num_sites*g_norm
            C_v_mol_sub = []
            C_v_kg_sub = []
            for T in T_lst:
                x = const.hbar*omega_hz/(2*const.k*T)
                csch_x, coth_x = np.zeros(x.shape[0]), np.zeros(x.shape[0])
                for i in range(len(x)):
                    csch_x[i], coth_x[i] = float(mp.csch(x[i])), float(mp.coth(x[i]))
                C_v_uc = const.k*np.trapz((csch_x**2)*(x**2)*g_norm_xSitesNum,omega_hz)
                C_v_mol_sub.append(C_v_uc*const.N_A)
                C_v_kg_sub.append(C_v_uc*1000/(struct.density*struct.volume*1e-24))
            self.C_v_mol.append(C_v_mol_sub)
            self.C_v_kg.append(C_v_kg_sub)
        
class AtomEmbeddingAndSumLastLayer(torch.nn.Module):
    def __init__(self, atom_type_in, atom_type_out, model):
        super().__init__()
        self.linear = torch.nn.Linear(atom_type_in, atom_type_out)
        self.model = model
        self.relu = torch.nn.ReLU()
    def forward(self, x, *args, batch=None, **kwargs):
        output = self.linear(x)
        output = self.relu(output)
        output = self.model(output, *args, **kwargs)
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        output = torch_scatter.scatter_add(output, batch, dim=0)
        output = self.relu(output)
        maxima, _ = torch.max(output,axis=1)
        output = output.div(maxima.unsqueeze(1))
        return output
