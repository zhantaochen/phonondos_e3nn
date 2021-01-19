from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
import pandas as pd
import sys

# To access the Materials Project dataset via pymatgen, you will need an API key which can obtain via
# https://www.materialsproject.org/dashboard
# See https://pymatgen.org/pymatgen.ext.matproj.html for more details.

# If you have set your Materials Project API key in your ~/.pmgrc.yaml with (in the command line)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        USER_API_KEY = sys.argv[1]
    else:
        USER_API_KEY = None

    icsd_pd = pd.read_csv('data/mp_data.csv', header=None)
    with MPRester(USER_API_KEY) as m:
        for result in m.query(
            criteria={'material_id':  {"$in": icsd_pd.iloc[:, 0].to_list()}},
            properties=['material_id', 'cif'], chunk_size=len(icsd_pd)
        ):
            with open('data/'+result['material_id']+'.cif', 'w') as f:
                f.write(result['cif'])
    print('Download complete!')
