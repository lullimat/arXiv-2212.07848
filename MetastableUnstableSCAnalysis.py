__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2022 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli"]
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__version__ = "0.1"
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"


from idpy.Utils.ManageData import ManageData
from idpy.Utils.Statements import AllTrue

from idpy.Utils.DictHandle import RunThroughDict, Edit_NPArrayToList, Edit_ListToNPArray, Check_WhichType

import sympy as sp
import numpy as np

def ComputeMeanErr(array):
    _mean_swap = np.mean(array)
    _var_swap = np.var(array)
    _N = len(array)
    _err_swap = np.sqrt(_var_swap / (_N - 1))
    return _mean_swap, _err_swap

def StructureFactors3D(lbm):
    if len(lbm.sims_vars['dim_sizes']) != 3:
        print("Warning! This function is for 3D, you are using it in",
              len(lbm.sims_vars['dim_sizes']), "dimensions!")
    
    first_flag = False
    if 'LUT_cos' not in lbm.sims_vars:
        _dim_sizes = lbm.sims_vars['dim_sizes']
        
        _nx_list = np.arange(0, _dim_sizes[0] // 2)
        lbm.sims_vars['nx_list'] = _nx_list
        '''
        Preparing lookup tables for FT
        '''
        _LUT_cos, _LUT_sin = [], []
        for _nx in _nx_list:
            _LUT_cos += [np.cos(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
            _LUT_sin += [np.sin(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
        lbm.sims_vars['LUT_cos'] = np.array(_LUT_cos)
        lbm.sims_vars['LUT_sin'] = np.array(_LUT_sin)
        '''
        Creating lists for observables
        '''            
        lbm.sims_vars['n2_ft'] = []        
        
        first_flag = True

    _dim_sizes, _V = lbm.sims_vars['dim_sizes'], lbm.sims_vars['V']
    _dim = len(_dim_sizes)
    _n_swap = lbm.sims_idpy_memory['n'].D2H()
    _n_swap = _n_swap.reshape(np.flip(_dim_sizes))

    '''
    Begin FT
    '''
    _LUT_cos = lbm.sims_vars['LUT_cos']
    _LUT_sin = lbm.sims_vars['LUT_sin']
    _nx_list = lbm.sims_vars['nx_list']
    
    _n2_ft = []
    
    for _i_nx, _nx in enumerate(_nx_list):
        _n2_ft_swap = 0

        _Sum_cos_n_x, _Sum_sin_n_x = 0, 0
        _Sum_cos_n_y, _Sum_sin_n_y = 0, 0    
        _Sum_cos_n_z, _Sum_sin_n_z = 0, 0        

        # X-direction
        for _x in range(_dim_sizes[0]):
            _Sum_cos_n_x += np.sum(_n_swap[:,:,_x]) * _LUT_cos[_i_nx][_x]
            _Sum_sin_n_x += np.sum(_n_swap[:,:,_x]) * _LUT_sin[_i_nx][_x]

        # Y-direction
        for _y in range(_dim_sizes[1]):
            _Sum_cos_n_y += np.sum(_n_swap[:,_y,:]) * _LUT_cos[_i_nx][_y]
            _Sum_sin_n_y += np.sum(_n_swap[:,_y,:]) * _LUT_sin[_i_nx][_y]

        # Z-direction
        for _z in range(_dim_sizes[2]):
            _Sum_cos_n_z += np.sum(_n_swap[_z,:,:]) * _LUT_cos[_i_nx][_z]
            _Sum_sin_n_z += np.sum(_n_swap[_z,:,:]) * _LUT_sin[_i_nx][_z]
            
        
        _n2_ft_swap += (_Sum_cos_n_x ** 2 + _Sum_sin_n_x ** 2) / _V       
        _n2_ft_swap += (_Sum_cos_n_y ** 2 + _Sum_sin_n_y ** 2) / _V       
        _n2_ft_swap += (_Sum_cos_n_z ** 2 + _Sum_sin_n_z ** 2) / _V
        _n2_ft_swap /= 3

        _n2_ft += [_n2_ft_swap]

    lbm.sims_vars['n2_ft'] += [_n2_ft]

def StructureFactors2D(lbm):
    if len(lbm.sims_vars['dim_sizes']) != 2:
        print("Warning! This function is for 2D, you are using it in",
              len(lbm.sims_vars['dim_sizes']), "dimensions!")
    
    first_flag = False
    if 'LUT_cos' not in lbm.sims_vars:
        _dim_sizes = lbm.sims_vars['dim_sizes']
        
        _nx_list = np.arange(0, _dim_sizes[0] // 2)
        lbm.sims_vars['nx_list'] = _nx_list
        '''
        Preparing lookup tables for FT
        '''
        _LUT_cos, _LUT_sin = [], []
        for _nx in _nx_list:
            _LUT_cos += [np.cos(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
            _LUT_sin += [np.sin(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
        lbm.sims_vars['LUT_cos'] = np.array(_LUT_cos)
        lbm.sims_vars['LUT_sin'] = np.array(_LUT_sin)
        '''
        Creating lists for observables
        '''            
        lbm.sims_vars['n2_ft'] = []        
        
        first_flag = True

    _dim_sizes, _V = lbm.sims_vars['dim_sizes'], lbm.sims_vars['V']
    _dim = len(_dim_sizes)
    _n_swap = lbm.sims_idpy_memory['n'].D2H()
    _n_swap = _n_swap.reshape(np.flip(_dim_sizes))

    '''
    Begin FT
    '''
    _LUT_cos = lbm.sims_vars['LUT_cos']
    _LUT_sin = lbm.sims_vars['LUT_sin']
    _nx_list = lbm.sims_vars['nx_list']
    
    _n2_ft = []
    
    for _i_nx, _nx in enumerate(_nx_list):
        _n2_ft_swap = 0

        _Sum_cos_n_x, _Sum_sin_n_x = 0, 0
        _Sum_cos_n_y, _Sum_sin_n_y = 0, 0

        # X-direction
        for _x in range(_dim_sizes[0]):
            _Sum_cos_n_x += np.sum(_n_swap[:,_x]) * _LUT_cos[_i_nx][_x]
            _Sum_sin_n_x += np.sum(_n_swap[:,_x]) * _LUT_sin[_i_nx][_x]

        # Y-direction
        for _y in range(_dim_sizes[1]):
            _Sum_cos_n_y += np.sum(_n_swap[_y,:]) * _LUT_cos[_i_nx][_y]
            _Sum_sin_n_y += np.sum(_n_swap[_y,:]) * _LUT_sin[_i_nx][_y]            
        
        _n2_ft_swap += (_Sum_cos_n_x ** 2 + _Sum_sin_n_x ** 2) / _V       
        _n2_ft_swap += (_Sum_cos_n_y ** 2 + _Sum_sin_n_y ** 2) / _V
        _n2_ft_swap /= 2

        _n2_ft += [_n2_ft_swap]

    lbm.sims_vars['n2_ft'] += [_n2_ft]

def StructureFactors1D(lbm):
    if len(lbm.sims_vars['dim_sizes']) != 1:
        print("Warning! This function is for 1D, you are using it in",
              len(lbm.sims_vars['dim_sizes']), "dimensions!")
    
    first_flag = False
    if 'LUT_cos' not in lbm.sims_vars:
        _dim_sizes = lbm.sims_vars['dim_sizes']
        
        _nx_list = np.arange(0, _dim_sizes[0] // 2)
        lbm.sims_vars['nx_list'] = _nx_list
        '''
        Preparing lookup tables for FT
        '''
        _LUT_cos, _LUT_sin = [], []
        for _nx in _nx_list:
            _LUT_cos += [np.cos(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
            _LUT_sin += [np.sin(np.arange(_dim_sizes[0]) * 2. * np.pi * _nx / _dim_sizes[0])]
        lbm.sims_vars['LUT_cos'] = np.array(_LUT_cos)
        lbm.sims_vars['LUT_sin'] = np.array(_LUT_sin)
        '''
        Creating lists for observables
        '''            
        lbm.sims_vars['n2_ft'] = []
        
        first_flag = True

    _dim_sizes, _V = lbm.sims_vars['dim_sizes'], lbm.sims_vars['V']
    _dim = len(_dim_sizes)
    _n_swap = lbm.sims_idpy_memory['n'].D2H()
    _n_swap = _n_swap.reshape(np.flip(_dim_sizes))

    '''
    Begin FT
    '''
    _LUT_cos = lbm.sims_vars['LUT_cos']
    _LUT_sin = lbm.sims_vars['LUT_sin']
    _nx_list = lbm.sims_vars['nx_list']
    
    _n2_ft = []
    
    for _i_nx, _nx in enumerate(_nx_list):
        _n2_ft_swap = 0

        _Sum_cos_n_x, _Sum_sin_n_x = 0, 0
        
        # X-direction
        for _x in range(_dim_sizes[0]):
            _Sum_cos_n_x += _n_swap[_x] * _LUT_cos[_i_nx][_x]
            _Sum_sin_n_x += _n_swap[_x] * _LUT_sin[_i_nx][_x]
        
        _n2_ft_swap += (_Sum_cos_n_x ** 2 + _Sum_sin_n_x ** 2) / _V       
        _n2_ft += [_n2_ft_swap]

    lbm.sims_vars['n2_ft'] += [_n2_ft]


class StructureFactorsData(ManageData):
    dump_file = 'SFactorsData.json'
    obs_list = ['Sk', 'k_range', 'n_start']
    content_list = ['dims', 'Ls', 'n_types', 'As', 'Gs', 'kBTs']
    n = sp.Symbol('n')
    
    def __init__(self, dim = 3, L = None, n_type = None, a = None, G = None, kBT = None, dump_file = None):        
        self.dim, self.L, self.n_type, self.a, self.G, self.kBT = dim, L, n_type, a, G, kBT
        self.values_dict = {'dims': self.dim, 'Ls': self.L, 'n_types': self.n_type, 
                            'As': self.a, 'Gs': self.G, 'kBTs': self.kBT}
        
        if dump_file is not None:
            self.dump_file = dump_file
            
        _chk_all_none = []
        for _opt in self.values_dict:
            _chk_all_none += [self.values_dict[_opt] is None]
    
        ManageData.__init__(self, dump_file = self.dump_file)
        
        if not ManageData.ReadJson(self):
            print("File", self.dump_file, "not found: creating in default state...")
            self.is_data = False
            self.CreateContent()
            ManageData.DumpJson(self, indent = 4)
        elif not AllTrue(_chk_all_none):
            self.is_data = self.CheckContent()
            if not self.is_data:
                self.CreateContent()
            
        
    def CreateContent(self):
        '''
        Check that the content lists have actually been created
        '''
        for _content in self.content_list:
            _key = self.ContentKey(_content)
            if not ManageData.IsThereKey(self, _key):
                ManageData.PushData(self, data = [], key = _key)
        
    def CheckContent(self):                
        '''
        Check for the specific values
        '''
        _chks = []
        for _content in self.content_list:
            if ManageData.IsThereKey(self, self.ContentKey(_content)):
                _chks += [self.values_dict[_content] in ManageData.PullData(self, self.ContentKey(_content))]
            else:
                _chks += [False]

        if AllTrue(_chks):
            return self.PullData(self.DataKeyPrefix())['Sk']['N']
        else:
            return False
    
    def AddContent(self):
        for _content in self.content_list:
            _swap_list = ManageData.PullData(self, self.ContentKey(_content))
            if self.values_dict[_content] is not None:
                _swap_list += [self.values_dict[_content]] 
        
    def ContentKey(self, content):
        if content not in self.content_list:
            raise Exception(content, "not in ", self.content_list)
        else:
            if content == 'dims':
                return 'dims'
            if content == 'Ls':
                return str(self.dim) + '/L'
            if content == 'n_types':
                return str(self.dim) + '/' + str(self.L) + '/n_types'
            if content == 'As':
                return str(self.dim) + '/' + str(self.L) + '/' + self.n_type + '/As'
            if content == 'Gs':
                return str(self.dim) + '/' + str(self.L) + '/' + self.n_type + '/' + str(self.a) + '/Gs'
            if content == 'kBTs':
                return (str(self.dim) + '/' + str(self.L) + '/' + self.n_type + '/' + 
                        str(self.a) + '/' + str(self.G) + '/kBTs')
            
    def GetContent(self, content):
        return ManageData.PullData(self, self.ContentKey(content))
    
    def DataKeyPrefix(self):
        _str = \
            str(self.dim) + '/' + str(self.L) + '/' + self.n_type + '/' + \
            str(self.a) + '/' + str(self.G) + '/' + str(self.kBT)
        return _str
    
    def PushData(self, data_dict = None):
        if data_dict is None:
            raise Exception("invalid value 'data_dict'")
            
        if not self.is_data:
            self.AddContent()
            ManageData.PushData(self, data=data_dict, key=self.DataKeyPrefix())
            RunThroughDict(dictionary=self.data_dictionary, edit_function=Edit_NPArrayToList)
            ManageData.DumpJson(self, indent=4)
        else:
            print("The data seems to already in ", self.dump_file, "(!!!)")
            print("If you want to push the new data you need to erase the old ones first...")
