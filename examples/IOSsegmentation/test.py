import sys
import os


#the datamaker and ImageAnalysisComponents utilities reside two levels above this notebook.
utility_path = os.path.realpath(os.path.join(os.path.pardir, os.path.pardir))
sys.path.append(utility_path)
from regnmf import datamaker
from regnmf import ImageAnalysisComponents as ia



param = {'act_time': [0.01, 0.1, 0.3, 0.8, 1.0, 1.0],
         'cov': 0.3,
         'latents': 40,
         'mean': 0.2,
         'no_samples': 50,
         'noisevar': 0.2,
         'shape': (50, 50),
         'width':0.1,
         'var': 0.08}

data = datamaker.Dataset(param)



anal_param = {'sparse_param': 0.5,
              'factors': 80,
              'smooth_param': 2,
              'init':'convex',
              'sparse_fct':'global_sparse',
              'verbose':0
              }



input_data = ia.TimeSeries(data.observed, shape=param['shape'])


nmf_cuda = ia.NNMF_cuda(maxcount=50, num_components=anal_param['factors'], **anal_param)

'''
pr = cProfile.Profile()
pr.enable()
'''

nmf_cuda_out = nmf_cuda(input_data)

'''
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
'''


nmf = ia.NNMF(maxcount=50, num_components=anal_param['factors'], **anal_param)
'''
import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()
'''
nmf_out = nmf(input_data)

print "done"

'''
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
'''







