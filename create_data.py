import numpy as n
import pandas as p

num=500

"""
run this file to generate your own dataset!
"""

n.random.seed(5007)

x_test = 3*n.random.rand(num,1)
y_test = 9 + 2*x_test + n.random.rand(num,1)

dict_info = {'Brightness' : x_test.reshape(-1,),
             'True Size' : y_test.reshape(-1,)}

input_df = p.DataFrame(dict_info)

input_df.to_csv('input_star_data.csv', index=False)