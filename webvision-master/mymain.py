from __future__ import print_function
import argparse
import config
import eval


data_info = config.LoadInfo()

top_k = 5
eval_val = 'val.txt'

merged_df = eval.MergeValFile(data_info, eval_val)
print ("Top %d accurarcy is %0.3f." % (
    top_k, eval.TopK(merged_df, top_k)))

