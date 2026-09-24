\* Pipeline_execution_time_minimize *\
Minimize
OBJ: pipeline_total_time
Subject To
_C1: encoder_1_0_0 = 0
_C10: decoder_2_0_1 = 0
_C11: decoder_3_0_0 = 0
_C12: decoder_3_0_1 = 0
_C13: encoder_0_0_0 + encoder_0_0_1 + encoder_4_0_0 + encoder_4_0_1 = 12
_C14: decoder_0_0_0 + decoder_0_0_1 + decoder_4_0_0 + decoder_4_0_1 = 12
_C15: layer_frontier_1_0_0 >= 0
_C16: - 1e-06 decoder_0_0_0 - 1e-06 decoder_4_0_0 + layer_frontier_1_0_1 >= 0
_C17: encoder_0_0_0 + 1000000 layer_frontier_1_0_0 <= 1000000
_C18: encoder_4_0_0 + 1000000 layer_frontier_1_0_0 <= 1000000
_C19: encoder_0_0_1 + 1000000 layer_frontier_1_0_1 <= 1000000
_C2: encoder_1_0_1 = 0
_C20: encoder_4_0_1 + 1000000 layer_frontier_1_0_1 <= 1000000
_C21: - 1e-06 decoder_4_0_0 + recompute_frontier_0_0_0 >= 0
_C22: - 1e-06 decoder_4_0_1 + recompute_frontier_0_1_0 >= 0
_C23: encoder_0_0_0 + 1000000 recompute_frontier_0_0_0 <= 1000000
_C24: encoder_0_0_1 + 1000000 recompute_frontier_0_1_0 <= 1000000
_C25: - 12 decoder_0_0_0 - 16 decoder_4_0_0 - 15 encoder_0_0_0
 - 20 encoder_4_0_0 + max_stage_time >= 30
_C26: - 12 decoder_0_0_1 - 16 decoder_4_0_1 - 15 encoder_0_0_1
 - 20 encoder_4_0_1 + max_stage_time >= 60
_C27: - 12 decoder_0_0_0 - 12 decoder_0_0_1 - 16 decoder_4_0_0
 - 16 decoder_4_0_1 - 15 encoder_0_0_0 - 15 encoder_0_0_1 - 20 encoder_4_0_0
 - 20 encoder_4_0_1 + var_sum_FPi_BPi >= 0
_C28: - max_stage_time + pipeline_total_time >= 0
_C29: memory_overhead_0 >= 0
_C3: encoder_2_0_0 = 0
_C30: 400 decoder_0_0_0 + 160 decoder_4_0_0 + 500 encoder_0_0_0
 + 200 encoder_4_0_0 + memory_overhead_0 <= 79500
_C31: memory_overhead_1 >= 0
_C32: 240 decoder_0_0_1 + 120 decoder_4_0_1 + 300 encoder_0_0_1
 + 150 encoder_4_0_1 + memory_overhead_1 <= 79500
_C4: encoder_2_0_1 = 0
_C5: encoder_3_0_0 = 0
_C6: encoder_3_0_1 = 0
_C7: decoder_1_0_0 = 0
_C8: decoder_1_0_1 = 0
_C9: decoder_2_0_0 = 0
Bounds
 0 <= decoder_0_0_0
 0 <= decoder_0_0_1
 0 <= decoder_1_0_0
 0 <= decoder_1_0_1
 0 <= decoder_2_0_0
 0 <= decoder_2_0_1
 0 <= decoder_3_0_0
 0 <= decoder_3_0_1
 0 <= decoder_4_0_0
 0 <= decoder_4_0_1
 0 <= encoder_0_0_0
 0 <= encoder_0_0_1
 0 <= encoder_1_0_0
 0 <= encoder_1_0_1
 0 <= encoder_2_0_0
 0 <= encoder_2_0_1
 0 <= encoder_3_0_0
 0 <= encoder_3_0_1
 0 <= encoder_4_0_0
 0 <= encoder_4_0_1
 0 <= memory_overhead_0
 0 <= memory_overhead_1
Generals
decoder_0_0_0
decoder_0_0_1
decoder_1_0_0
decoder_1_0_1
decoder_2_0_0
decoder_2_0_1
decoder_3_0_0
decoder_3_0_1
decoder_4_0_0
decoder_4_0_1
encoder_0_0_0
encoder_0_0_1
encoder_1_0_0
encoder_1_0_1
encoder_2_0_0
encoder_2_0_1
encoder_3_0_0
encoder_3_0_1
encoder_4_0_0
encoder_4_0_1
memory_overhead_0
memory_overhead_1
Binaries
layer_frontier_1_0_0
layer_frontier_1_0_1
recompute_frontier_0_0_0
recompute_frontier_0_1_0
End
