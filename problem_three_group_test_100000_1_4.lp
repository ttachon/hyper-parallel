\* Pipeline_execution_time_minimize *\
Minimize
OBJ: pipeline_total_time
Subject To
_C1: group_a_1_0_0 = 0
_C10: group_a_3_0_1 = 0
_C11: group_a_3_0_2 = 0
_C12: group_a_3_0_3 = 0
_C13: group_b_1_0_0 = 0
_C14: group_b_1_0_1 = 0
_C15: group_b_1_0_2 = 0
_C16: group_b_1_0_3 = 0
_C17: group_b_2_0_0 = 0
_C18: group_b_2_0_1 = 0
_C19: group_b_2_0_2 = 0
_C2: group_a_1_0_1 = 0
_C20: group_b_2_0_3 = 0
_C21: group_b_3_0_0 = 0
_C22: group_b_3_0_1 = 0
_C23: group_b_3_0_2 = 0
_C24: group_b_3_0_3 = 0
_C25: group_c_1_0_0 = 0
_C26: group_c_1_0_1 = 0
_C27: group_c_1_0_2 = 0
_C28: group_c_1_0_3 = 0
_C29: group_c_2_0_0 = 0
_C3: group_a_1_0_2 = 0
_C30: group_c_2_0_1 = 0
_C31: group_c_2_0_2 = 0
_C32: group_c_2_0_3 = 0
_C33: group_c_3_0_0 = 0
_C34: group_c_3_0_1 = 0
_C35: group_c_3_0_2 = 0
_C36: group_c_3_0_3 = 0
_C37: group_a_0_0_0 + group_a_0_0_1 + group_a_0_0_2 + group_a_0_0_3
 + group_a_4_0_0 + group_a_4_0_1 + group_a_4_0_2 + group_a_4_0_3 = 8
_C38: group_b_0_0_0 + group_b_0_0_1 + group_b_0_0_2 + group_b_0_0_3
 + group_b_4_0_0 + group_b_4_0_1 + group_b_4_0_2 + group_b_4_0_3 = 8
_C39: group_c_0_0_0 + group_c_0_0_1 + group_c_0_0_2 + group_c_0_0_3
 + group_c_4_0_0 + group_c_4_0_1 + group_c_4_0_2 + group_c_4_0_3 = 8
_C4: group_a_1_0_3 = 0
_C40: group_a_0_0_1 + group_a_4_0_1 + group_b_0_0_1 + group_b_4_0_1
 + group_c_0_0_1 + group_c_4_0_1 >= 1
_C41: group_a_0_0_2 + group_a_4_0_2 + group_b_0_0_2 + group_b_4_0_2
 + group_c_0_0_2 + group_c_4_0_2 >= 1
_C42: layer_frontier_1_0_0 >= 0
_C43: - 1e-06 group_b_0_0_0 - 1e-06 group_b_4_0_0 + layer_frontier_1_0_1 >= 0
_C44: - 1e-06 group_b_0_0_0 - 1e-06 group_b_0_0_1 - 1e-06 group_b_4_0_0
 - 1e-06 group_b_4_0_1 + layer_frontier_1_0_2 >= 0
_C45: - 1e-06 group_b_0_0_0 - 1e-06 group_b_0_0_1 - 1e-06 group_b_0_0_2
 - 1e-06 group_b_4_0_0 - 1e-06 group_b_4_0_1 - 1e-06 group_b_4_0_2
 + layer_frontier_1_0_3 >= 0
_C46: layer_frontier_2_0_0 >= 0
_C47: - 1e-06 group_c_0_0_0 - 1e-06 group_c_4_0_0 + layer_frontier_2_0_1 >= 0
_C48: - 1e-06 group_c_0_0_0 - 1e-06 group_c_0_0_1 - 1e-06 group_c_4_0_0
 - 1e-06 group_c_4_0_1 + layer_frontier_2_0_2 >= 0
_C49: - 1e-06 group_c_0_0_0 - 1e-06 group_c_0_0_1 - 1e-06 group_c_0_0_2
 - 1e-06 group_c_4_0_0 - 1e-06 group_c_4_0_1 - 1e-06 group_c_4_0_2
 + layer_frontier_2_0_3 >= 0
_C5: group_a_2_0_0 = 0
_C50: group_a_0_0_0 + 1000000 layer_frontier_1_0_0 <= 1000000
_C51: group_a_4_0_0 + 1000000 layer_frontier_1_0_0 <= 1000000
_C52: group_a_0_0_1 + 1000000 layer_frontier_1_0_1 <= 1000000
_C53: group_a_4_0_1 + 1000000 layer_frontier_1_0_1 <= 1000000
_C54: group_a_0_0_2 + 1000000 layer_frontier_1_0_2 <= 1000000
_C55: group_a_4_0_2 + 1000000 layer_frontier_1_0_2 <= 1000000
_C56: group_a_0_0_3 + 1000000 layer_frontier_1_0_3 <= 1000000
_C57: group_a_4_0_3 + 1000000 layer_frontier_1_0_3 <= 1000000
_C58: group_b_0_0_0 + 1000000 layer_frontier_2_0_0 <= 1000000
_C59: group_b_4_0_0 + 1000000 layer_frontier_2_0_0 <= 1000000
_C6: group_a_2_0_1 = 0
_C60: group_b_0_0_1 + 1000000 layer_frontier_2_0_1 <= 1000000
_C61: group_b_4_0_1 + 1000000 layer_frontier_2_0_1 <= 1000000
_C62: group_b_0_0_2 + 1000000 layer_frontier_2_0_2 <= 1000000
_C63: group_b_4_0_2 + 1000000 layer_frontier_2_0_2 <= 1000000
_C64: group_b_0_0_3 + 1000000 layer_frontier_2_0_3 <= 1000000
_C65: group_b_4_0_3 + 1000000 layer_frontier_2_0_3 <= 1000000
_C66: - 1e-06 group_b_4_0_0 - 1e-06 group_c_4_0_0 + recompute_frontier_0_0_0
 >= 0
_C67: - 1e-06 group_c_4_0_0 + recompute_frontier_0_0_1 >= 0
_C68: - 1e-06 group_b_4_0_1 - 1e-06 group_c_4_0_1 + recompute_frontier_0_1_0
 >= 0
_C69: - 1e-06 group_c_4_0_1 + recompute_frontier_0_1_1 >= 0
_C7: group_a_2_0_2 = 0
_C70: - 1e-06 group_b_4_0_2 - 1e-06 group_c_4_0_2 + recompute_frontier_0_2_0
 >= 0
_C71: - 1e-06 group_c_4_0_2 + recompute_frontier_0_2_1 >= 0
_C72: - 1e-06 group_b_4_0_3 - 1e-06 group_c_4_0_3 + recompute_frontier_0_3_0
 >= 0
_C73: - 1e-06 group_c_4_0_3 + recompute_frontier_0_3_1 >= 0
_C74: group_a_0_0_0 + 1000000 recompute_frontier_0_0_0 <= 1000000
_C75: group_a_0_0_1 + 1000000 recompute_frontier_0_1_0 <= 1000000
_C76: group_a_0_0_2 + 1000000 recompute_frontier_0_2_0 <= 1000000
_C77: group_a_0_0_3 + 1000000 recompute_frontier_0_3_0 <= 1000000
_C78: group_b_0_0_0 + 1000000 recompute_frontier_0_0_1 <= 1000000
_C79: group_b_0_0_1 + 1000000 recompute_frontier_0_1_1 <= 1000000
_C8: group_a_2_0_3 = 0
_C80: group_b_0_0_2 + 1000000 recompute_frontier_0_2_1 <= 1000000
_C81: group_b_0_0_3 + 1000000 recompute_frontier_0_3_1 <= 1000000
_C82: - 15 group_a_0_0_0 - 20 group_a_4_0_0 - 12 group_b_0_0_0
 - 16 group_b_4_0_0 - 9 group_c_0_0_0 - 12 group_c_4_0_0 + max_stage_time
 >= 30
_C83: - 15 group_a_0_0_1 - 20 group_a_4_0_1 - 12 group_b_0_0_1
 - 16 group_b_4_0_1 - 9 group_c_0_0_1 - 12 group_c_4_0_1 + max_stage_time >= 0
_C84: - 15 group_a_0_0_2 - 20 group_a_4_0_2 - 12 group_b_0_0_2
 - 16 group_b_4_0_2 - 9 group_c_0_0_2 - 12 group_c_4_0_2 + max_stage_time >= 0
_C85: - 15 group_a_0_0_3 - 20 group_a_4_0_3 - 12 group_b_0_0_3
 - 16 group_b_4_0_3 - 9 group_c_0_0_3 - 12 group_c_4_0_3 + max_stage_time
 >= 60
_C86: - 15 group_a_0_0_0 - 15 group_a_0_0_1 - 15 group_a_0_0_2
 - 15 group_a_0_0_3 - 20 group_a_4_0_0 - 20 group_a_4_0_1 - 20 group_a_4_0_2
 - 20 group_a_4_0_3 - 12 group_b_0_0_0 - 12 group_b_0_0_1 - 12 group_b_0_0_2
 - 12 group_b_0_0_3 - 16 group_b_4_0_0 - 16 group_b_4_0_1 - 16 group_b_4_0_2
 - 16 group_b_4_0_3 - 9 group_c_0_0_0 - 9 group_c_0_0_1 - 9 group_c_0_0_2
 - 9 group_c_0_0_3 - 12 group_c_4_0_0 - 12 group_c_4_0_1 - 12 group_c_4_0_2
 - 12 group_c_4_0_3 + var_sum_FPi_BPi >= 0
_C87: - max_stage_time + pipeline_total_time >= 0
_C88: memory_overhead_0 >= 0
_C89: 900 group_a_0_0_0 + 300 group_a_4_0_0 + 720 group_b_0_0_0
 + 240 group_b_4_0_0 + 540 group_c_0_0_0 + 180 group_c_4_0_0
 + memory_overhead_0 <= 99500
_C9: group_a_3_0_0 = 0
_C90: memory_overhead_1 >= 0
_C91: 700 group_a_0_0_1 + 250 group_a_4_0_1 + 560 group_b_0_0_1
 + 200 group_b_4_0_1 + 420 group_c_0_0_1 + 150 group_c_4_0_1
 + memory_overhead_1 <= 100000
_C92: memory_overhead_2 >= 0
_C93: 500 group_a_0_0_2 + 200 group_a_4_0_2 + 400 group_b_0_0_2
 + 160 group_b_4_0_2 + 300 group_c_0_0_2 + 120 group_c_4_0_2
 + memory_overhead_2 <= 100000
_C94: memory_overhead_3 >= 0
_C95: 300 group_a_0_0_3 + 150 group_a_4_0_3 + 240 group_b_0_0_3
 + 120 group_b_4_0_3 + 180 group_c_0_0_3 + 90 group_c_4_0_3
 + memory_overhead_3 <= 99500
Bounds
 0 <= group_a_0_0_0
 0 <= group_a_0_0_1
 0 <= group_a_0_0_2
 0 <= group_a_0_0_3
 0 <= group_a_1_0_0
 0 <= group_a_1_0_1
 0 <= group_a_1_0_2
 0 <= group_a_1_0_3
 0 <= group_a_2_0_0
 0 <= group_a_2_0_1
 0 <= group_a_2_0_2
 0 <= group_a_2_0_3
 0 <= group_a_3_0_0
 0 <= group_a_3_0_1
 0 <= group_a_3_0_2
 0 <= group_a_3_0_3
 0 <= group_a_4_0_0
 0 <= group_a_4_0_1
 0 <= group_a_4_0_2
 0 <= group_a_4_0_3
 0 <= group_b_0_0_0
 0 <= group_b_0_0_1
 0 <= group_b_0_0_2
 0 <= group_b_0_0_3
 0 <= group_b_1_0_0
 0 <= group_b_1_0_1
 0 <= group_b_1_0_2
 0 <= group_b_1_0_3
 0 <= group_b_2_0_0
 0 <= group_b_2_0_1
 0 <= group_b_2_0_2
 0 <= group_b_2_0_3
 0 <= group_b_3_0_0
 0 <= group_b_3_0_1
 0 <= group_b_3_0_2
 0 <= group_b_3_0_3
 0 <= group_b_4_0_0
 0 <= group_b_4_0_1
 0 <= group_b_4_0_2
 0 <= group_b_4_0_3
 0 <= group_c_0_0_0
 0 <= group_c_0_0_1
 0 <= group_c_0_0_2
 0 <= group_c_0_0_3
 0 <= group_c_1_0_0
 0 <= group_c_1_0_1
 0 <= group_c_1_0_2
 0 <= group_c_1_0_3
 0 <= group_c_2_0_0
 0 <= group_c_2_0_1
 0 <= group_c_2_0_2
 0 <= group_c_2_0_3
 0 <= group_c_3_0_0
 0 <= group_c_3_0_1
 0 <= group_c_3_0_2
 0 <= group_c_3_0_3
 0 <= group_c_4_0_0
 0 <= group_c_4_0_1
 0 <= group_c_4_0_2
 0 <= group_c_4_0_3
 0 <= memory_overhead_0
 0 <= memory_overhead_1
 0 <= memory_overhead_2
 0 <= memory_overhead_3
Generals
group_a_0_0_0
group_a_0_0_1
group_a_0_0_2
group_a_0_0_3
group_a_1_0_0
group_a_1_0_1
group_a_1_0_2
group_a_1_0_3
group_a_2_0_0
group_a_2_0_1
group_a_2_0_2
group_a_2_0_3
group_a_3_0_0
group_a_3_0_1
group_a_3_0_2
group_a_3_0_3
group_a_4_0_0
group_a_4_0_1
group_a_4_0_2
group_a_4_0_3
group_b_0_0_0
group_b_0_0_1
group_b_0_0_2
group_b_0_0_3
group_b_1_0_0
group_b_1_0_1
group_b_1_0_2
group_b_1_0_3
group_b_2_0_0
group_b_2_0_1
group_b_2_0_2
group_b_2_0_3
group_b_3_0_0
group_b_3_0_1
group_b_3_0_2
group_b_3_0_3
group_b_4_0_0
group_b_4_0_1
group_b_4_0_2
group_b_4_0_3
group_c_0_0_0
group_c_0_0_1
group_c_0_0_2
group_c_0_0_3
group_c_1_0_0
group_c_1_0_1
group_c_1_0_2
group_c_1_0_3
group_c_2_0_0
group_c_2_0_1
group_c_2_0_2
group_c_2_0_3
group_c_3_0_0
group_c_3_0_1
group_c_3_0_2
group_c_3_0_3
group_c_4_0_0
group_c_4_0_1
group_c_4_0_2
group_c_4_0_3
memory_overhead_0
memory_overhead_1
memory_overhead_2
memory_overhead_3
Binaries
layer_frontier_1_0_0
layer_frontier_1_0_1
layer_frontier_1_0_2
layer_frontier_1_0_3
layer_frontier_2_0_0
layer_frontier_2_0_1
layer_frontier_2_0_2
layer_frontier_2_0_3
recompute_frontier_0_0_0
recompute_frontier_0_0_1
recompute_frontier_0_1_0
recompute_frontier_0_1_1
recompute_frontier_0_2_0
recompute_frontier_0_2_1
recompute_frontier_0_3_0
recompute_frontier_0_3_1
End
