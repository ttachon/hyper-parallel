\* Pipeline_execution_time_minimize *\
Minimize
OBJ: pipeline_total_time
Subject To
_C1: BODY_2_0_0 = 0
_C10: BODY_4_0_1 = 0
_C11: BODY_4_0_2 = 0
_C12: BODY_4_0_3 = 0
_C13: BODY_0_0_0 + BODY_0_0_1 + BODY_0_0_2 + BODY_0_0_3 + BODY_1_0_0
 + BODY_1_0_1 + BODY_1_0_2 + BODY_1_0_3 = 32
_C14: BODY_0_0_1 + BODY_1_0_1 >= 1
_C15: BODY_0_0_2 + BODY_1_0_2 >= 1
_C16: - 9 BODY_0_0_0 - 9.24 BODY_1_0_0 + max_stage_time >= 30
_C17: - 9 BODY_0_0_1 - 9.24 BODY_1_0_1 + max_stage_time >= 0
_C18: - 9 BODY_0_0_2 - 9.24 BODY_1_0_2 + max_stage_time >= 0
_C19: - 9 BODY_0_0_3 - 9.24 BODY_1_0_3 + max_stage_time >= 60
_C2: BODY_2_0_1 = 0
_C20: - 9 BODY_0_0_0 - 9 BODY_0_0_1 - 9 BODY_0_0_2 - 9 BODY_0_0_3
 - 9.24 BODY_1_0_0 - 9.24 BODY_1_0_1 - 9.24 BODY_1_0_2 - 9.24 BODY_1_0_3
 + var_sum_FPi_BPi >= 0
_C21: - max_stage_time + pipeline_total_time >= 0
_C22: memory_overhead_0 >= 0
_C23: 600 BODY_0_0_0 + 440 BODY_1_0_0 + memory_overhead_0 <= 79500
_C24: memory_overhead_1 >= 0
_C25: 500 BODY_0_0_1 + 380 BODY_1_0_1 + memory_overhead_1 <= 80000
_C26: memory_overhead_2 >= 0
_C27: 400 BODY_0_0_2 + 320 BODY_1_0_2 + memory_overhead_2 <= 80000
_C28: memory_overhead_3 >= 0
_C29: 300 BODY_0_0_3 + 260 BODY_1_0_3 + memory_overhead_3 <= 79500
_C3: BODY_2_0_2 = 0
_C4: BODY_2_0_3 = 0
_C5: BODY_3_0_0 = 0
_C6: BODY_3_0_1 = 0
_C7: BODY_3_0_2 = 0
_C8: BODY_3_0_3 = 0
_C9: BODY_4_0_0 = 0
Bounds
 0 <= BODY_0_0_0
 0 <= BODY_0_0_1
 0 <= BODY_0_0_2
 0 <= BODY_0_0_3
 0 <= BODY_1_0_0
 0 <= BODY_1_0_1
 0 <= BODY_1_0_2
 0 <= BODY_1_0_3
 0 <= BODY_2_0_0
 0 <= BODY_2_0_1
 0 <= BODY_2_0_2
 0 <= BODY_2_0_3
 0 <= BODY_3_0_0
 0 <= BODY_3_0_1
 0 <= BODY_3_0_2
 0 <= BODY_3_0_3
 0 <= BODY_4_0_0
 0 <= BODY_4_0_1
 0 <= BODY_4_0_2
 0 <= BODY_4_0_3
 0 <= memory_overhead_0
 0 <= memory_overhead_1
 0 <= memory_overhead_2
 0 <= memory_overhead_3
Generals
BODY_0_0_0
BODY_0_0_1
BODY_0_0_2
BODY_0_0_3
BODY_1_0_0
BODY_1_0_1
BODY_1_0_2
BODY_1_0_3
BODY_2_0_0
BODY_2_0_1
BODY_2_0_2
BODY_2_0_3
BODY_3_0_0
BODY_3_0_1
BODY_3_0_2
BODY_3_0_3
BODY_4_0_0
BODY_4_0_1
BODY_4_0_2
BODY_4_0_3
memory_overhead_0
memory_overhead_1
memory_overhead_2
memory_overhead_3
End
