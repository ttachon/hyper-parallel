\* Pipeline_execution_time_minimize *\
Minimize
OBJ: pipeline_total_time
Subject To
_C1: BODY_2_0_0 = 0
_C10: - 9 BODY_0_0_0 - 9 BODY_0_0_1 - 9.24 BODY_1_0_0 - 9.24 BODY_1_0_1
 + var_sum_FPi_BPi >= 0
_C11: - max_stage_time + pipeline_total_time >= 0
_C12: memory_overhead_0 >= 0
_C13: 400 BODY_0_0_0 + 320 BODY_1_0_0 + memory_overhead_0 <= 79000
_C14: memory_overhead_1 >= 0
_C15: 300 BODY_0_0_1 + 260 BODY_1_0_1 + memory_overhead_1 <= 79000
_C2: BODY_2_0_1 = 0
_C3: BODY_3_0_0 = 0
_C4: BODY_3_0_1 = 0
_C5: BODY_4_0_0 = 0
_C6: BODY_4_0_1 = 0
_C7: BODY_0_0_0 + BODY_0_0_1 + BODY_1_0_0 + BODY_1_0_1 = 32
_C8: - 9 BODY_0_0_0 - 9.24 BODY_1_0_0 + max_stage_time >= 30
_C9: - 9 BODY_0_0_1 - 9.24 BODY_1_0_1 + max_stage_time >= 60
Bounds
 0 <= BODY_0_0_0
 0 <= BODY_0_0_1
 0 <= BODY_1_0_0
 0 <= BODY_1_0_1
 0 <= BODY_2_0_0
 0 <= BODY_2_0_1
 0 <= BODY_3_0_0
 0 <= BODY_3_0_1
 0 <= BODY_4_0_0
 0 <= BODY_4_0_1
 0 <= memory_overhead_0
 0 <= memory_overhead_1
Generals
BODY_0_0_0
BODY_0_0_1
BODY_1_0_0
BODY_1_0_1
BODY_2_0_0
BODY_2_0_1
BODY_3_0_0
BODY_3_0_1
BODY_4_0_0
BODY_4_0_1
memory_overhead_0
memory_overhead_1
End
