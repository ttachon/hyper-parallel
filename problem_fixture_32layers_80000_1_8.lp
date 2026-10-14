\* Pipeline_execution_time_minimize *\
Minimize
OBJ: pipeline_total_time
Subject To
_C1: BODY_2_0_0 = 0
_C10: BODY_3_0_1 = 0
_C11: BODY_3_0_2 = 0
_C12: BODY_3_0_3 = 0
_C13: BODY_3_0_4 = 0
_C14: BODY_3_0_5 = 0
_C15: BODY_3_0_6 = 0
_C16: BODY_3_0_7 = 0
_C17: BODY_4_0_0 = 0
_C18: BODY_4_0_1 = 0
_C19: BODY_4_0_2 = 0
_C2: BODY_2_0_1 = 0
_C20: BODY_4_0_3 = 0
_C21: BODY_4_0_4 = 0
_C22: BODY_4_0_5 = 0
_C23: BODY_4_0_6 = 0
_C24: BODY_4_0_7 = 0
_C25: BODY_0_0_0 + BODY_0_0_1 + BODY_0_0_2 + BODY_0_0_3 + BODY_0_0_4
 + BODY_0_0_5 + BODY_0_0_6 + BODY_0_0_7 + BODY_1_0_0 + BODY_1_0_1 + BODY_1_0_2
 + BODY_1_0_3 + BODY_1_0_4 + BODY_1_0_5 + BODY_1_0_6 + BODY_1_0_7 = 32
_C26: BODY_0_0_1 + BODY_1_0_1 >= 1
_C27: BODY_0_0_2 + BODY_1_0_2 >= 1
_C28: BODY_0_0_3 + BODY_1_0_3 >= 1
_C29: BODY_0_0_4 + BODY_1_0_4 >= 1
_C3: BODY_2_0_2 = 0
_C30: BODY_0_0_5 + BODY_1_0_5 >= 1
_C31: BODY_0_0_6 + BODY_1_0_6 >= 1
_C32: - 9 BODY_0_0_0 - 9.24 BODY_1_0_0 + max_stage_time >= 30
_C33: - 9 BODY_0_0_1 - 9.24 BODY_1_0_1 + max_stage_time >= 0
_C34: - 9 BODY_0_0_2 - 9.24 BODY_1_0_2 + max_stage_time >= 0
_C35: - 9 BODY_0_0_3 - 9.24 BODY_1_0_3 + max_stage_time >= 0
_C36: - 9 BODY_0_0_4 - 9.24 BODY_1_0_4 + max_stage_time >= 0
_C37: - 9 BODY_0_0_5 - 9.24 BODY_1_0_5 + max_stage_time >= 0
_C38: - 9 BODY_0_0_6 - 9.24 BODY_1_0_6 + max_stage_time >= 0
_C39: - 9 BODY_0_0_7 - 9.24 BODY_1_0_7 + max_stage_time >= 60
_C4: BODY_2_0_3 = 0
_C40: - 9 BODY_0_0_0 - 9 BODY_0_0_1 - 9 BODY_0_0_2 - 9 BODY_0_0_3
 - 9 BODY_0_0_4 - 9 BODY_0_0_5 - 9 BODY_0_0_6 - 9 BODY_0_0_7 - 9.24 BODY_1_0_0
 - 9.24 BODY_1_0_1 - 9.24 BODY_1_0_2 - 9.24 BODY_1_0_3 - 9.24 BODY_1_0_4
 - 9.24 BODY_1_0_5 - 9.24 BODY_1_0_6 - 9.24 BODY_1_0_7 + var_sum_FPi_BPi >= 0
_C41: - max_stage_time + pipeline_total_time >= 0
_C42: 1000 BODY_0_0_0 + 680 BODY_1_0_0 <= 79000
_C43: 900 BODY_0_0_1 + 620 BODY_1_0_1 <= 79500
_C44: 800 BODY_0_0_2 + 560 BODY_1_0_2 <= 79500
_C45: 700 BODY_0_0_3 + 500 BODY_1_0_3 <= 79500
_C46: 600 BODY_0_0_4 + 440 BODY_1_0_4 <= 79500
_C47: 500 BODY_0_0_5 + 380 BODY_1_0_5 <= 79500
_C48: 400 BODY_0_0_6 + 320 BODY_1_0_6 <= 79500
_C49: 300 BODY_0_0_7 + 260 BODY_1_0_7 <= 79000
_C5: BODY_2_0_4 = 0
_C6: BODY_2_0_5 = 0
_C7: BODY_2_0_6 = 0
_C8: BODY_2_0_7 = 0
_C9: BODY_3_0_0 = 0
Bounds
 0 <= BODY_0_0_0
 0 <= BODY_0_0_1
 0 <= BODY_0_0_2
 0 <= BODY_0_0_3
 0 <= BODY_0_0_4
 0 <= BODY_0_0_5
 0 <= BODY_0_0_6
 0 <= BODY_0_0_7
 0 <= BODY_1_0_0
 0 <= BODY_1_0_1
 0 <= BODY_1_0_2
 0 <= BODY_1_0_3
 0 <= BODY_1_0_4
 0 <= BODY_1_0_5
 0 <= BODY_1_0_6
 0 <= BODY_1_0_7
 0 <= BODY_2_0_0
 0 <= BODY_2_0_1
 0 <= BODY_2_0_2
 0 <= BODY_2_0_3
 0 <= BODY_2_0_4
 0 <= BODY_2_0_5
 0 <= BODY_2_0_6
 0 <= BODY_2_0_7
 0 <= BODY_3_0_0
 0 <= BODY_3_0_1
 0 <= BODY_3_0_2
 0 <= BODY_3_0_3
 0 <= BODY_3_0_4
 0 <= BODY_3_0_5
 0 <= BODY_3_0_6
 0 <= BODY_3_0_7
 0 <= BODY_4_0_0
 0 <= BODY_4_0_1
 0 <= BODY_4_0_2
 0 <= BODY_4_0_3
 0 <= BODY_4_0_4
 0 <= BODY_4_0_5
 0 <= BODY_4_0_6
 0 <= BODY_4_0_7
Generals
BODY_0_0_0
BODY_0_0_1
BODY_0_0_2
BODY_0_0_3
BODY_0_0_4
BODY_0_0_5
BODY_0_0_6
BODY_0_0_7
BODY_1_0_0
BODY_1_0_1
BODY_1_0_2
BODY_1_0_3
BODY_1_0_4
BODY_1_0_5
BODY_1_0_6
BODY_1_0_7
BODY_2_0_0
BODY_2_0_1
BODY_2_0_2
BODY_2_0_3
BODY_2_0_4
BODY_2_0_5
BODY_2_0_6
BODY_2_0_7
BODY_3_0_0
BODY_3_0_1
BODY_3_0_2
BODY_3_0_3
BODY_3_0_4
BODY_3_0_5
BODY_3_0_6
BODY_3_0_7
BODY_4_0_0
BODY_4_0_1
BODY_4_0_2
BODY_4_0_3
BODY_4_0_4
BODY_4_0_5
BODY_4_0_6
BODY_4_0_7
End
