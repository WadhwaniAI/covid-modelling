from collections import OrderedDict

# define SEIR model parameters (Gabriel Goh's version)
def init_params(R0 = 2.2, T_inf = 2.9, T_inc = 5.2, T_hosp = 5, T_death = 32, P_severe = 0.2, P_fatal = 0.02, N = 7e6, 
                init_infected = 1, intervention_day = 100, intervention_amount = 3, testing_rate_for_exposed = 0,
                positive_test_rate_for_exposed = 1, testing_rate_for_infected = 0, positive_test_rate_for_infected = 1, q = 0):
    
    T_trans = T_inf/R0
    T_recov_mild = (14 - T_inf)
    T_recov_severe = (31.5 - T_inf)
    
    P_mild = 1 - P_severe - P_fatal

    # define testing related parameters
    T_inf_detected = T_inf
    T_trans_detected = T_trans
    T_inc_detected = T_inc

    P_mild_detected = P_mild
    P_severe_detected = P_severe
    P_fatal_detected = P_fatal

    vanilla_params = {
        'T_trans': T_trans,
        'T_inc': T_inc,
        'T_inf': T_inf,

        'T_recov_mild': T_recov_mild,
        'T_recov_severe': T_recov_severe,
        'T_hosp': T_hosp,
        'T_death': T_death,

        'P_mild': P_mild,
        'P_severe': P_severe,
        'P_fatal': P_fatal,
        'intervention_day': intervention_day,
        'intervention_amount': intervention_amount,
        'N' : N
    }

    testing_params = {
        'T_trans': T_trans_detected,
        'T_inc': T_inc_detected,
        'T_inf': T_inf_detected,

        'P_mild': P_mild_detected,
        'P_severe': P_severe_detected,
        'P_fatal': P_fatal_detected,

        'q': q,
        'testing_rate_for_exposed': testing_rate_for_exposed,
        'positive_test_rate_for_exposed': positive_test_rate_for_exposed,
        'testing_rate_for_infected': testing_rate_for_infected,
        'positive_test_rate_for_infected': positive_test_rate_for_infected
    }

    # S, E, D_E, D_I, I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D
    state_init_values = OrderedDict()
    state_init_values['S'] = (N - init_infected)/N
    state_init_values['E'] = 0
    state_init_values['I'] = init_infected/N
    state_init_values['D_E'] = 0
    state_init_values['D_I'] = 0
    state_init_values['R_mild'] = 0
    state_init_values['R_severe_home'] = 0
    state_init_values['R_severe_hosp'] = 0
    state_init_values['R_fatal'] = 0
    state_init_values['C'] = 0
    state_init_values['D'] = 0
    
    return vanilla_params, testing_params, state_init_values