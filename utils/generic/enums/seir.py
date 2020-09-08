import enum

from enum import Enum
from collections import namedtuple

ParamLabel = namedtuple('Color', ['name'])

class SEIRParams(Enum):

    @property
    def name(self):
        return self.value.name

    e_hosp = ParamLabel('E_hosp_ratio')
    i_hosp = ParamLabel('I_hosp_ratio')
    p_fatal = ParamLabel('P_fatal')
    t_inc = ParamLabel('T_inc')
    t_inf = ParamLabel('T_inf')
    t_recov_severe = ParamLabel('T_recov_severe')
    r0 = ParamLabel('lockdown_R0')