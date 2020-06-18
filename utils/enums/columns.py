import enum

from enum import Enum
from collections import namedtuple

Column = namedtuple('Column', ['name', 'label', 'color'])

class Columns(Enum):

    @property
    def name(self):
        return self.value.name
        
    @property
    def label(self):
        return self.value.label
    
    @property
    def color(self):
        return self.value.color

    date = Column('date', 'date', None)
    critical = Column('critical', 'Critical', 'brown')
    stable_symptomatic = Column('stable_symptomatic', 'Stable Symptomatic', 'magenta')
    stable_asymptomatic = Column('stable_asymptomatic', 'Stable Asymptomatic', 'cyan')
    recovered = Column('recovered', 'Recovered Cases', 'green')
    deceased = Column('deceased', 'Deceased Cases', 'red')
    active = Column('hospitalised', 'Active Cases', 'orange')
    confirmed = Column('total_infected', 'Confirmed Cases', 'C0')

    @classmethod
    def which_compartments(cls):
        return [cls.recovered, cls.deceased, cls.active, cls.confirmed]

    @classmethod
    def total_compartments(cls):
        return [cls.recovered, cls.deceased, cls.active]
    
    @classmethod
    def active_compartments(cls):
        return [cls.stable_asymptomatic, cls.stable_symptomatic, cls.critical]

    @classmethod
    def curve_fit_compartments(cls):
        # TODO: we only want to fit the curve fits on the first three and calc confirmed as sum
            # the scripts don't accomodate this yet because confirmed is left out of df_true
        return [cls.recovered, cls.deceased, cls.active, cls.confirmed]


compartments = [
    Column('date', 'date', None),
    Column('critical', 'Critical', 'brown'),
    Column('stable_symptomatic', 'Stable Symptomatic', 'magenta'),
    Column('stable_asymptomatic', 'Stable Asymptomatic', 'cyan'),
    Column('recovered', 'Recovered Cases', 'green'),
    Column('deceased', 'Deceased Cases', 'red'),
    Column('hospitalised', 'Active Cases', 'orange'),
    Column('total_infected', 'Confirmed Cases', 'C0')
]
