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
    stable_symptomatic = Column('stable_symptomatic', 'Stable Symptomatic', 'yellow')
    stable_asymptomatic = Column('stable_asymptomatic', 'Stable Asymptomatic', 'cyan')
    recovered = Column('recovered', 'Recovered Cases', 'green')
    deceased = Column('deceased', 'Deceased Cases', 'red')
    active = Column('hospitalised', 'Active Cases', 'orange')
    confirmed = Column('total_infected', 'Confirmed Cases', 'C0')
    
enums = [
    Column('date', 'date', None),
    Column('critical', 'Critical', 'brown'),
    Column('stable_symptomatic', 'Stable Symptomatic', 'yellow'),
    Column('stable_asymptomatic', 'Stable Asymptomatic', 'cyan'),
    Column('recovered', 'Recovered Cases', 'green'),
    Column('deceased', 'Deceased Cases', 'red'),
    Column('hospitalised', 'Active Cases', 'orange'),
    Column('total_infected', 'Confirmed Cases', 'C0')
]