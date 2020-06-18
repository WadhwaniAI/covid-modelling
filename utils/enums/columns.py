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

    @staticmethod
    def from_name(name):
        if name == 'date':
            return Columns.date
        elif name == 'recovered':
            return Columns.recovered
        elif name == 'deceased':
            return Columns.deceased
        elif name == 'hospitalised':
            return Columns.active
        elif name == 'total_infected':
            return Columns.confirmed
        else:
            raise Exception(f"Enum for name {name} not found")
    
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
