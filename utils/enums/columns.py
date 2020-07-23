import enum

from enum import Enum
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

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
    daily_cases = Column('daily_cases', 'New Cases Added', 'purple')

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
    
cmap = plt.get_cmap('plasma')
bed_colors = [cmap(i) for i in np.linspace(0, 0.8, 5)]

cmap = plt.get_cmap('RdPu')
severity_colors = [cmap(i) for i in np.linspace(0.4, 0.9, 3)]
compartments = {
    'date': [Column('date', 'date', None)],
    'base': [
        Column('recovered', 'Recovered Cases', 'green'),
        Column('deceased', 'Deceased Cases', 'red'),
        Column('hospitalised', 'Active Cases', 'orange'),
        Column('total_infected', 'Confirmed Cases', 'C0')
    ],
    'severity': [
        Column('critical', 'Critical', severity_colors[0]),
        Column('stable_symptomatic', 'Stable Symptomatic', severity_colors[1]),
        Column('stable_asymptomatic', 'Stable Asymptomatic', severity_colors[2])
    ],
    'bed': [
        Column('ventilator', 'Ventilator Beds', bed_colors[0]),
        Column('icu', 'ICU Beds', bed_colors[1]),
        Column('o2_beds', 'O2 Beds', bed_colors[2]),
        Column('non_o2_beds', 'Non O2 Beds', bed_colors[3]),
        Column('hq', 'Home Quarantine', bed_colors[4])
    ]
}
