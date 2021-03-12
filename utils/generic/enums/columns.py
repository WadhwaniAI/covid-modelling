from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class Colors(Enum):
    bed_colors = [plt.get_cmap('plasma')(i) for i in np.linspace(0, 0.8, 5)]
    severity_colors = [plt.get_cmap('RdPu')(i) for i in np.linspace(0.4, 0.9, 3)]
    
class Column:
    def __init__(self, name, label, color):
        self.name = name
        self.label = label
        self.color = color

    @property
    def name(self):
        return self._name
        
    @property
    def label(self):
        return self._label
    
    @property
    def color(self):
        return self._color

    @name.setter
    def name(self, name):
        self._name = name

    @label.setter
    def label(self, label):
        self._label = label

    @color.setter
    def color(self, color):
        self._color = color


class Columns(Enum):
    date = Column('date', 'date', None)
    critical = Column('critical', 'Critical', Colors.severity_colors[0])
    symptomatic = Column('symptomatic', 'Stable Symptomatic', Colors.severity_colors[1])
    asymptomatic = Column('asymptomatic', 'Stable Asymptomatic', Colors.severity_colors[2])
    recovered = Column('recovered', 'Recovered Cases', 'green')
    deceased = Column('deceased', 'Deceased Cases', 'red')
    active = Column('active', 'Active Cases', 'orange')
    total = Column('total', 'Confirmed Cases', 'C0')
    ventilator = Column('ventilator', 'Ventilator Beds', Colors.bed_colors[0])
    icu = Column('icu', 'ICU Beds', Colors.bed_colors[1])
    o2_beds = Column('o2_beds', 'O2 Beds', Colors.bed_colors[2])
    non_o2_beds = Column('non_o2_beds', 'Non O2 Beds', Colors.bed_colors[3])
    hq = Column('hq', 'Home Quarantine', Colors.bed_colors[4])

    @classmethod
    def CARD_compartments(cls):
        return [cls.recovered, cls.deceased, cls.active, cls.total]
    
    @classmethod
    def severity_compartments(cls):
        return [cls.asymptomatic, cls.symptomatic, cls.critical]

    @classmethod
    def bed_compartments(cls):
        return [cls.ventilator, cls.icu, cls.o2_beds, cls.non_o2_beds, cls.hq]

    @classmethod
    def curve_fit_compartments(cls):
        # TODO: we only want to fit the curve fits on the first three and calc total as sum
        # the scripts don't accommodate this yet because total is left out of df_true
        return [cls.recovered, cls.deceased, cls.active, cls.total]

    @staticmethod
    def from_name(name):
        try:
            return getattr(Columns, name)
        except Exception as e:
            print(e)


compartments = {
    'date': Column('date', 'date', None),
    'base': [
        Column('total', 'Confirmed Cases', 'C0'),
        Column('active', 'Active Cases', 'orange'),
        Column('recovered', 'Recovered Cases', 'green'),
        Column('deceased', 'Deceased Cases', 'red')
    ],
    'base_diff': [
        Column('daily_cases', 'New Cases Added', 'indigo')
    ],
    'severity': [
        Column('critical', 'Critical', Colors.severity_colors[0]),
        Column('symptomatic', 'Stable Symptomatic', Colors.severity_colors[1]),
        Column('asymptomatic', 'Stable Asymptomatic', Colors.severity_colors[2])
    ],
    'bed': [
        Column('ventilator', 'Ventilator Beds', Colors.bed_colors[0]),
        Column('icu', 'ICU Beds', Colors.bed_colors[1]),
        Column('o2_beds', 'O2 Beds', Colors.bed_colors[2]),
        Column('non_o2_beds', 'Non O2 Beds', Colors.bed_colors[3]),
        Column('hq', 'Home Quarantine', Colors.bed_colors[4])
    ]
}
