######################################
# CS1110 - Intro to Programming      #
# Module 8 - Prog Exercise 11 - Ex 6 #
# Author: Noah Wood                  #
# Date:   05/07/2022                 #
######################################
from sklearn import linear_model
import math
import csv
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class Point:
    __slots__ = ["opened", "high", "low", "close", "adj_close", "volume",
                 'average', 'max', 'day', 'month', 'year', 'date', 'diff']
    def __init__(self, date, opened, high, low, close, adj_close, volume):
        self.opened = float(opened)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.adj_close = float(adj_close)
        self.volume = float(volume)
        self.average = self._average()
        self.max = self._max()
        self.diff = self.high-self.low
        self._set_dates(date)
        
    def _set_dates(self, date):
        y,m,d = date.split('-')
        self.day = int(d)
        self.month = int(m)
        self.year = int(y)
        self.date = date
        
    def _average(self):
        return (self.opened + self.high + self.low + self.close + self.adj_close)/5

    def _max(self):
        return max(self.opened, self.high, self.low, self.close, self.adj_close, self.average)

STEP = {
    'd':1,
    'w':7,
    'm':30,
    'q':91,
    'y':365
    }

LABEL = {
    'd':'Day',
    'w':'Week',
    'm':'Month',
    'q':'Quarter',
    'y': 'Year'
    }

MONTHS = {
    1:"Jan",
    2:"Feb",
    3:"March",
    4:"April",
    5:"May",
    6:"June",
    7:"July",
    8:"Aug",
    9:"Sep",
    10:"Oct",
    11:"Nov",
    12:"Dec"
    }

def Slope(x1,x2,y1,y2):
    """Returns the slope as a floating point number.

        slope = rise/run
    """
    return (y2-y1)/(x2-x1)
    

class Stockalizer:
    def __init__(self, step='m'):
        self.lpoints = []
        self.set_step(step)
        self.plt = plt
        
    def load_yahoo_csv(self, file):
        """Loads data from yahoo's stock CSV"""
        data = csv.reader(open(file))
        next(data) # Skip first/header line
        for point in data:
            # Return if we hit end of file
            if point[2] == 'null':
                self._load_end()
                return
            # Create a new point
            x = Point(*point)
           # Add to list (reduces complexity of whole database computation)
            self.lpoints.append(x)        
        self._load_end()

    def highest_day(self):
        """Returns the point on the highest recorded day."""
        p = 0
        for i in range(len(self.lpoints)):
            if self.lpoints[i].close > self.lpoints[p].close:
                p = i
        return self.lpoints[p]

    def lowest_day(self):
        """Returns the point on the lowest recorded day."""
        p = -1
        for i in range(len(self.lpoints)):
            if self.lpoints[i].close < self.lpoints[p].close:
                p = i
        return self.lpoints[p]

    def average_close(self):
        """Returns the average closing price."""
        return sum([x.close for x in self.lpoints])/len(self.lpoints)

    def overall_slope(self):
        """Calculates the slope between first and last points."""
        return Slope(len(self.lpoints), 0,
                     self.lpoints[-1].close, self.lpoints[0].close)

    def biggest_difference(self):
        """Calculates the largest difference between max and min in any day."""
        return max([x.diff for x in self.lpoints])

    def smallest_difference(self):
        """Calculates the smallest difference between max and min in any day."""
        return min([x.diff for x in self.lpoints])

    def largest_increase(self):
        """Calculates the largest increase over a period of time."""
        increase = 0
        point1 = None
        point2 = None
        for i in range(0,len(self.lpoints)-self.step,self.step):
            b1 = self.lpoints[i]
            b2 = self.lpoints[i+self.step]
            a = b2.close - b1.close
            if a > increase:
                point1 = b1
                point2 = b2
                increase = a
        self.largest = [point1, point2, a]
        
    def smallest_increase(self):
        """Calculates smalles increase over a period of time."""
        increase = 0
        point1 = None
        point2 = None
        for i in range(0, len(self.lpoints)-self.step, self.step):
            b1 = self.lpoints[i]
            b2 = self.lpoints[i+self.step]
            a = b2.close - b1.close
            if a < increase:
                point1 = b1
                point2 = b2
                increase = a
        self.smallest = [point1, point2, a]
                
    def _load_end(self):
        """Sets up some values after loading data"""
        # Set X and Y values for plotting
        self.X = np.arange(0, len(self.lpoints))
        self.Y = [x.close for x in self.lpoints]
        # Set the number of ticks for x graph
        self.x_ticks = np.arange(self.step, self.step +
                                 len(self.X)+self.step, step=self.step)
        
    def set_step(self, step):
        """Sets the self.step attribute based on step.

            step = char (or int) for which step to use
        """
        if type(step) == type(0):
            self.step = int(step)
            self.label = "Period: {} Days.".format(step)
        else:
            self.step = STEP[step]
            self.label = LABEL[step]

    def _get_peaks(self, func):
        n = int(math.ceil(len(self.Y)/self.step))
        s = 0
        x = []
        y = []
        for i in range(0,n):
            j = self.Y.index(func(self.Y[s:s+self.step]))
            x.append(j + self.step)
            y.append(self.Y[j])
            s = s + self.step
        return [x,y]

    def get_peaks(self):
        """Sets the min and max peaks."""
        self.min_peaks = self._get_peaks(min)
        self.max_peaks = self._get_peaks(max)

    def get_amp_offset(self):
        a = []
        o = []
        for i in range(len(self.min_peaks[1])):
            amps = (self.max_peaks[1][i] - self.min_peaks[1][i])/2
            offset = self.min_peaks[1][i] + amps
            for j in range(0, self.step):
                a.append(amps)
                o.append(offset)
        self.amplitudes = a
        self.offsets = o

    def get_omega(self):
        """Calculates the num of cycles between 0 and 2pi."""
        self.omega = 2.*np.pi / self.step
        
    def var_sinusoid(self, x, y, phase=0):
        """Calculates the sine of a group of points.
    
            x = x point on graph
            y = data
            phase (default 0)
            uses:
                self.amplitudes
                self.offsets
                self.omega
            Assume phase is 0 for variable, set phase to phase for known phases.            
        """
        out = []
        # Base values have more data in predicted amplitudes than base
        for i in range(len(x)):
            out.append(self.amplitudes[i]*math.sin(self.omega*x[i]-phase)+self.offsets[i])
        return out

    def plot(self, x, y, color='red', width=1, style='solid'):
        self.plt.plot(x, y, color=color, linewidth=width, linestyle=style)

    def show_plot(self):
        self.plt.show()

    def predict(self):
        """Generates a min/max prediction based on a linear regression of sine plot."""
        # reshape the peaks because linear_model sucks
        xmin = [[x] for x in self.min_peaks[1]]
        xmax = [[x] for x in self.max_peaks[1]]
        # min/max models
        mmin = linear_model.LinearRegression()
        mmin.fit(xmin, self.min_peaks[0])
        mmax = linear_model.LinearRegression()
        mmax.fit(xmax, self.max_peaks[0])
        # predict lines
        pmin = mmin.predict(xmin)
        pmax = mmax.predict(xmax)
        # Draw prediction max/min
        self.plot(pmin,xmin,'red',style='dashed')
        self.plot(pmax,xmax,'green', style='dashed')

    def predict_sine(self):
        """Generates a stock prediction based on sinusoidal regression."""
        x = np.array(self.X)
        t = x[len(x)-1]
        for i in range(0, self.step):
            x = np.append(x,t)
            t = t + 1
        i = len(self.max_peaks[0])-1
        amp = (self.max_peaks[1][i] - self.min_peaks[1][i])/2
        off = self.min_peaks[1][i] + amp
        for j in range(0,self.step):
            self.amplitudes.append(amp)
            self.offsets.append(off)
        self.plot(x, self.var_sinusoid(x, self.step), color="blue", width=1, style='dashed')

    def setup_chart(self):
        x_labels=[]
        y_labels=[]
        min_y = int(self.lowest_day().close)
        max_y = int(self.highest_day().close)
        self.y_ticks = []
        for i in range(min_y, max_y+5000, 5000):
            y_labels.append("${},000".format(int(i/1000)))
            self.y_ticks.append(i)
        y = int(self.lpoints[0].year)
        if self.label == 'Year':
            # skip first year to match up data
            y = int(self.lpoints[365].year)
            x_labels.append(y)
            for i in range(1, len(self.x_ticks)):
                x_labels.append((x_labels[i-1]+1))
        elif self.label == 'Month':
            m = int(self.lpoints[0].month)
            x_labels.append(MONTHS[m])
            for i in range(1, len(self.x_ticks)):
                m += 1
                if m == 1:
                    y += 1
                    x_labels.append(MONTHS[m] + ' ' + str(y))
                else:
                    x_labels.append(MONTHS[m])
                    if m == 12:
                        m = 0
        elif self.label == 'Quarter':
            # skip first year to match data
            y = int(self.lpoints[365].year)
            q = 1
            x_labels.append("Q"+str(q))
            for i in range(1, len(self.x_ticks)):
                q += 1
                if q > 4:
                    y += 1
                    q = 1
                    x_labels.append("Q" + str(q) + "\n"+str(y))
                else:
                    x_labels.append("Q"+str(q))
        else:
            x_labels.append(1)
            for i in range(1, len(self.x_ticks)):
                x_labels.append((x_labels[i-1]+1))
        self.plt.xticks(self.x_ticks, x_labels)
        self.plt.ylabel("Closing Price")
        self.plt.xlabel(self.label)
        self.plt.yticks(self.y_ticks, y_labels)
        self.plt.suptitle("Stock Market Analysis: {} to {}\n Real, Sine Analysis, & Future Prediction".format(self.lpoints[0].date, self.lpoints[-1].date))
        self.add_legend()

    def add_legend(self):
        red = mpatch.Patch(color='red', label='Real')
        blue = mpatch.Patch(color='blue', label='Predicted')
        r2 = mpatch.Patch(color='red', label='Min Linear Regression Prediction', fill=False, hatch='.....')
        g = mpatch.Patch(color='green', label='Max Linear Regression Prediction', fill=False, hatch='.....')
        b2 = mpatch.Patch(color='blue', label='Sinusoidal Regression Prediction', fill=False, hatch='.....')
        self.plt.legend(handles=[red,blue, g, r2, b2])
        self.largest_increase()
        self.smallest_increase()
        self.print_values()

    def print_values(self):
        string = """Max Difference: ${:.2f}
Min Difference: ${:.2f}
Largest Increase: ${:.2f}
    From: {}
    To:   {}
Largest Decrease: ${:.2f}
    From: {}
    To:   {}
Average Slope: {:.2f}""".format(self.biggest_difference(), self.smallest_difference(),
                   self.largest[2], self.largest[0].date, self.largest[1].date,
                   self.smallest[2], self.smallest[0].date, self.smallest[1].date,
                   self.overall_slope())
        self.plt.figtext(0.15, .2, string, fontsize=10)
        pass

    def approximate_line(self):
        """Approximates stock prices using sinusoidal regression."""
        self.get_peaks()
        self.get_amp_offset()
        self.get_omega()
        self.setup_chart()
        # Draw base plot
        self.plot(self.X, self.Y, color='red', style='solid')
        self.predict()
        # Draw sine regression plot
        self.plot(self.X, self.var_sinusoid(self.X, self.step), color='blue', style='solid')
        self.predict_sine()
        plt.show()        


def uinp():
    x = Stockalizer()
    f = input("Enter a filename: ")
    while True:
        try:
            a = input("Enter a step (default q): ").lower()
            if len(a) == 0:
                a = 'q'
            if a not in STEP.keys():
                a = int(a)
            break
        except Exception as e:
            print("""Error: Please enter a step value or a number.
                  Step values are:
                  d - Day
                  w - Week
                  m - Month
                  q - Quarter (default)
                  y - Year
                  """)
    x.set_step(a)
    x.load_yahoo_csv(f)
    x.approximate_line()
    exit()

if __name__ in '__main__':
    uinp()
#    x = Stockalizer()
#    x.set_step('q')
#    file = 'C:/Users/ngwoo/Downloads/BTC-USD.csv'
#    file = 'C:/Users/ngwoo/Downloads/^N225.csv'
#    x.load_yahoo_csv(file)
#    x.approximate_line()
