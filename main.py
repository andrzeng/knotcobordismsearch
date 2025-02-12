from datetime import datetime
from presetgrids import *
from utils import *
from search import *
from grid import *
from knotwrapper import *


if __name__ == '__main__':
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    gd1 = GridDiagram(gridsize=9)
    gd1.Load_From_Diagram(grid_m_10_132)

    gd2 = GridDiagram(gridsize=9)
    gd2.Load_From_Diagram(grid_m_10_145_1)

    Lambda_minus = KnotWrapper(gd=gd1)
    Lambda_plus = KnotWrapper(gd=gd2)

    t_minus, t_plus = search(Lambda_minus, 
                    Lambda_plus, 
                    tree_depth=70,
                    max_grid_size=12,
                    max_births=1,
                    max_deaths=5, 
                    logfile=f"m(10_132) to m(10_145)_1 log{date_time}.txt",
                    isotopy_only=False)
