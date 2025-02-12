import time
from datetime import datetime
from numba import jit
import numpy as np
from utils import *

class GridDiagram:
    def __init__(self, gridsize, gridsize_max=15):
        self.n_max = gridsize_max # The maximum possible grid size. It is not possible to enlarge the grid if it reaches this size
        self.n = gridsize

        self.X_by_row = np.full(gridsize_max, -1) # An array s.t. X_by_row[i] is the column index of the 'X' marking in ith row.
        self.O_by_row = np.full(gridsize_max, -1) # An array s.t. O_by_row[i] is the column index of the 'O' marking in ith row.

    # Count the number of components
    def count_components(self):
        all_seen_X_rows = []
        components = 0
        cur_start_X_row = 0
        while(True):
            seen_X_rows = []
            while(True):
                if(cur_start_X_row in all_seen_X_rows):
                    cur_start_X_row += 1
                    if(cur_start_X_row == self.n):
                        return components
                else:
                    break
            
            cur_row = cur_start_X_row
            while(True):
                seen_X_rows.append(cur_row)
                all_seen_X_rows.append(cur_row)
                cur_row = first_occurrence(self.X_by_row, self.O_by_row[cur_row])
                if(cur_row in seen_X_rows):
                    components += 1
                    break

    # Count the number of up and down cusps                
    def count_cusps(self):
        up = 0
        down = 0

        for row, col in enumerate(self.X_by_row[:self.n]):
            dir = get_corner_dir(self.X_by_row, self.O_by_row, row, col, self.n)
            if(dir == DIR_NW):
                down += 1
            elif(dir == DIR_SE):
                up += 1
        
        for row, col in enumerate(self.O_by_row[:self.n]):
            dir = get_corner_dir(self.X_by_row, self.O_by_row, row, col, self.n)
            if(dir == DIR_NW):
                up += 1
            elif(dir == DIR_SE):
                down += 1
        
        return up, down
    
    def writhe(self):
        pos, neg = count_crossings(self.X_by_row, self.O_by_row, self.n)
        return pos - neg

    def tb(self):
        return self.writhe() - sum(self.count_cusps())/2
    
    def r(self):
        up, down = self.count_cusps()
        return (down - up)/2
    
    # A function to check that the X_by_row and O_by_row arrays are valid. Used for debugging
    def validate(self):
        if(set(self.X_by_row) != set(range(-1, self.n))):
            return -1
        if(set(self.O_by_row) != set(range(-1, self.n))):
            return -2
        if(np.count_nonzero(self.X_by_row != -1) != self.n):
            return -3
        if(np.count_nonzero(self.O_by_row != -1) != self.n):
            return -4
        return 0

    def convert_to_grid(self):
        grid = np.full((self.n, self.n), ' ')
        for idx in range(self.n):
            grid[idx, self.X_by_row[idx]] = 'X'
            grid[idx, self.O_by_row[idx]] = 'O'
            
        return grid

    def row_commutation_permitted(self, first_row_idx):
        if(first_row_idx < 0 or first_row_idx >= self.n-1):
            return False
        
        # Determine if the markings of one row are "sandwiched" between another or if they are disjoint
        a_l = [self.X_by_row[first_row_idx], self.O_by_row[first_row_idx]]
        a_l.sort()
        b_l = [self.X_by_row[first_row_idx+1], self.O_by_row[first_row_idx+1]]
        b_l.sort()

        # For what these variable names mean, see the row_commutation() function
        a_1 = a_l[0]
        a_2 = a_l[1]
        b_1 = b_l[0]
        b_2 = b_l[1]
        assert(a_1 < a_2)
        assert(b_1 < b_2)
        
        # The first two cases are when one is nested in another. The last two are when they are disjoint.
        if((a_1 < b_1 and a_2 > b_2) or (a_1 > b_1 and a_2 < b_2) or (a_2 < b_1) or (b_2 < a_1)):
            return True
        else: # Cannot perform it
            return False

    def column_commutation_permitted(self, first_col_idx):
        if(first_col_idx < 0 or first_col_idx >= self.n-1):
            return False
        
        # a_l contains the row indices of the letter markings in the first column
        # b_l contains the row indices of the letter markings in the second column
        a_l = [first_occurrence(self.X_by_row, first_col_idx), first_occurrence(self.O_by_row, first_col_idx)]
        b_l = [first_occurrence(self.X_by_row, first_col_idx+1), first_occurrence(self.O_by_row, first_col_idx+1)]

        # Sort them in increasing order
        a_l.sort()
        b_l.sort()
        
        a_1 = a_l[0]
        a_2 = a_l[1]
        b_1 = b_l[0]
        b_2 = b_l[1]

        if((a_1 < b_1 and a_2 > b_2) or (a_1 > b_1 and a_2 < b_2) or (a_2 < b_1) or (b_2 < a_1)):
            return True
        else: # Cannot
            return False

    def stabilization(self, row, col, letter, dir):
        stabilization_worker(self.X_by_row, self.O_by_row, row, col, letter, dir, self.n, [None, None],[None, None])
        self.n = self.n+1

    def stabilization_permitted(self, row, col, letter):
        if(letter == 1):
            if(self.X_by_row[row] == col):
                return True
            else:
                return False
        else:
            if(self.O_by_row[row] == col):
                return True
            else:
                return False

    def destabilization(self, 
                        row_coord, # Row coordinate of empty spot in square
                        col_coord, # Column coordinate of empty spot in square
                        letter, 
                        dir): # Position opposite of empty spot
       
        assert(self.X_by_row[row_coord] != col_coord and self.O_by_row[row_coord] != col_coord)

        if(letter == X):
            self.X_by_row[row_coord] = col_coord
            other_letter_dict = self.O_by_row
        else:
            self.O_by_row[row_coord] = col_coord
            other_letter_dict = self.X_by_row

        # Get the coordinates of the cell opposite to the empty cell in the 2x2 square
        if(dir == DIR_NE):
            opp_x = col_coord-1
            opp_y = row_coord+1
        elif(dir == DIR_NW):
            opp_x = col_coord+1
            opp_y = row_coord+1
        elif(dir == DIR_SE):
            opp_x = col_coord-1
            opp_y = row_coord-1
        elif(dir == DIR_SW):
            opp_x = col_coord+1
            opp_y = row_coord-1

        assert(other_letter_dict[opp_y] == opp_x)

        # Shrink the grid
        self.X_by_row[opp_y:self.n-1] = self.X_by_row[opp_y+1:self.n]
        self.X_by_row[self.n-1] = -1

        self.O_by_row[opp_y:self.n-1] = self.O_by_row[opp_y+1:self.n]
        self.O_by_row[self.n-1] = -1

        increment_if_geq(self.X_by_row, opp_x, -1)
        increment_if_geq(self.O_by_row, opp_x, -1)
        
        self.n = self.n-1 

    def Legendrian_Birth_Upper_Left_Corner(self): # Additional Legendrian elementary cobordism moves
        increment_if_geq(self.X_by_row, 0, 2)
        increment_if_geq(self.O_by_row, 0, 2)

        self.X_by_row[2:self.n+2] = self.X_by_row[0:self.n]
        self.O_by_row[2:self.n+2] = self.O_by_row[0:self.n]
        
        self.X_by_row[0] = 1
        self.X_by_row[1] = 0

        self.O_by_row[1] = 1
        self.O_by_row[0] = 0
        
        self.n += 2

    def Load_From_Diagram(self, diagram):
        a,b = diagram.shape
      
        assert(a == self.n)
        for row_idx, row in enumerate(diagram):
            for col_idx, entry in enumerate(row):
                if(entry == 'X'):
                    self.X_by_row[row_idx] = col_idx
                elif(entry == 'O'):
                    self.O_by_row[row_idx] = col_idx
                elif(entry == ' '):
                    pass
                else:
                    raise NotImplementedError

    def destabilization_permitted(self, 
                                 row_coord, # Coordinates of empty spot in 2x2 grid
                                 col_coord, 
                                 letter, # Letter that presumably appears twice in the square
                                 dir): # Direction
        
        if(row_coord < 0 or row_coord > self.n-1 or col_coord < 0 and col_coord > self.n-1):
            return False
        
        if(self.X_by_row[row_coord] == col_coord or self.O_by_row[row_coord] == col_coord):
            return False
        
        if(letter == X):
            same_letter_dict = self.X_by_row
            other_letter_dict = self.O_by_row
        elif(letter == O):
            same_letter_dict = self.O_by_row
            other_letter_dict = self.X_by_row

        
        if(dir == DIR_NE):
            # For each direction, we perform three validity checks:

            # 1: Whether the 2x2 destabilization square is too close to the boundaries
            if(row_coord == self.n-1 or col_coord == 0):
                return False
            
            # 2: Whether the two relevant corners of the 2x2 square are the correct letter
            if(not (same_letter_dict[row_coord+1] == col_coord and same_letter_dict[row_coord] == col_coord-1)):
                return False
            
            # 3: Whether there exists an empty cell in the required location
            if(other_letter_dict[row_coord+1] != col_coord-1):
                return False
            
        elif(dir == DIR_SW):
            if(row_coord == 0 or col_coord == self.n-1):
                return False
            if(not (same_letter_dict[row_coord-1] == col_coord and same_letter_dict[row_coord] == col_coord+1)):
                return False
            if(other_letter_dict[row_coord-1] != col_coord+1):
                return False
            
        elif(dir == DIR_NW):
            if(row_coord == self.n-1 or col_coord == self.n-1):
                return False
            if(not (same_letter_dict[row_coord+1] == col_coord and same_letter_dict[row_coord] == col_coord+1)):
                return False
            if(other_letter_dict[row_coord+1] != col_coord+1):
                return False
        
        else: 
            if(row_coord == 0 or col_coord == 0):
                return False
            if(not (same_letter_dict[row_coord-1] == col_coord and same_letter_dict[row_coord] == col_coord-1)):
                return False
            if(other_letter_dict[row_coord-1] != col_coord-1):
                return False
        
        return True
    
    # Perform a copinch
    def Legendrian_CoPinch_Row(self, first_row):
        assert(first_row <= self.n-2)
        
        # Get the column indices of the letter markings in each row in a sorted array
        row1_nonempty = [self.X_by_row[first_row], self.O_by_row[first_row]]
        row1_nonempty.sort()
        
        row2_nonempty = [self.X_by_row[first_row+1], self.O_by_row[first_row+1]]
        row2_nonempty.sort()

        
        if(row1_nonempty[0] > row2_nonempty[1]):

            # Make sure the corner directions are correct
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[0], self.n) == DIR_NW and 
               get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[1], self.n) == DIR_SE):
               
                if(self.X_by_row[first_row] == row1_nonempty[0]):
                    
                    # Since this function should only have been run after the Legendrian_Copinch_row_permitted function has been run on the row,
                    # we run this assert() to ensure that both of the markings in the middle are of the same letter
                    assert(self.X_by_row[first_row+1] == row2_nonempty[1])

                    # Perform the letter swap for the copinch
                    self.X_by_row[first_row], self.X_by_row[first_row+1] = self.X_by_row[first_row+1], self.X_by_row[first_row]
                
                elif(self.O_by_row[first_row] == row1_nonempty[0]):
                    assert(self.O_by_row[first_row+1] == row2_nonempty[1])
                    self.O_by_row[first_row], self.O_by_row[first_row+1] = self.O_by_row[first_row+1], self.O_by_row[first_row]
                
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        # The other case
        elif(row1_nonempty[1] > row2_nonempty[0] and row1_nonempty[0] < row2_nonempty[0] and row2_nonempty[1] > row1_nonempty[1]):
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[1], self.n) == DIR_SE and 
               get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[0], self.n) == DIR_NW):
               
                if(self.X_by_row[first_row] == row1_nonempty[1]):
                    assert(self.X_by_row[first_row+1] == row2_nonempty[0])
                    self.X_by_row[first_row], self.X_by_row[first_row+1] = self.X_by_row[first_row+1], self.X_by_row[first_row]
                
                elif(self.O_by_row[first_row] == row1_nonempty[1]):
                    assert(self.O_by_row[first_row+1] == row2_nonempty[0])
                    self.O_by_row[first_row], self.O_by_row[first_row+1] = self.O_by_row[first_row+1], self.O_by_row[first_row]
                
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    def Legendrian_CoPinch_Row_Permitted(self, first_row):
        if(first_row > self.n-2):
            return False
        row1_nonempty = [self.X_by_row[first_row], self.O_by_row[first_row]]
        row1_nonempty.sort()
        
        row2_nonempty = [self.X_by_row[first_row+1], self.O_by_row[first_row+1]]
        row2_nonempty.sort()

        if(row1_nonempty[0] > row2_nonempty[1]):
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[0], self.n) == DIR_NW and get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[1], self.n) == DIR_SE):
                if((self.X_by_row[first_row] == row1_nonempty[0] and self.X_by_row[first_row+1] == row2_nonempty[1]) or (self.O_by_row[first_row] == row1_nonempty[0] and self.O_by_row[first_row+1] == row2_nonempty[1])):
                    for col_idx in range(row2_nonempty[1]+1, row1_nonempty[0]):
                        col_nonempty = [first_occurrence(self.X_by_row, col_idx), first_occurrence(self.O_by_row, col_idx)]
                        assert(-1 not in col_nonempty)
                        col_nonempty.sort()

                        if(not (col_nonempty[0] > first_row+1 or col_nonempty[1] < first_row)):
                            return False
                    
                    return True

        elif(row1_nonempty[1] > row2_nonempty[0] and row1_nonempty[0] < row2_nonempty[0] and row2_nonempty[1] > row1_nonempty[1]):
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[1], self.n) == DIR_SE and get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[0], self.n) == DIR_NW):
                
                if((self.X_by_row[first_row] == row1_nonempty[1] and self.X_by_row[first_row+1] == row2_nonempty[0]) or (self.O_by_row[first_row] == row1_nonempty[1] and self.O_by_row[first_row+1] == row2_nonempty[0])):    
                    for col_idx in range(row2_nonempty[0]+1, row1_nonempty[1]):
                        col_nonempty = [first_occurrence(self.X_by_row, col_idx), first_occurrence(self.O_by_row, col_idx)]
                        assert(-1 not in col_nonempty)
                        col_nonempty.sort()

                        if(not (col_nonempty[0] > first_row+1 or col_nonempty[1] < first_row)):
                            return False
                    
                    return True
        
        return False

    def Legendrian_Pinch_Row(self, first_row):

        if(first_row > self.n-2):
            return False
        
        row1_nonempty = [self.X_by_row[first_row], self.O_by_row[first_row]]
        row1_nonempty.sort()
        
        row2_nonempty = [self.X_by_row[first_row+1], self.O_by_row[first_row+1]]
        row2_nonempty.sort()
        
        if(row1_nonempty[0] < row2_nonempty[1] and row1_nonempty[1] > row2_nonempty[1] and row1_nonempty[0] > row2_nonempty[0]):
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[0], self.n) == DIR_SW and get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[1], self.n) == DIR_NE):
                if(self.X_by_row[first_row] == row1_nonempty[0]):
                    assert(self.X_by_row[first_row+1] == row2_nonempty[1])
                    self.X_by_row[first_row], self.X_by_row[first_row+1] = self.X_by_row[first_row+1], self.X_by_row[first_row]
                elif(self.O_by_row[first_row] == row1_nonempty[0]):
                    assert(self.O_by_row[first_row+1] == row2_nonempty[1])
                    self.O_by_row[first_row], self.O_by_row[first_row+1] = self.O_by_row[first_row+1], self.O_by_row[first_row]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        elif(row1_nonempty[1] < row2_nonempty[0]):
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[1], self.n) == DIR_NE and get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[0], self.n) == DIR_SW):
                if(self.X_by_row[first_row] == row1_nonempty[1]):
                    assert(self.X_by_row[first_row+1] == row2_nonempty[0])
                    self.X_by_row[first_row], self.X_by_row[first_row+1] = self.X_by_row[first_row+1], self.X_by_row[first_row]
                elif(self.O_by_row[first_row] == row1_nonempty[1]):
                    assert(self.O_by_row[first_row+1] == row2_nonempty[0])
                    self.O_by_row[first_row], self.O_by_row[first_row+1] = self.O_by_row[first_row+1], self.O_by_row[first_row]
                else:
                    raise NotImplementedError
                
                
        else:
            raise NotImplementedError

    # Check to see if pinching is permitted for a row
    def Legendrian_Pinch_Row_Permitted(self, 
                                       first_row # The index of the first of the two rows. The second row is the row directly below this one
                                       ):
        
        # Ensure first_row is not the very last row
        if(first_row > self.n-2):
            return False
        

        row1_nonempty = [self.X_by_row[first_row], self.O_by_row[first_row]]
        row1_nonempty.sort()
        
        row2_nonempty = [self.X_by_row[first_row+1], self.O_by_row[first_row+1]]
        row2_nonempty.sort()
        
        if(row1_nonempty[0] < row2_nonempty[1] and row1_nonempty[1] > row2_nonempty[1] and row1_nonempty[0] > row2_nonempty[0]):
            
            # check Corners are valid
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[0], self.n) == DIR_SW and 
               get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[1], self.n) == DIR_NE):
               
                # Make sure the two middle markings are the same letter
                if((self.X_by_row[first_row] == row1_nonempty[0] and self.X_by_row[first_row+1] == row2_nonempty[1]) or 
                   (self.O_by_row[first_row] == row1_nonempty[0] and self.O_by_row[first_row+1] == row2_nonempty[1])):
                    
                    # Make sure there are no crossing obstructions between the two middle letters 
                    for col_idx in range(row1_nonempty[0]+1, row2_nonempty[1]):
                        col_nonempty = [first_occurrence(self.X_by_row, col_idx), first_occurrence(self.O_by_row, col_idx)]
                        col_nonempty.sort()

                        if(not (col_nonempty[0] > first_row+1 or col_nonempty[1] < first_row)):
                            return False
                        
                    return True
        elif(row1_nonempty[1] < row2_nonempty[0]):
            
            # Check corners are valid
            if(get_corner_dir(self.X_by_row, self.O_by_row, first_row, row1_nonempty[1], self.n) == DIR_NE and 
               get_corner_dir(self.X_by_row, self.O_by_row, first_row+1, row2_nonempty[0], self.n) == DIR_SW):
                
                # Make sure the two middle markings are the same letter
                if((self.X_by_row[first_row] == row1_nonempty[1] and self.X_by_row[first_row+1] == row2_nonempty[0]) or 
                   (self.O_by_row[first_row] == row1_nonempty[1] and self.O_by_row[first_row+1] == row2_nonempty[0])):

                    # Make sure there are no crossing obstructions between the two middle letters 
                    for col_idx in range(row1_nonempty[1]+1, row2_nonempty[0]):
                        col_nonempty = [first_occurrence(self.X_by_row, col_idx), first_occurrence(self.O_by_row, col_idx)]
                        col_nonempty.sort()

                        if(not (col_nonempty[0] > first_row+1 or col_nonempty[1] < first_row)):
                            return False

                    return True
                    
        return False
    
    def Legendrian_Death(self, 
                         row_idx,    # row and column of top left position of the 2x2 unknot
                         col_idx):
        if(self.n <= 3):
            raise NotImplementedError
        
        if(row_idx < 0 or row_idx >= self.n-1 or col_idx < 0 or col_idx >= self.n-1):
            raise NotImplementedError
        
        '''
            Make sure it either looks like
            XO
            OX

            or

            OX
            XO
        '''
        if((self.X_by_row[row_idx] == col_idx and self.X_by_row[row_idx+1] == col_idx+1 and self.O_by_row[row_idx] == col_idx+1 and self.O_by_row[row_idx+1] == col_idx)
           or
           (self.O_by_row[row_idx] == col_idx and self.O_by_row[row_idx+1] == col_idx+1 and self.X_by_row[row_idx] == col_idx+1 and self.X_by_row[row_idx+1] == col_idx)
           ):
            self.X_by_row[row_idx:self.n-2] = self.X_by_row[row_idx+2:self.n]
            self.O_by_row[row_idx:self.n-2] = self.O_by_row[row_idx+2:self.n]
            self.X_by_row[self.n-2] = -1
            self.X_by_row[self.n-1] = -1
            self.O_by_row[self.n-2] = -1
            self.O_by_row[self.n-1] = -1
            
            self.n = self.n - 2
            increment_if_geq(self.X_by_row, col_idx+2, -2)
            increment_if_geq(self.O_by_row, col_idx+2, -2)
            
        else:
            raise NotImplementedError
       
    def Legendrian_Death_Permitted(self, 
                         row_idx,    # row and column of top left position of the "loop"
                         col_idx):
        if(self.n <= 3):
            return False
        
        if(row_idx < 0 or row_idx >= self.n-1 or col_idx < 0 or col_idx >= self.n-1):
            return False
        
        if((self.X_by_row[row_idx] == col_idx and self.X_by_row[row_idx+1] == col_idx+1 and self.O_by_row[row_idx] == col_idx+1 and self.O_by_row[row_idx+1] == col_idx)
           or
           (self.O_by_row[row_idx] == col_idx and self.O_by_row[row_idx+1] == col_idx+1 and self.X_by_row[row_idx] == col_idx+1 and self.X_by_row[row_idx+1] == col_idx)
           ):
            return True
            
        else:
            return False
    
    def copy_gd(self): # Returns copy of grid diagram
        new_gd = GridDiagram(self.n)
        new_gd.X_by_row = self.X_by_row.copy()
        new_gd.O_by_row = self.O_by_row.copy()
    
        return new_gd
    
    # Get integer hash of a grid
    def hash(self):
        return hash((self.X_by_row.tobytes(), self.O_by_row.tobytes(), self.n))
    

    



# Get possible grid moves to be performed on a grid 
def possible_moves(gd: GridDiagram, 
                   
                   # Optional flags
                   get_pinches=True, 
                   get_copinches=True, 
                   get_deaths=True):

    ok_row_commutations = []
    ok_col_commutations = []
    ok_row_pinches = []
    ok_row_copinches = []

    ok_death_moves = []
    ok_stab = []
    ok_destab = []

    # The return array
    ret = []

    for row_idx in range(gd.n):
        if(gd.row_commutation_permitted(row_idx)):
            ok_row_commutations.append(row_idx)
            
    if(get_pinches):
        for row_idx in range(gd.n):
            if(gd.Legendrian_Pinch_Row_Permitted(row_idx)):
                ok_row_pinches.append(row_idx)

    if(get_copinches):
        for row_idx in range(gd.n):
            if(gd.Legendrian_CoPinch_Row_Permitted(row_idx)):
                ok_row_copinches.append(row_idx)
    
    for col_idx in range(gd.n):
        if(gd.column_commutation_permitted(col_idx)):
            ok_col_commutations.append(col_idx)

    if(get_deaths and gd.n > 2):
        for row in range(gd.n):
            col = gd.X_by_row[row]
            if(gd.Legendrian_Death_Permitted(row, col)):
                ok_death_moves.append((row, col))
        for row in range(gd.n):
            col = gd.O_by_row[row]
            if(gd.Legendrian_Death_Permitted(row, col)):
                ok_death_moves.append((row, col))
        
    for row in range(gd.n):
        col = gd.X_by_row[row]
        ok_stab.append((row, col, DIR_NE, 1))
        ok_stab.append((row, col, DIR_SW, 1))
    
    for row in range(gd.n):
        col = gd.O_by_row[row]
        ok_stab.append((row, col, DIR_NE, 2))
        ok_stab.append((row, col, DIR_SW, 2))
   

    for row in range(gd.n):
        col = gd.X_by_row[row]
        if(gd.destabilization_permitted(row, col+1, letter=1, dir=DIR_NE)):
            ok_destab.append((row, col+1, DIR_NE, 1))
   
    for row in range(gd.n):
        col = gd.X_by_row[row]
        if(gd.destabilization_permitted(row+1, col, letter=1, dir=DIR_SW)):
            ok_destab.append((row+1, col, DIR_SW, 1))
    
    for row in range(gd.n):
        col = gd.O_by_row[row]
        if(gd.destabilization_permitted(row, col+1, letter=2, dir=DIR_NE)):
            ok_destab.append((row, col+1, DIR_NE, 2))
   
    for row in range(gd.n):
        col = gd.O_by_row[row]
        if(gd.destabilization_permitted(row+1, col, letter=2, dir=DIR_SW)):
            ok_destab.append((row+1, col, DIR_SW, 2))
    

    ret = [ok_row_commutations, ok_col_commutations]
    if(get_pinches):
        ret.append(ok_row_pinches)
    if(get_copinches):
        ret.append(ok_row_copinches)
    if(get_deaths):
        ret.append(ok_death_moves)
    ret += [ok_stab, ok_destab]

    
    return ret