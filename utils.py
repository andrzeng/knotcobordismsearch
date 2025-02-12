import numpy as np
from datetime import datetime
from numba import jit

# Define constants for possible corner directions
DIR_NE = 0
DIR_NW = 1
DIR_SE = 2
DIR_SW = 3

X = 1
O = 2

dirs_names = ['NE', 'NW', 'SE', 'SW']

'''def numerical_to_grid(array):
    n = array.shape[0]
    grid = np.array([np.array([' '] * n)] * n)
    for row_idx, row in enumerate(array):
        for col_idx, entry in enumerate(row):
            if(entry == 1):
                grid[row_idx, col_idx] = 'X'
            elif(entry == 2):
                grid[row_idx, col_idx] = 'O'

    return grid'''

def marking_tuple_to_grid(tup):
    X_by_row = tup[0]
    O_by_row = tup[1]

    n = max(X_by_row)+1
    grid = np.full((n,n), ' ')
    for i in range(n):
        for j in range(n):
            if(X_by_row[i] == j):
                grid[i,j] = 'X'
            if(O_by_row[i] == j):
                grid[i,j] = 'O'

    return grid

logfile = ''

def set_logfile(file):
    global logfile
    logfile = file

def log(text, file=None, print_to_out=True, print_datetime=True):
    if(file is None):
        global logfile
    else:
        logfile = file
    
    print_text = ''
    if(print_datetime):
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print_text += date_time
        print_text += ': '
    print_text += text
    print_text += '\n'

    with open(logfile, "a") as f:
        f.write(print_text)
    
    if(print_to_out):
        print(print_text, end='')



'''
    Functions with the @jit decorator are compiled by Numba and run directly as machine code, bypassing the usual Python interpreter. This makes "jitted" (JIT stands for just in time)
    functions 1-2 orders of magnitude faster than non-jitted functions. The trade-off is that jitted functions can only call Numpy functions, 
    predefined Python functions, and other jitted functions. There are other restrictions as well, see https://numba.readthedocs.io/en/stable/user/jit.html
'''

'''
    This function returns the first occurrence of a value "val" in an array "arr" , for -1 if the value is not found.
'''
@jit(nopython=True)
def first_occurrence(arr, val):
    for i in range(len(arr)):
        if(arr[i] == val):
            return i
    return -1

'''
    This function adds "value" to each element of "arr" that is greater than or equal to "min_threshold"
'''
@jit(nopython=True)
def increment_if_geq(arr, min_threshold, value):
    for i in range(len(arr)):
        if(arr[i] >= min_threshold):
            arr[i] += value

'''
    This function counts the number of + and - crossings in a grid, given its "X_by_row" and "O_by_row" arrays, along with its height/width size "n"
'''
@jit(nopython=True)
def count_crossings(X_by_row, O_by_row, n):
    pos = 0
    neg = 0
    for row in range(n):
        x_marking, o_marking = X_by_row[row], O_by_row[row]
        min_col = min(x_marking, o_marking)
        cols_interval = [min_col, x_marking + o_marking - min_col]

        for col in range(cols_interval[0], cols_interval[1]):
            x_row_idx = first_occurrence(X_by_row, col)
            o_row_idx = first_occurrence(O_by_row, col)
            min_row = min(x_row_idx, o_row_idx)
            rows_interval = [min_row, x_row_idx + o_row_idx - min_row]

            if(row > rows_interval[0] and row < rows_interval[1]): # crossing detected
                if(min_col == x_marking): # X           O in the row
                    if(min_row == x_row_idx): # X            O  in the col
                        neg += 1
                    else:
                        pos += 1
                else: # O         X  in the row
                    if(min_row == x_row_idx): # X            O  in the col
                        pos += 1
                    else:
                        neg += 1
                        
    return pos, neg  

'''
    This returns the direction of a corner at coordinates ("row", "col")
'''
@jit(nopython=True)
def get_corner_dir(X_by_row, O_by_row, row, col, n):
    assert(row < n and col < n)
    if(X_by_row[row] == col):
        if(O_by_row[row] < col and first_occurrence(O_by_row, col) > row):
            return DIR_NE
        if(O_by_row[row] > col and first_occurrence(O_by_row, col) > row):
            return DIR_NW
        if(O_by_row[row] < col and first_occurrence(O_by_row, col) < row):
            return DIR_SE
        if(O_by_row[row] > col and first_occurrence(O_by_row, col) < row):
            return DIR_SW
    elif(O_by_row[row] == col):
        if(X_by_row[row] < col and first_occurrence(X_by_row, col) > row):
            return DIR_NE
        if(X_by_row[row] > col and first_occurrence(X_by_row, col) > row):
            return DIR_NW
        if(X_by_row[row] < col and first_occurrence(X_by_row, col) < row):
            return DIR_SE
        if(X_by_row[row] > col and first_occurrence(X_by_row, col) < row):
            return DIR_SW
    else:
        raise NotImplementedError

'''
    This function performs stabilization on a grid.
'''
@jit(nopython=True)
def stabilization_worker(X_by_row, O_by_row, row, col, letter, dir, n, row_olm_coords, col_olm_coords):
        # Step 1: Find locations of the other letter along the row and column of stabilization

        # If the letter at the stabilization location is 'X', then the "other_letter" is 'O', and vice versa.
        if(letter == X):
            same_letter_dict = X_by_row
            other_letter_dict = O_by_row
        else:
            same_letter_dict = O_by_row
            other_letter_dict = X_by_row

        # Find the indices of the other_letter in the row an column. 
        ol_idx_in_col = first_occurrence(other_letter_dict, col)
        ol_idx_in_row = other_letter_dict[row]

        # Put them in coordinate form
        col_olm_coords = [ol_idx_in_col, col]
        row_olm_coords = [row, ol_idx_in_row]

        # We enlarge the grid by increasing coordinate values after the location of stabilization
        increment_if_geq(X_by_row, col+1, 1)
        increment_if_geq(O_by_row, col+1, 1)
        
        X_by_row[row+2: n+1] = X_by_row[row+1 : n]
        O_by_row[row+2: n+1] = O_by_row[row+1 : n]
        
        '''
            We fill in the empty 2x2 square in the grid with three new markings. For example
            X:NE would be

            X 
            OX
        '''
        if(dir == DIR_NE):
            same_letter_dict[row] = col
            same_letter_dict[row+1] = col+1
            other_letter_dict[row+1] = col

            offset_row = 0
            offset_col = 1
        elif(dir == DIR_NW):
            same_letter_dict[row] = col+1
            same_letter_dict[row+1] = col
            other_letter_dict[row+1] = col+1

            offset_row = 0
            offset_col = 0
        elif(dir == DIR_SE):
            same_letter_dict[row] = col+1
            same_letter_dict[row+1] = col
            other_letter_dict[row] = col

            offset_row = 1
            offset_col = 1
        else: 
            same_letter_dict[row] = col
            same_letter_dict[row+1] = col+1
            other_letter_dict[row] = col+1

            offset_row = 1
            offset_col = 0

        # Add back missing markings in the row, col
        col_olm_coords[1] += offset_col
        row_olm_coords[0] += offset_row
        if(col_olm_coords[0] > row):
            col_olm_coords[0] += 1
        if(row_olm_coords[1] > col):
            row_olm_coords[1] += 1
        
        other_letter_dict[col_olm_coords[0]] = col_olm_coords[1]
        other_letter_dict[row_olm_coords[0]] = row_olm_coords[1]

# Perform a row commutation
@jit(nopython=True)
def row_commutation(X_by_row, O_by_row, first_row_idx, n):
    if(first_row_idx < 0 or first_row_idx >= n-1):
        raise NotImplementedError
    
    # Determine if the markings of one row are "sandwiched" between another or if they are disjoint
    a_l = [X_by_row[first_row_idx], O_by_row[first_row_idx]]
    a_l.sort()
    b_l = [X_by_row[first_row_idx+1], O_by_row[first_row_idx+1]]
    b_l.sort()

    # a_1 = sorted length-2 array of the column indices of the letters in the first row
    # a_2 the same but for the second row
    a_1 = a_l[0]
    a_2 = a_l[1]
    b_1 = b_l[0]
    b_2 = b_l[1]
    assert(a_1 < a_2)
    assert(b_1 < b_2)
    
    # The first two cases are when one is nested in another. The last two are when they are disjoint.
    if((a_1 < b_1 and a_2 > b_2) or (a_1 > b_1 and a_2 < b_2) or (a_2 < b_1) or (b_2 < a_1)):
        # Is possible
        X_by_row[first_row_idx], X_by_row[first_row_idx+1] = X_by_row[first_row_idx+1], X_by_row[first_row_idx]
        O_by_row[first_row_idx], O_by_row[first_row_idx+1] = O_by_row[first_row_idx+1], O_by_row[first_row_idx]

    else: # Cannot
        raise NotImplementedError

# Column commutation
@jit(nopython=True)
def column_commutation(X_by_row, O_by_row, first_col_idx, n):
    # We assume it is permitted
    
    if(first_col_idx < 0 or first_col_idx >= n-1):
        raise NotImplementedError
    
    X_row_coord_firstcol = first_occurrence(X_by_row, first_col_idx)
    X_row_coord_secondcol = first_occurrence(X_by_row, first_col_idx+1)
    X_by_row[X_row_coord_firstcol] = first_col_idx+1
    X_by_row[X_row_coord_secondcol] = first_col_idx

    O_row_coord_firstcol = first_occurrence(O_by_row, first_col_idx)
    O_row_coord_secondcol = first_occurrence(O_by_row, first_col_idx+1)
    O_by_row[O_row_coord_firstcol] = first_col_idx+1
    O_by_row[O_row_coord_secondcol] = first_col_idx