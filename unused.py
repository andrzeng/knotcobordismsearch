
    def stabilization(self, row, col, letter, dir):
        # Step 1: Find locations of the other letter along the row and column of stabilization
        if(letter == 1):
            same_letter_dict = self.X_by_row
            other_letter_dict = self.O_by_row
        else:
            same_letter_dict = self.O_by_row
            other_letter_dict = self.X_by_row

        ol_idx_in_col = first_occurrence(other_letter_dict, col)
        ol_idx_in_row = other_letter_dict[row]

        col_olm_coords = [ol_idx_in_col, col]
        row_olm_coords = [row, ol_idx_in_row]

        # Step 2: Clear the row and column
        
        # Step 3: Add new row, col

        increment_if_geq(self.X_by_row, col+1, 1)
        increment_if_geq(self.O_by_row, col+1, 1)
        
        self.X_by_row[row+2: self.n+1] = self.X_by_row[row+1 : self.n]
        self.O_by_row[row+2: self.n+1] = self.O_by_row[row+1 : self.n]
   
        # Step 4: Insert new markings in the 2x2 square
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
        else: #elif(dir == DIR_SW):
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
        
        self.n = self.n+1