import time
from datetime import datetime
from presetgrids import *
from numba import jit
import numpy as np
from utils import *
from search import *
from grid import *
from knotwrapper import *


def search(L_minus,
           L_plus,
           tree_depth,

           # Default Search parameters
           max_grid_size=8,
           max_births=2,
           max_deaths=2,
           max_stabilizations=100000,
           max_destabilizations=100000,
           max_pinches=100000,
           max_copinches=100000,
           max_commutations=100000,
           isotopy_only = False,

           # Default log file 
           logfile='log.txt'
           ):
  

    if(isotopy_only):
        max_births = 0
        max_deaths = 0
        max_pinches = 0
        max_copinches = 0

    set_logfile(logfile)

    minus_tb = L_minus.gd.tb()
    plus_tb = L_plus.gd.tb()

    minus_r = L_minus.gd.r()
    plus_r = L_plus.gd.r()

    # 0 will be the "negative" side; "1" will be the positive side.
    tree_minus_seen = {} # Format will be "hashvalue" : [list of unique knot descriptors]
    tree_plus_seen = {} # Same

    tree_minus_current = [L_minus]
    tree_plus_current = [L_plus]
    tree_minus_seen[L_minus.hash()] = L_minus
    tree_plus_seen[L_plus.hash()] = L_plus
    
    t_minus_births=0
    t_minus_stabilizations=0
    t_minus_destabilizations=0
    t_minus_copinches=0
    t_minus_r_commutations=0 
    t_minus_c_commutations=0

    t_plus_deaths=0
    t_plus_stabilizations=0
    t_plus_destabilizations=0
    t_plus_pinches=0
    t_plus_r_commutations=0
    t_plus_c_commutations=0

    found = False

    log("starting")
    param_str = f"""tree_depth={tree_depth}, \n
           max_grid_size={max_grid_size}, \n
           max_births={max_births}, \n
           max_deaths={max_deaths}, \n
           max_stabilizations={max_stabilizations}, \n
           max_destabilizations={max_destabilizations}, \n
           max_pinches={max_pinches}, \n
           max_copinches={max_copinches}, \n
           max_commutations={max_commutations}, \n
           isotopy_only={isotopy_only}"""
    log(param_str)
    log("Minus direction:\n")
    log("\n" + str(L_minus.gd.convert_to_grid()))
    log(f"(tb,r) = ({minus_tb},{minus_r})")
    log("Positive direction:\n")
    log("\n" + str(L_plus.gd.convert_to_grid()))
    log(f"(tb,r) = ({plus_tb},{plus_r})")

    for level in range(tree_depth):
        log(f'At level {level}. At this level, there are {len(tree_minus_current)} diagrams to be checked from the first tree and {len(tree_plus_current)} diagrams to be checked from the second tree.')
        log(f'{len(tree_minus_seen)} diagrams have been seen in the first tree in total')
        log(f'{len(tree_plus_seen)} diagrams have been seen in the second tree in total')
        
        tree_minus_next = []
        tree_plus_next = []

        for index, knotwrapper in enumerate(tree_minus_current):
            if(index % 10000 == 0):
                log(f"Processing {index}/{len(tree_minus_current)}")
            
            # Get possible moves
            ok_row_commutations, ok_col_commutations, ok_row_copinches, ok_death_moves, ok_stab, ok_destab = possible_moves(knotwrapper.gd, get_pinches=False)
          
            if(knotwrapper.n_commutations < max_commutations):
                for row_idx in ok_row_commutations:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    row_commutation(candidate_gd.X_by_row, candidate_gd.O_by_row, row_idx, candidate_gd.n)

                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_minus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'commuted row {row_idx} with row {row_idx+1}', commutations=1)
                        tree_minus_next.append(new_kw)
                        tree_minus_seen[gd_hash] = new_kw
                        t_minus_r_commutations += 1

                        if(gd_hash in tree_plus_seen):
                            found = True
                            break

                for col_idx in ok_col_commutations:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    column_commutation(candidate_gd.X_by_row, candidate_gd.O_by_row, col_idx, candidate_gd.n)

                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_minus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'commuted col {col_idx} with col {col_idx+1}', commutations=1)
                        tree_minus_next.append(new_kw)
                        tree_minus_seen[gd_hash] = new_kw
                        t_minus_c_commutations += 1
                        
                        if(gd_hash in tree_plus_seen):
                            found = True
                            break
                        
            if(knotwrapper.n_copinches < max_copinches):   
                for row_idx in ok_row_copinches:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.Legendrian_CoPinch_Row(row_idx)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_minus_seen):
                        
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'copinched row {row_idx} with row {row_idx+1}', copinches=1)
                        tree_minus_seen[gd_hash] = new_kw

                    if not(candidate_gd.count_components() == 1 and candidate_gd.r() != plus_r):
                        tree_minus_next.append(new_kw)
                        t_minus_copinches += 1

                        if(gd_hash in tree_plus_seen):
                            found = True
                            break
           
            # Legendrian birth in upper left corner:
            if(knotwrapper.n_births < max_births and knotwrapper.n < max_grid_size):
                candidate_gd = knotwrapper.gd.copy_gd()
                candidate_gd.Legendrian_Birth_Upper_Left_Corner()
                
                gd_hash = candidate_gd.hash()
                if(gd_hash not in tree_minus_seen):
                    new_kw = produce_new_kw(candidate_gd, knotwrapper, f'Birthed an unknot in the upper left corner', births=1)
                    tree_minus_next.append(new_kw)
                    tree_minus_seen[gd_hash] = new_kw

                    t_minus_births += 1

                    if(gd_hash in tree_plus_seen):
                        found = True
                        break
                      

            if(knotwrapper.n_stabilizations < max_stabilizations and knotwrapper.n < max_grid_size):
                for row_idx, col_idx, dir_name_idx, the_letter in ok_stab:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    
                    candidate_gd.stabilization(row_idx, col_idx, the_letter, dir_name_idx)
                    
                    assert(candidate_gd.n <= max_grid_size)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_minus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f"Performed {the_letter}:{dirs_names[dir_name_idx]} stabilization at {(row_idx, col_idx)}", stabilizations=1)
                        tree_minus_next.append(new_kw)
                        tree_minus_seen[gd_hash] = new_kw
                        t_minus_stabilizations += 1

                        if(gd_hash in tree_plus_seen):
                            found = True
                            break
                        

            if(knotwrapper.n_destabilizations < max_destabilizations):
                for row_idx, col_idx, dir_name_idx, the_letter in ok_destab:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.destabilization(row_idx, col_idx, the_letter, dir_name_idx)

                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_minus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f"Performed {the_letter}:{dirs_names[dir_name_idx]} destabilization at {(row_idx, col_idx)}", destabilizations=1)
                        tree_minus_next.append(new_kw)
                        tree_minus_seen[gd_hash] = new_kw
                        
                        t_minus_destabilizations += 1
                        if(gd_hash in tree_plus_seen):
                            found = True
                            break

    
        for index, knotwrapper in enumerate(tree_plus_current):
            if(found):
                break

            if(index % 10000 == 0):
                log(f"Processing {index}/{len(tree_plus_current)}")
            ok_row_commutations, ok_col_commutations, ok_row_pinches, ok_death_moves, ok_stab, ok_destab = possible_moves(knotwrapper.gd, get_copinches=False)
            
            if(knotwrapper.n_commutations < max_commutations):
                for row_idx in ok_row_commutations:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    row_commutation(candidate_gd.X_by_row, candidate_gd.O_by_row, row_idx, candidate_gd.n)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'commuted row {row_idx} with row {row_idx+1}', commutations=1)
                        tree_plus_next.append(new_kw)
                        tree_plus_seen[gd_hash] = new_kw

                        t_plus_r_commutations += 1
                        if(gd_hash in tree_minus_seen):
                            found = True
                            break
                        
                for col_idx in ok_col_commutations:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    column_commutation(candidate_gd.X_by_row, candidate_gd.O_by_row, col_idx, candidate_gd.n)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):             
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'commuted col {col_idx} with col {col_idx+1}', commutations=1)
                        tree_plus_next.append(new_kw)
                        tree_plus_seen[gd_hash] = new_kw

                        t_plus_c_commutations += 1
                        if(gd_hash in tree_minus_seen):
                            found = True
                            break
       
            if(knotwrapper.n_pinches < max_pinches):         
                for row_idx in ok_row_pinches:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.Legendrian_Pinch_Row(row_idx)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'pinched row {row_idx} with row {row_idx+1}', pinches=1)
                        tree_plus_seen[gd_hash] = new_kw

                        components = candidate_gd.count_components()
                        tb_diff = minus_tb - candidate_gd.tb()
                        if (not (components == 1 and candidate_gd.r() != minus_r)) and not(tb_diff >= components):
                            tree_plus_next.append(new_kw)
                            
                            t_plus_pinches += 1
                            if(gd_hash in tree_minus_seen):
                                found = True
                                break
        
      
            # Death moves allowed from + dir
            if(knotwrapper.n_deaths < max_deaths):
                for (row_idx, col_idx) in ok_death_moves:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.Legendrian_Death(row_idx, col_idx)
                    
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'a link at {(row_idx, col_idx)} died.', deaths=1)
                        tree_plus_next.append(new_kw)
                        tree_plus_seen[gd_hash] = new_kw

                        t_plus_deaths += 1
                        if(gd_hash in tree_minus_seen):
                            found = True
                            break
            

            if(knotwrapper.n_stabilizations < max_stabilizations and knotwrapper.n < max_grid_size):
                for row_idx, col_idx, dir_name_idx, the_letter in ok_stab:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.stabilization(row_idx, col_idx, the_letter, dir_name_idx)
                    assert(candidate_gd.n <= max_grid_size)
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):
                        
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'Performed {the_letter}:{dirs_names[dir_name_idx]} stabilization at {(row_idx, col_idx)}', stabilizations=1)
                        tree_plus_next.append(new_kw)
                        tree_plus_seen[gd_hash] = new_kw

                        t_plus_stabilizations += 1
                        if(gd_hash in tree_minus_seen):
                            found = True
                            break

            if(knotwrapper.n_destabilizations < max_destabilizations):
                for row_idx, col_idx, dir_name_idx, the_letter in ok_destab:
                    candidate_gd = knotwrapper.gd.copy_gd()
                    candidate_gd.destabilization(row_idx, col_idx, the_letter, dir_name_idx)
                    
                    gd_hash = candidate_gd.hash()
                    if(gd_hash not in tree_plus_seen):
                        new_kw = produce_new_kw(candidate_gd, knotwrapper, f'Performed {the_letter}:{dirs_names[dir_name_idx]} destabilization at {(row_idx, col_idx)}', destabilizations=1)
                        tree_plus_next.append(new_kw)
                        tree_plus_seen[gd_hash] = new_kw
                        t_plus_destabilizations += 1

                        if(gd_hash in tree_minus_seen):
                            found = True
                            break
  

        
        log(f't_minus:\nbirths: {t_minus_births}\nstabilizations: {t_minus_stabilizations}\ndestabilizations: {t_minus_destabilizations}\ncopinches: {t_minus_copinches}\ncommutations: {(t_minus_r_commutations,t_minus_c_commutations)}')
        log(f't_plus:\ndeaths: {t_plus_deaths}\nstabilizations: {t_plus_stabilizations}\ndestabilizations: {t_plus_destabilizations}\npinches: {t_plus_pinches}\ncommutations: {(t_plus_r_commutations,t_plus_c_commutations)}')
      
        intersecting = set(tree_minus_seen).intersection(set(tree_plus_seen))
        if(len(intersecting) > 0): # We've found one
            log("Found:")
            break
        
        tree_minus_current = []
        tree_plus_current = []
        for item in tree_minus_next:
            tree_minus_current.append(item)
        for item in tree_plus_next:
            tree_plus_current.append(item)

    if(len(intersecting) >= 1):
        log('The move history: ')
        hashvalue = list(intersecting)[0]
        log('Move history from the MINUS (-) direction:\n')
        for item in tree_minus_seen[hashvalue].move_history:
            log(str(item))

        log('Move history from the PLUS (+) direction:\n')
        for item in tree_plus_seen[hashvalue].move_history:
            log(str(item))
        

        log('Grid history from the MINUS (-) direction:\n')
        for item_idx, item in enumerate(tree_minus_seen[hashvalue].grid_history):
            log("\n" + str(marking_tuple_to_grid(item)))
            if(item_idx <= len(tree_minus_seen[hashvalue].move_history)-1):
                log("\n" + str(tree_minus_seen[hashvalue].move_history[item_idx]))
       
        log('Grid history from the PLUS (+) direction:\n')
        for item_idx, item in enumerate(reversed(tree_plus_seen[hashvalue].grid_history)):
            log("\n" + str(marking_tuple_to_grid(item)))
            if(len(tree_plus_seen[hashvalue].move_history) - item_idx >= 0 and (len(tree_plus_seen[hashvalue].move_history) - item_idx -1 >= 0)):
                log("\n" + str(tree_plus_seen[hashvalue].move_history[len(tree_plus_seen[hashvalue].move_history) - item_idx -1]))
              
    return tree_minus_seen, tree_plus_seen
