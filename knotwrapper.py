

class KnotWrapper:
    def __init__(self, gd=None):
        self.components = 0
        self.move_history = [] # Store moves in verbal form (e.g. "pinch row 4 with row 5")
        self.grid_history = [] # Store moves in grid form

        # Count the number of moves made so far
        self.n_births = 0 
        self.n_deaths = 0
        self.n_stabilizations = 0
        self.n_destabilizations = 0
        self.n_pinches = 0
        self.n_copinches = 0
        self.n_commutations = 0

        if(gd != None):
            self.load_gd(gd)
    
    def validate_gd(self):
        return self.gd.validate()
        
    def load_gd(self, gd):
        self.gd = gd
        self.grid_history.append((gd.X_by_row, gd.O_by_row))
        self.n = gd.n
        assert(self.n >= 2)
        
    def load_move_history(self, prev_kw):
        for old_history_item in prev_kw.move_history:
            self.move_history.append(old_history_item)
            
        self.n_births = prev_kw.n_births
        self.n_deaths = prev_kw.n_deaths
        self.n_stabilizations = prev_kw.n_stabilizations
        self.n_destabilizations = prev_kw.n_destabilizations
        self.n_pinches = prev_kw.n_pinches
        self.n_copinches = prev_kw.n_copinches
        self.n_commutations = prev_kw.n_commutations

    def load_grid_history(self, prev_kw):
        for old_grid_item in prev_kw.grid_history:
            self.grid_history.append(old_grid_item)

    def add_move(self, move):
        self.move_history.append(move)

    def add_grid_history(self, grid):
        self.grid_history.append(grid)

    def hash(self):
        return self.gd.hash()

# Create a new Knotwrapper object after performing a grid move
def produce_new_kw(prev_gd, hist_kw, message,
                   births=0,
                   deaths=0,
                   stabilizations=0,
                   destabilizations=0,
                   pinches=0,
                   copinches=0,
                   commutations=0,
                   ):
    
    new_kw = KnotWrapper()
    
    new_kw.load_grid_history(hist_kw)
    new_kw.load_move_history(hist_kw)
    new_kw.load_gd(prev_gd)

    # Append the new move's message to the move history
    new_kw.add_move(message)
    
    new_kw.n_births += births
    new_kw.n_deaths += deaths
    new_kw.n_stabilizations += stabilizations
    new_kw.n_destabilizations += destabilizations
    new_kw.n_pinches += pinches
    new_kw.n_copinches += copinches
    new_kw.n_commutations += commutations
    
    return new_kw