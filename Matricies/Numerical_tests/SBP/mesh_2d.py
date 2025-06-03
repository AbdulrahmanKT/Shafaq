
###################################
#---------Element-----------------#
###################################
class Element: 
    def __init__(self, index, time , solution, left_bound, right_bound, top_bound, bottom_bound, lgl):
        """
        This class defines a single element. 
        - index: ID of the element 
        - time: Time step
        - solution: solution values 
        - bounds: the physical boundaries of the element
        - lgl: is the set of the lgl points that were used to define the SBP operators
        """
        
        
        self.index = index
        self.time = time
        self.solution = solution
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.top_bound = top_bound
        self.bottom_bound = bottom_bound
        self.lgl = lgl 
        
        
        
    def print_computational_points(self): 
        X, Y = np.meshgrid(self.lgl,self.lgl)
        return X, Y
     
    def print_box(self):
        t = np.array([[self.left_bound, self.right_bound],[self.top_bound,self.top_bound]])
        b = np.array([[self.left_bound, self.right_bound],[self.bottom_bound,self.bottom_bound]])
        l = np.array([ [self.left_bound, self.left_bound], [self.bottom_bound,self.top_bound]])
        r = np.array([[self.right_bound, self.right_bound], [self.bottom_bound,self.top_bound]])
        return plt.plot(t[0],t[1]),plt.plot(b[0],b[1]),  plt.plot(l[0],l[1]), plt.plot(r[0],r[1])
    
    def print_box_with_nodes(self):
        t = np.array([[self.left_bound, self.right_bound],[self.top_bound,self.top_bound]])
        b = np.array([[self.left_bound, self.right_bound],[self.bottom_bound,self.bottom_bound]])
        l = np.array([ [self.left_bound, self.left_bound], [self.bottom_bound,self.top_bound]])
        r = np.array([[self.right_bound, self.right_bound], [self.bottom_bound,self.top_bound]])
        x     = ((self.right_bound-self.left_bound)/2)*self.lgl + ((self.right_bound+self.left_bound)/2)*np.ones_like(self.lgl) 
        y     = ((self.top_bound-self.bottom_bound)/2)*self.lgl + ((self.top_bound+self.bottom_bound)/2)*np.ones_like(self.lgl) 
        X, Y  = np.meshgrid(x, y)
        return plt.plot(t[0],t[1]),plt.plot(b[0],b[1]),  plt.plot(l[0],l[1]), plt.plot(r[0],r[1]), plt.plot(X, Y, "+")
        
        
            
    def print_nodes_physical(self):
        x     = ((self.right_bound-self.left_bound)/2)*self.lgl + ((self.right_bound+self.left_bound)/2)*np.ones_like(self.lgl) 
        y     = ((self.top_bound-self.bottom_bound)/2)*self.lgl + ((self.top_bound+self.bottom_bound)/2)*np.ones_like(self.lgl) 
        X, Y  = np.meshgrid(x, y)
        return X, Y
    
    def Jacob(self): 
        A_physical = (self.right_bound - self.left_bound)*(self.top_bound - self.bottom_bound)
        S_x = (self.right_bound - self.left_bound)/2
        S_y = (self.top_bound - self.bottom_bound)/2  
        inv_jacobian = (S_x*S_y)**(-1)
        return inv_jacobian, A_physical
    

###################################
#---------Element-----------------#
###################################

###################################
#---------Mesh--------------------#
###################################
### Creating the class for the mesh element

class Mesh:
    def __init__(self, x_min, x_max,  y_max, y_min , nex, ney, n):
        lgl = np.zeros(n+1)
        lgl[:]= lgl(n)[0,:]
        self.lgl     = lgl
        self.x_min   = x_min
        self.x_max   = x_max
        self.y_min   = y_min
        self.y_max   = y_max
        self.nex     = nex
        self.ney     = ney
        self.elements = [[None for _ in range(ney)] for _ in range(nex)]  # 2D list of elements
        #self.generate_mesh()
        self.t = 0
        
        
    def generate_mesh(self):
        """
        Creates elements and assigns them to the mesh.
        """
       

        self.solution = None
        dx = (self.x_max - self.x_min) / self.nex
        dy = (self.y_max - self.y_min) / self.ney
        index = 0 # For intializing element IDs 

        for i in range(self.nex):
            for j in range(self.ney):
                left_bound = self.x_min + i * dx
                right_bound = self.x_min + (i+1) * dx
                bottom_bound = self.y_min + j * dy
                top_bound = self.y_min + (j+1) * dy

                element = Element( index, self.t , self.solution, left_bound, right_bound, top_bound, bottom_bound, lgl)
                self.elements[i][j] = element
                index += 1

    def get_element(self, i, j):
        """
        Retrieves an element by its (i, j) position in the grid.
        """
        return self.elements[i][j]
    def print_all_elements(self): 
        k = []
        plt.figure(figsize=(6,6))  # Create a figure
        for i in range(self.nex):
            for j in range(self.ney): 
                k = self.elements[i][j]
                k.print_box_with_nodes()
                
                
    #def set_solution_at_time(self,solution):
        
    #####################################################
    ########### Element 1D ##############################
    #####################################################
