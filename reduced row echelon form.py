import re

# valid moves: 
# swap rows
# multiply a row by a nonzero scalar
# add a multiple of one row to another
# 
# 
#input rules: 
#must be a string in the form ax_1 +/- ... +/- yx_n = z
#spaces do not matter: e.g. 4x_1 - x_2 = 4 == 4x_1-x_2=4

#GOALS: get more comfortable with regex,& practice programming basic concept from scratch

#equation1 = "x_1 + 2x_2 + x_3 + x_4 = 7"
#equation2 = "x_1+2x_2+2x_3-x_4=12"
#equation3 = "2x_1 + 4x_2 - 6x_4 = 4"

equation1 = "2x_2 + x_3 + x_4 = 7"
equation2 = "x_1+2x_2+2x_3-x_4=12"
equation3 = "2x_1 + 4x_2 - 6x_4 = 4"

def row_echelon_form(*args, max_dimension):
    #format rows as numbers & put in row echelon form
    #outcome in the form A = [[a, ..., b], [c, ..., d], etc.]
    
    A = []

    for arg in args:
        #step 1: break up elements, keeping negatives
        elements = re.split(r'[+=]|(?=[-])', arg)
    
        #step 2: trim whitespace
        def remove_whitespace(e): 
            return e.replace(" ", "")
        elements = list(map(remove_whitespace, elements))
        print(elements)
       
        #step 3: isolate coefficients; set 0 for dimensions not included
        row = [] #will have length =  max_dimension + 1
        for i in range(1, max_dimension+1):
                coef = 0 
                for element in elements[:-1]:
                    if re.search(fr'x_{i}', element):
                       #then variable is present
                       coef_string = re.match(r'^(-?\d*)', element).group(1)
                       
                       if coef_string in (""):
                            coef = 1
                       elif coef_string in ("-"):
                            coef = -1
                       else:
                            coef = float(coef_string)

                       break
                row.append(coef)
        #add the hyperplane position
        row.append(float(elements[-1]))
        A.append(row)

    #sanity check that all rows are the same length:
    sanity_lens = set()
    for row in A: 
        sanity_lens.add(len(row))
    if len(sanity_lens) > 1:
         print("There is a row-length problem with the creation of the row echelon matrix.")
    else: 
         print("Row-echelon matrix created")
         return A
    

        
def reduced_row_echelon_form(A, max_dimensions):
    print(A)

    rows = len(A)

    pivots = [] 
    #tuples (row, element). Max amount == max_dimensions
    
    #iterations with stopping conditions

    #pivot cleaning

def overall_pivot_search (A, max_dimensions, pivots, rows):
    for row in range(0, rows): 
        for element in range(0, max_dimensions):
            #pivot in column 1
            if (A[row][element] == 1) and element == 0:
                check = pivot_row_search(A, max_dimensions, pivots, row, element) 
                
                if check != 0 :
                    pivots.remove(check)
                    pivots.append((row, element))
                else: 
                    pivots.append((row, element))
            #any other pivot
            if A[row][element] == 1 and all(A[row][k] == 0 for k in range(element)):

                check = pivot_col_search(A, max_dimensions, pivots, row, element)
                if check != 0:
                    pivots.remove(check)
                    pivots.append((row, element))

def pivot_row_search(A, max_dimensions, pivots, row, element):
    #does a pivot exist for the given column/row?
    for pivot in pivots:
        if pivot[0] == row:
            return pivot
    return 0

def pivot_col_search(A, max_dimensions, pivots, row, element):
    #does a pivot exist for the given column/row?
    for pivot in pivots:
        if pivot[1] == element:
            return pivot
    return 0
       
                


def swap_rule(A, max_dimensions, rows):
    for row in range(0, rows):
        #print(f"Row: {row}")
        for element in range(0, max_dimensions):
            
            #print(f"Element: {element}")
            #print(f"A[row][element]: {A[row][element]}")
            #print(f"A[row + 1][element]: {A[row+1][element]}")

            if (A[row][element] == 0) and (A[row + 1][element] == 1):
                print(f"Swapping rows {row} and {row + 1}")
                A[row], A[row + 1 ] = A[row + 1], A[row]

def multiplication_rule(A, max_dimensions, rows, pivots):
    for row in range(0, rows):
        for element in range(0, max_dimensions):
            for pivot in pivots:
                if pivot[0] == row:
                    c = 1/A[pivot[0]][pivot[1]]
                    for element in A[row]:
                        A[row][element] = (A[row][element])*c
    return A
         
def addition_rule(A, max_dimensions, rows, pivots):
    for row1 in range(0, rows):
        for row2 in range(row1+1, rows):
        #find the corresponding element that is closest in magnitude
            for element in range(0, max_dimensions):
                magnitudes = () #row, magnitude difference
                magnitudes.append(abs(A[row1][element]) - abs(A[row2][element]))

            #element is negative, add a positive
            if A[row][element] < 0 and 
            #element is positive, subtract a positive
            #element is positive, add a negative
            #element is negative, subtract a negative
    return A


def pivot_cleaning(A, max_dimensions, rows, pivots):
    #last step after pivots are all found
    for pivot in pivots: 
 
            

     
    

 

A = row_echelon_form(equation1, equation2, equation3, max_dimension = 4)
reduced_row_echelon_form(A, 4)