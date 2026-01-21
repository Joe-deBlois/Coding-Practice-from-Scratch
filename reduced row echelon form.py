import re
import copy

#input rules: 
#must be a string in the form ax_1 +/- ... +/- yx_n = z
#spaces do not matter: e.g. 4x_1 - x_2 = 4 == 4x_1-x_2=4

#GOALS: get more comfortable with regex,& practice programming basic concept from scratch

###################################
#           PIPELINE              #
###################################

def row_echelon_form(*args, max_dimensions):
    #format rows as numbers & put in row echelon form
    #outcome in the form A = [[a, ..., b], [c, ..., d], etc.]
    
    A = []

    for arg in args:
        #step 1: break up into lhs & rhs
        lhs, rhs = arg.split('=')
        rhs = float(rhs.replace(" ", ""))
                    
        #step 2: break up elements, keeping negatives
        elements = re.split(r'[+=]|(?=[-])', lhs)
    
        #step 2: trim whitespace
        def remove_whitespace(e): 
            return e.replace(" ", "")
        elements = list(map(remove_whitespace, elements))
        
       
        #step 3: isolate coefficients; set 0 for dimensions not included
        row = [] #will have length =  max_dimensions + 1
        for i in range(1, max_dimensions +1):
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
        row.append(rhs)
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
    
    A_new = copy.deepcopy(A)
    while True:
        old_A = copy.deepcopy(A_new)

        # Step 1: find pivots
        pivot_search(A_new, max_dimensions, pivots, rows)
        print(f"Pivots: {pivots}")
        pivot_values = []
        for pivot in pivots: 
            pivot_values.append(A_new[pivot[0]][pivot[1]])
        print(f"Pivot values: {pivot_values}")

        # Step 2: swap rows to bring pivots on top
        swap_rule(A_new, max_dimensions, rows, pivots)

        #pivot_search(A_new, max_dimensions, pivots, rows)
        #pivot_values = []
        #for pivot in pivots: 
        #    pivot_values.append(A_new[pivot[0]][pivot[1]])
        #print(f"Pivot values: {pivot_values}")
        

        #Step 3: Forward addition rule
        forward_addition_rule(A_new, pivots, max_dimensions, rows)

        #pivot_search(A_new, max_dimensions, pivots, rows)
        #pivot_values = []
        #for pivot in pivots: 
        #    pivot_values.append(A_new[pivot[0]][pivot[1]])
        #print(f"Pivot values: {pivot_values}")
       
        # Step 4: normalize pivots to 1
        multiplication_rule(A_new, max_dimensions, rows, pivots)

        #pivot_search(A_new, max_dimensions, pivots, rows)
        #pivot_values = []
        #for pivot in pivots: 
        #    pivot_values.append(A_new[pivot[0]][pivot[1]])
        #print(f"Pivot values: {pivot_values}")


        # Step 5: Backward addition rule
        backward_addition_rule(A_new, pivots, max_dimensions, rows)

        #pivot_search(A_new, max_dimensions, pivots, rows)
        #pivot_values = []
        #for pivot in pivots: 
        #    pivot_values.append(A_new[pivot[0]][pivot[1]])
        #print(f"Pivot values: {pivot_values}")

        
        if A_new == old_A:
            break
    return A_new
       
        




def pivot_search(B, max_dimensions, pivots, rows, tol=1e-12):
    pivots.clear()
    current_row = 0

    for col in range(max_dimensions):
        pivot_row = None
        for row in range(current_row, rows):
            if abs(B[row][col]) > tol:  # handle floating point
                pivot_row = row
                break

        if pivot_row is not None:
            pivots.append((pivot_row, col))
            current_row += 1
            if current_row == rows:
                break
  

def swap_rule(C, max_dimensions, rows, pivots):
    for i, (pivot_row, pivot_col) in enumerate(pivots):
        if pivot_row != i:
            C[i], C[pivot_row] = C[pivot_row], C[i]
            pivots[i] = (i, pivot_col)
            print(f"Swapped rows {i} and {pivot_row}")
    return C
    

def multiplication_rule(D, max_dimensions, rows, pivots):
    #keep track of which pivot rows have been normalized
    normalized_pivots = set()

    for pivot in pivots:
        if pivot[0] in normalized_pivots:
            continue
        if D[pivot[0]][pivot[1]] != 1 and D[pivot[0]][pivot[1]] != 0:  
            c = 1/D[pivot[0]][pivot[1]] 
            print(f"pivot {D[pivot[0]][pivot[1]]} * c: {c} = 1")
            print(f"Multiplied row {pivot[0]} by a factor {c}") 
            for element_idx in range(max_dimensions + 1):
                D[pivot[0]][element_idx] = D[pivot[0]][element_idx]*c
            
        normalized_pivots.add(pivot[0])
    return D

def forward_addition_rule(E, pivots, max_dimensions, rows):
    for pivot_row, pivot_col in pivots:
        pivot_value = E[pivot_row][pivot_col]
        for row in range(pivot_row + 1, rows):
            factor = E[row][pivot_col] / pivot_value if abs(pivot_value) > 1 else 0
            if abs(factor) > 0:
                print(f"\nEliminating entry E[{row},{pivot_col}] = {E[row][pivot_col]}")
                print(f"Using pivot E[{pivot_row},{pivot_col}] = {pivot_value}")
                print(f"Row factor: {factor}")
                for col in range(max_dimensions + 1):
                    old_val = E[row][col]
                    E[row][col] -= factor * E[pivot_row][col]
                    if abs(E[row][col]) < 0:
                        E[row][col] = 0
                    print(f"  E[{row},{col}]: {old_val} - {factor}*{E[pivot_row][col]} = {E[row][col]}")
    return E


def backward_addition_rule(E, pivots, max_dimensions, rows):
    for pivot_row, pivot_col in reversed(pivots):
        pivot_value = E[pivot_row][pivot_col]
        for row in range(pivot_row - 1, -1, -1):  # rows above the pivot
            factor = E[row][pivot_col] / pivot_value if abs(pivot_value) > 1 else 0
            if abs(factor) > 0:
                print(f"\nEliminating entry E[{row},{pivot_col}] = {E[row][pivot_col]}")
                print(f"Using pivot E[{pivot_row},{pivot_col}] = {pivot_value}")
                print(f"Row factor: {factor}")
                for col in range(len(E[0])):  # all columns
                    old_val = E[row][col]
                    E[row][col] -= factor * E[pivot_row][col]
                    if abs(E[row][col]) < 0:
                        E[row][col] = 0
                    print(f"  E[{row},{col}]: {old_val} - {factor}*{E[pivot_row][col]} = {E[row][col]}")
    return E


            

     
#################################################
#                    EXECUTION                  #
#################################################

#equation1_1 = "x_1 + 2x_2 + x_3 + x_4 = 7"
#equation2_1 = "2x_1 + 4x_2 + 6x_4 = 4"
#equation3_1 = "x_1+2x_2+2x_3-x_4=12"

#print("---------------------------------------------------------------------")
#A = row_echelon_form(equation1_1, equation2_1, equation3_1, max_dimensions = 4)
#for row in A: 
    print(row)
#A = reduced_row_echelon_form(A, 4)
#for row in A: 
    print(row)
#print("-------------------------------------------------------------------")

equation1_2 = "x_1 +x_2 +x_3 =3"
equation2_2 = "x_1 +2x_2 +3x_3=0"
equation3_2 = "x_1 + 3x_2 + 4x_3 = -2"

print("---------------------------------------------------------------------")
B = row_echelon_form(equation1_2, equation2_2, equation3_2, max_dimensions = 3)
for row in B: 
    print(row)
B = reduced_row_echelon_form(B, 3)
for row in B: 
    print(row)
print("-------------------------------------------------------------------")

equation1_3 = "x_1 +2x_2 +x_3 +x_4 = 8"
equation2_3 = "x_1 +2x_2 + 2x+3 - x_4 = 12"
equation3_3 = "2x_1 + 4x_2 + 6x_4 = 4"

print("---------------------------------------------------------------------")
C = row_echelon_form(equation1_3, equation2_3, equation3_3, max_dimensions = 4)
for row in C: 
    print(row)
C = reduced_row_echelon_form(C, 3)
for row in C: 
    print(row)
print("-------------------------------------------------------------------")