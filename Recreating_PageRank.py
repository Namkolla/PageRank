################################################################
# Importing data into a matrix

import numpy as np 
import pandas as pd
import time

start=time.time()

def text_to_mat(file_name):
	f = open (file_name, 'r')
	l = [list(map(int,line.split(','))) for line in f]
	l2 = np.array(l)

	# Creating an adjacency matrix
	n = np.max(l2[:,0])
	mat_z1 = np.zeros(shape=(n,n))
	for row in l2:
		i,j,v = row
		i = i-1
		j = j-1
		mat_z1[i,j] = v
	f.close()
	return mat_z1

mat_z = text_to_mat('links.txt')

################################################################
# Set constants

alpha = 0.85
epsilon = 0.00001

################################################################
# Modifying the adjacency matrix

np.fill_diagonal(mat_z, 0)
mat_z = mat_z.astype(float)

col_sums = np.sum(mat_z, axis=0)
a = col_sums.shape[0]
col_sums = np.reshape(col_sums,(1,a))
i, j = mat_z.shape
norm_mat = np.repeat(col_sums,i,axis=0)
norm_mat = norm_mat.astype(float)
mat_h = mat_z/norm_mat
mat_h = mat_h.astype(float)

################################################################
# Identifying the dangling nodes

dnglg = np.zeros(shape=(i,j))
k = 0
for column in mat_h.T:
	if np.isnan(np.min(column)): 
		dnglg[k] = 1
	else: 
		dnglg[k] = 0
	k = k + 1
dnglg = dnglg.T[1,]

################################################################
# Create article vector

art_a = np.full((10747,1),1/10747)

################################################################
# Create initial start vector
pi_naut = np.zeros(shape=(i,1))
pi_naut = pi_naut + (1/i)
pi_naut = pi_naut.astype(float)

################################################################
# Calculate the influence vector
mat_h = np.nan_to_num(mat_h)

a = dnglg.shape[0]
dnglg = np.reshape(dnglg,(1,a))


def find_inf_vector(H_mat, alpha, dangling, eps, art_vector, inf_vector,count):
	left_side = np.dot(H_mat,inf_vector)*alpha

	paren = (np.dot(dangling,inf_vector) * alpha)+(1-alpha)

	right_side = paren[0,0]*art_vector

	new_inf_vector = left_side + right_side

		# Calculate residuals
	res = new_inf_vector - inf_vector
	res = np.absolute(res)
	res = np.sum(res)
	if res < eps: 
		return new_inf_vector, count
	else: 
		new_inf_vector, count = find_inf_vector(H_mat, alpha, dangling, eps, art_vector, new_inf_vector,count+1)
		return new_inf_vector, count


eigen_vec_P, count = find_inf_vector(mat_h, alpha, dnglg, epsilon, art_a, pi_naut,0)

################################################################
# Calculating the eigenfactor (EF) score

numerator = np.dot(mat_h,eigen_vec_P)
tot_sum = np.sum(numerator, axis=0)
i, j = numerator.shape
norm_vec = np.repeat(tot_sum,i,axis=0)
c = norm_vec.shape[0]
norm_vec = np.reshape(norm_vec,(c,1))

EF = numerator/norm_vec * 100

EF_df = pd.DataFrame(EF)
print('Final EF data frame (sorted): ',EF_df.sort_values(by=[0],axis=0, ascending = False))

print('No. of iterations taken: ', count)

end = time.time()
print('Time it took was: ',end-start)

################################################################

# Printed journal numbers are going to be 1 less than actual number (in txt file) 
	# because of zero-based indexing 

# 			Journal no.: 
# Rank:		(must add 1 to each)		EF score:

# 1  		8929  						1.108406
# 2 		724   						0.247404
# 3 		238   						0.243826
# 4 		6522  						0.235179
# 5 		6568  						0.226093
# 6 		6696  						0.225262
# 7 		6666  						0.216702
# 8 		4407  						0.206476
# 9 		1993  						0.201441
# 10 		2991  						0.185041
# 11 		5965  						0.182752
# 12 		6178  						0.180756
# 13 		1921  						0.175077
# 14 		7579  						0.170448
# 15 		899   						0.170209
# 16 		1558  						0.168001
# 17 		1382  						0.163565
# 18 		1222  						0.150742
# 19 		421   						0.149372
# 20 		5001  						0.149007

# No. of iterations taken was 33

# Time it took was 35.54 seconds (includes reading the file & creating the matrix)
