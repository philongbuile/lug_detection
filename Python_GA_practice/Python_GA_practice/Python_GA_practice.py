#TEMPLATE_MATCHING_USING_GA
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

n = 3 #size of individual
m = 26 #size of population #must be even num because ò elitism=2
generation = 100

image = Image.open('img.jpg')
templ = Image.open('template_1.jpg')

##things with img
img_list = np.asarray(image).flatten().tolist()
temp_list = np.asarray(image).flatten().tolist()


img_arr = np.asarray(image)


w_img, h_img = image.size
w_temp, h_temp = templ.size

data = []
template = []
pr_img_list = []

count = 0
for i in range(h_img*w_img):
	data.append(img_list[count])
	count+=3
count = 0
for i in range(h_temp*w_temp):
	template.append(temp_list[count])
	count+=3



fitness = []
def find_corr_x_y(x,y):
	n = len(x)
	product = []
	for xi, yi in zip(x,y):   #create alist contain xi*yi
		product.append(xi*yi)
	#calculate fundamental component
	sum_x = sum(x)
	sum_y = sum(y)
	sum_x_sqr = sum_x**2
	sum_y_sqr = sum_y**2
	sum_product_x_y = sum(product)
	x_sqr = []
	for xi in x:
		x_sqr.append(xi**2)
	sum_sqr_x = sum(x_sqr)
	y_sqr = []
	for yi in y:
		y_sqr.append(yi**2)
	sum_sqr_y = sum(y_sqr)
	#calculate correlation coefficient
	nummerator = n*sum_product_x_y - sum_x*sum_y
	denominator_1 = (n*sum_sqr_x - sum_x_sqr)**0.5
	denominator_2 = (n*sum_sqr_y - sum_y_sqr)**0.5
	denominator = denominator_1*denominator_2
	return nummerator/(denominator+0.000000000000000000000000000000000001)
def generate_i():
	return random.randint(0,h_img-h_temp)
def generate_j():
	return random.randint(0,w_img -w_temp)
def generate_r():
	return random.randint(0,180)

def generate_individual():
	indv = []
	indv.append(generate_i())
	indv.append(generate_j())
	indv.append(generate_r())
	return indv


def generate_population():
	population = []
	for _ in range(m):
		population.append(generate_individual())
	return population
def computeFitness(individual,templ):
	kernel = []
	temp =[]
	templ = templ.rotate(individual[2])
	temp_arr = np.asarray(templ) 
	for i in range(h_temp):
		for j in range(w_temp):
			kernel.append(int(img_arr[i+individual[0],j + individual[1],0])) 
			temp.append(int(temp_arr[i,j,0])) 
	result = find_corr_x_y(temp, kernel)
	return result
def selection(sorted_population):
	index1 = random.randint(0,m-1)
	while True:
		index2 = random.randint(0,m-1)
		if index1 != index2:
			break
	individual_s = sorted_population[index1]
	if index2 > index1:
		individual_s = sorted_population[index2]
	return individual_s

def cross_over(individual_1, individual_2, cross_over_rate = 0.8):
	individual1_n = individual_1.copy()
	individual2_n = individual_2.copy()
	for i in range(n):
		if random.random() < cross_over_rate:
			individual1_n[i] = individual_2[i]
			individual2_n[i] = individual_1[i]
	return individual1_n, individual2_n
def mutation(individual, mutation_rate = 0.3):
	individual_m = individual.copy()
	for i in range(n):
		if random.random() < mutation_rate:
			if i == 0:
				individual_m[i] = generate_i()
			if i== 1:
				individual_m[i] = generate_j()
			if i == 2:
				individual_m[i]=generate_r()
	return individual_m
#def sort_population(old_population):
#	for i in range(m):
#		for j in range(i+1,m):
#			if computeFitness(old_population[i])>computeFitness(old_population[j]):
#				temp = old_population[i]
#				old_population[i] = old_population[j]
#				old_population[j] = temp
#	return old_population
def sort_population(population):
	sorted_population = population
	for i in range(m):
		for j in range(i+1,m):
			if (computeFitness(sorted_population[i],templ) > computeFitness(sorted_population[j],templ)):
				temp = sorted_population[i]
				sorted_population[i] = sorted_population[j]
				sorted_population[j] = temp
	return sorted_population
def generate_new_population(old_population, elitism, gen):
	sorted_population = sort_population(old_population)
	if gen%1 ==0:
		a = computeFitness(sorted_population[m-1],templ)
		fitness.append(a)
		print("best gen: ", gen,a, sorted_population[m-1])
	new_population = []
	while len(new_population) < (m - elitism):
		individual_s1 = selection(sorted_population)
		individual_s2 = selection(sorted_population)

		individual_c1, individual_c2 = cross_over(individual_s1,individual_s2)

		individual_m1 = mutation(individual_c1)
		individual_m2 = mutation(individual_c2)

		new_population.append(individual_m1)
		new_population.append(individual_m2)
	
	for indv in sorted_population[m-elitism:]:
		new_population.append(indv.copy())
	return new_population

population = generate_population()

for _ in range(generation):
	population = generate_new_population(population,2, _)

plt.plot(fitness)
plt.show()


sorted_population = sort_population(population)

pr_temp_list=[]
count=0
for i in range(h_temp*w_temp):
	pr_temp_list.append([temp_list[count], temp_list[count+1], temp_list[count+2]])
	count+=3

pr_img_list =[]
count=0
for i in range(h_img*w_img):
	pr_img_list.append([img_list[count],img_list[count+1],img_list[count+2] ])
	count+=3


for i in range(w_temp):
	pr_img_list[sorted_population[m-1][0]*w_img + sorted_population[m-1][1] + i] = [0,0,0]
	pr_img_list[(sorted_population[m-1][0]+h_temp)*w_img + sorted_population[m-1][1] + i] = [0,0,0]	


for i in range(h_temp):
	pr_img_list[(sorted_population[m-1][0]+i)*w_img + sorted_population[m-1][1]] = [0,0,0]
	pr_img_list[(sorted_population[m-1][0]+i)*w_img + sorted_population[m-1][1] + w_temp] = [0,0,0]	
out_img =np.zeros((h_img, w_img, 3), dtype=np.uint8)
for i in range(h_img):
	for j in range(w_img):
		out_img[i,j] = pr_img_list[i*w_img + j]

imgg = Image.fromarray(out_img, 'RGB')
imgg.save('test.jpg')

imgg.show()
image.close()
templ.close()


