from vpython import *
import numpy as np
import matplotlib.pyplot as plt

#################################################################################
# This program is using to demo the multiparticle movement inside a box         #
# usually using to explain the ideal gas interaction, or the Brownian motion    #
# in the case of much larger mass of heaver particle compare with lighter       #
# particle. This program is base on the vPython to visualize the simulation     #
#################################################################################


##########################
# definition of function #
##########################

def get_Particle_Distance(Particle_i, Particle_j):
	distance_vec = Particle_i.pos - Particle_j.pos
	distance_sqr = np.power(distance_vec.x, 2) + np.power(distance_vec.y, 2) + np.power(distance_vec.z, 2)
	return np.power(distance_sqr, 0.5)


def Collision(Particle_i, Particle_j, M_i, M_j):
	v1 = Particle_i.v * ((M_i-M_j)/(M_i+M_j)) + Particle_j.v * (2*M_j/(M_i+M_j))
	v2 = Particle_i.v * ((2*M_i)/(M_i+M_j)) + Particle_j.v * ((M_j-M_i)/(M_i+M_j))
	Particle_i.v = v1
	Particle_j.v = v2

def calc_Average_partcicle_V_3d(particle_array):
	nParticle = len(particle_array)
	vx=[]
	vy=[]
	vz=[]
	v_3d=[]
	for i in range(0, nParticle-1):
		vx.append(particle_array[i].v.x)
		vy.append(particle_array[i].v.y)
		vz.append(particle_array[i].v.z)
	for i in range(0, nParticle-1):
		v_3d_sqr = np.power(vx[i], 2) + np.power(vy[i], 2) + np.power(vz[i], 2)
		v_3d_temp = np.power(v_3d_sqr, 0.5)
		v_3d.append(v_3d_temp)
	V_x_average = sum(vx)/nParticle
	V_y_average = sum(vy)/nParticle
	V_z_average = sum(vz)/nParticle

	return sum(v_3d)/nParticle

def get_Average(particle_array):
	nParticle = len(particle_array)
	return sum(particle_array)/nParticle

class Particle_Inf:
	def __init__(self, particle_array):
		self.particle_array = particle_array
		self.nParticle = len(particle_array)

	def vx(self):
		outputarray = []
		for i in range(0, self.nParticle):
			vx = self.particle_array[i].v.x
			outputarray.append(vx)
		return outputarray

	def vy(self):
		outputarray = []
		for i in range(0, self.nParticle):
			vy = self.particle_array[i].v.y
			outputarray.append(vy)
		return outputarray

	def vz(self):
		outputarray = []
		for i in range(0, self.nParticle):
			vz = self.particle_array[i].v.z
			outputarray.append(vz)
		return outputarray

	def v3d(self):
		outputarray = []
		for i in range(0, self.nParticle):
			vx = self.particle_array[i].v.x
			vy = self.particle_array[i].v.y
			vz = self.particle_array[i].v.z
			v3d_sqr = np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2)
			v3d_temp = np.power(v3d_sqr, 0.5)
			outputarray.append(v3d_temp)
		return outputarray

	def average_v3d(self):
		v3d_array = []
		for i in range(0, self.nParticle):
			vx = self.particle_array[i].v.x
			vy = self.particle_array[i].v.y
			vz = self.particle_array[i].v.z
			v3d_sqr = np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2)
			v3d_temp = np.power(v3d_sqr, 0.5)
			v3d_array.append(v3d_temp)

		return sum(v3d_array)/self.nParticle

	def E(self, Mass):
		outputarray = []
		for i in range(0, self.nParticle):
			vx = self.particle_array[i].v.x
			vy = self.particle_array[i].v.y
			vz = self.particle_array[i].v.z
			v3d_sqr = np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2)
			Energy = (1/2)*Mass*v3d_sqr
			outputarray.append(Energy)
		return outputarray

	def TotalE(self, Mass):
		Energy_array = []
		for i in range(0, self.nParticle):
			vx = self.particle_array[i].v.x
			vy = self.particle_array[i].v.y
			vz = self.particle_array[i].v.z
			v3d_sqr = np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2)
			Energy = (1/2)*Mass*v3d_sqr
			Energy_array.append(Energy)
		return sum(Energy_array)



#####################
# Paremeter setting #
#####################
wall_L = 20  # the length of the wall 
thk = 0.5 # thick of the wall

t = 0 # total time
dt = 0.05 # time interval for calculation

m1 = np.array([1]*10) # mass of the first particle mass = 1
m2 = np.array([0.25]*10) # mass of the second particle mass = 0.25
m = np.hstack((m1,m2)) # horizontal stack

##############################################
# physical signatures definition of particle #
##############################################
Time_counter = []

First_Particle_Pos_x = []
First_Particle_Pos_y = []
Second_Particle_Pos_x = []
Second_Particle_Pos_y = []
Third_Particle_Pos_x = []
Third_Particle_Pos_y = []
Fourth_Particle_Pos_x = []
Fourth_Particle_Pos_y = []

particle_1_average_v_list = []
particle_2_average_v_list = []
particle_1_TotalEnergy_list = []
particle_2_TotalEnergy_list = []
System_TotalEnergy_list = []

# create the canvas
scene = canvas(title = "ParticleCollition", width = 600, height = 600, x = 0, y = 0, center = vector(0, 0, 0), backgruond = vector(0, 0.6, 0.6))

##################
# create the box #
##################
wall_left  = box(pos = vector(-wall_L/2, 0, 0), size = vector(thk, wall_L, 1), color = color.green)
wall_right = box(pos = vector(wall_L/2, 0, 0), size = vector(thk, wall_L, 1), color = color.green)
wall_up    = box(pos = vector(0, wall_L/2, 0), size = vector(wall_L, thk, 1), color = color.green)
wall_down  = box(pos = vector(0, -wall_L/2, 0), size = vector(wall_L, thk, 1), color = color.green)

######################
# prepare the graphs #
######################
g1 = graph(title = "Average velocity of particle 1", width = 600, height = 450, x = 0, y = 600, xtitle = "t(s)", ytitle = "blue: y(m), red: v(m/s)")
v_3d = gcurve(graph = g1, color = color.red)


###########################
# Creation of the paticle #
###########################
particle = []
particle_size = []
particle_mass = []
for i in range(0, 5):	
	particle.append(sphere(pos = vector(np.random.uniform(-wall_L/2+thk, wall_L/2-thk), np.random.uniform(-wall_L/2+thk, wall_L/2-thk), 0 ), radius = 0.5, color = color.red))
	particle_size.append(0.5)
	particle_mass.append(1)

for i in range(0, 50):	
	particle.append(sphere(pos = vector(np.random.uniform(-wall_L/2+thk, wall_L/2-thk), np.random.uniform(-wall_L/2+thk, wall_L/2-thk), 0 ), radius = 0.1, color = color.blue))
	particle_size.append(0.1)
	particle_mass.append(0.1)

for i in range(0,5):
	vx = np.random.uniform(-2, 2)*np.random.normal(1, 0.2)
	vy = np.random.uniform(-2, 2)*np.random.normal(1, 0.2)
	particle[i].v = vector(vx, vy, 0)
for i in range(5, 55):
	vx = np.random.uniform(-10, 10)*np.random.normal(1, 0.2)
	vy = np.random.uniform(-10, 10)*np.random.normal(1, 0.2)
	particle[i].v = vector(vx, vy, 0)

# Simulation
while (t<10):
	rate(50)
	Time_counter.append(t)

	for i in range(0, len(particle)):
		particle[i].pos += particle[i].v*dt

		if(particle[i].pos.x >= wall_L/2 - thk - particle_size[i]): # if the particle hit the right boundary
			particle[i].pos.x = wall_L/2 - thk - particle_size[i]
			particle[i].v.x = -particle[i].v.x
		if(particle[i].pos.x <= -wall_L/2 + thk + particle_size[i]): # if the particle hit the left boundary
			particle[i].pos.x = -wall_L/2 + thk + particle_size[i]
			particle[i].v.x = -particle[i].v.x
		if(particle[i].pos.y >= wall_L/2 - thk - particle_size[i]): # if the particle hit the upper boundary
			particle[i].pos.y = wall_L/2 - thk - particle_size[i]
			particle[i].v.y = -particle[i].v.y
		if(particle[i].pos.y <= -wall_L/2 + thk + particle_size[i]): # if the particle hit the down boundary
			particle[i].v.y = -particle[i].v.y + particle_size[i]

		for j in range(0, len(particle)):
			if( j == i ):
				continue
			if( j != i ):
				if( get_Particle_Distance(particle[i], particle[j]) >  particle_size[i]+particle_size[j] ):
					continue
				if( get_Particle_Distance(particle[i], particle[j]) <=  particle_size[i]+particle_size[j] ):
					particle[i].pos + (particle[i].pos - particle[j].pos)*((particle_size[i]+particle_size[j]) - get_Particle_Distance(particle[i], particle[j]))/(2*get_Particle_Distance(particle[i], particle[j]))
					particle[j].pos - (particle[i].pos - particle[j].pos)*((particle_size[i]+particle_size[j]) - get_Particle_Distance(particle[i], particle[j]))/(2*get_Particle_Distance(particle[i], particle[j]))
					Collision(particle[i], particle[j], particle_mass[i], particle_mass[j])

	t += dt
	
	particle_1 = particle[0:5]
	particle_2 = particle[5:54]

	First_Particle_Pos_x.append(particle_1[0].pos.x)
	First_Particle_Pos_y.append(particle_1[0].pos.y)
	Second_Particle_Pos_x.append(particle_1[1].pos.x)
	Second_Particle_Pos_y.append(particle_1[1].pos.y)
	Third_Particle_Pos_x.append(particle_1[2].pos.x)
	Third_Particle_Pos_y.append(particle_1[2].pos.y)
	Fourth_Particle_Pos_x.append(particle_1[3].pos.x)
	Fourth_Particle_Pos_y.append(particle_1[3].pos.y)

	particle_1_Inf = Particle_Inf(particle_1)
	particle_2_Inf = Particle_Inf(particle_2)

	particle_1_average_v = particle_1_Inf.average_v3d()
	particle_2_average_v = particle_2_Inf.average_v3d()
	particle_1_average_v_list.append(particle_1_average_v)
	particle_2_average_v_list.append(particle_2_average_v)

	particle_1_TotalEnergy = particle_1_Inf.TotalE(1)
	particle_2_TotalEnergy = particle_2_Inf.TotalE(0.1)
	particle_1_TotalEnergy_list.append(particle_1_TotalEnergy)
	particle_2_TotalEnergy_list.append(particle_2_TotalEnergy)
	System_TotalEnergy_list.append(particle_1_TotalEnergy + particle_2_TotalEnergy)

	v_3d.plot(pos = (t, particle_1_average_v))


plt.subplot(1,2,1)
plt.hist(particle_1_average_v_list, density=True)
plt.xlabel('average velocity (m/s)')
plt.ylabel('#Events')
plt.title('Red particle RMS V distribution')
plt.grid(True)

plt.subplot(1,2,2)
plt.hist(particle_2_average_v_list, density=True)
plt.xlabel('average velocity (m/s)')
plt.ylabel('#Events')
plt.title('Blue particle RMS V distribution')
plt.grid(True)

plt.show()


plt.plot(Time_counter, particle_1_TotalEnergy_list)
plt.title('Red Particle Totol Energy')
plt.xlabel('Time(s)')
plt.ylabel('Total Energy(J)')
plt.grid(True)

plt.show()

fig = plt.figure()
subplot1 = fig.add_subplot(221)
subplot1.plot(First_Particle_Pos_x, First_Particle_Pos_y)
subplot1.set_title('First Particle x-y diagram')
subplot1.set_xlabel('x position')
subplot1.set_ylabel('y position')
subplot1.grid(True)
plt.margins(0.4)

subplot2 = fig.add_subplot(222)
subplot2.plot(Second_Particle_Pos_x, Second_Particle_Pos_y)
subplot2.set_title('Second Particle x-y diagram')
subplot2.set_xlabel('x position')
subplot2.set_ylabel('y position')
subplot2.grid(True)
plt.margins(0.4)

subplot3 = fig.add_subplot(223)
subplot3.plot(Third_Particle_Pos_x, Third_Particle_Pos_y)
subplot3.set_title('Third Particle x-y diagram')
subplot3.set_xlabel('x position')
subplot3.set_ylabel('y position')
subplot3.grid(True)
plt.margins(0.4)

subplot4 = fig.add_subplot(224)
subplot4.plot(Fourth_Particle_Pos_x, Fourth_Particle_Pos_y)
subplot4.set_title('Fourth Particle x-y diagram')
subplot4.set_xlabel('x position')
subplot4.set_ylabel('y position')
subplot4.grid(True)
plt.margins(0.4)
plt.tight_layout()
plt.savefig('xy_movement_diagram' + '.eps', format='eps')
plt.savefig('xy_movement_diagram')
plt.show()

