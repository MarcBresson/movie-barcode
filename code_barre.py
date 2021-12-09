from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
from skimage import io
from PIL import Image
from os import path
import av
import time
import json
import colorsys
import math

def get_dominant_color(pil_img, taille_palette):
	# Resize image to speed up processing
	start_compression_image = time.time()
	img = pil_img.copy()
	img.thumbnail((200, 200),Image.BOX)
	# img_colonne = pil_img.copy()
	# img_colonne.resize((1,1080),Image.BOX)
	end_compression_image = time.time()

	start_palette = time.time()
	# Reduce colors (uses k-means internally)
	paletted = img.convert('P', palette=Image.ADAPTIVE, colors=taille_palette)

	# Find the color that occurs most often
	palette = paletted.getpalette()
	color_counts = sorted(paletted.getcolors(), reverse=True)
	tableau_image = [0]*(len(color_counts))
	for i in range(len(color_counts)):
		palette_index = color_counts[i][1]
		tableau_image[i] = palette[palette_index*3:palette_index*3+3]
	end_palette = time.time()

	return tableau_image,end_compression_image-start_compression_image,end_palette-start_palette

def generer_couleurs_film(chemin, output_folder='result/',taille_palette=256):

	film = av.open(chemin)
	film.streams.video[0].thread_type = 'AUTO'
	cap = cv2.VideoCapture(chemin)
	nbr_total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	cap.release()

	nbr_frame_a_traiter = 1920*2

	# Create the resulting image
	couleurs_dominantes = [0]*nbr_frame_a_traiter
	temps_compression_total = 0
	temps_palette_total = 0
	compteur = 0
	incr = 0

	for frame in film.decode(video=0):
		if compteur>incr*nbr_total_frame/nbr_frame_a_traiter :
			tableau_image,temps_compression,temps_palette = get_dominant_color(frame.to_image(),taille_palette)
			temps_compression_total += temps_compression
			temps_palette_total += temps_palette

			print("frame : "+str(incr)+"/"+str(nbr_frame_a_traiter)+" - "+str(tableau_image[0]))
			couleurs_dominantes[incr]=tableau_image

			incr+=1

		compteur+=1

	nom_fichier = chemin.split("/")[-1].split(".")
	nom_fichier.pop()
	nom_fichier = "_".join(nom_fichier)
	result_folder = path.join(path.dirname(__file__), output_folder)
	with open(path.join(result_folder, nom_fichier+'-data.json'), 'w') as outfile:
		json.dump(couleurs_dominantes, outfile)

	print("temps de compression des images : "+str(temps_compression_total))
	print("temps de génération de la palette : "+str(temps_palette_total))

	return nom_fichier



def generer_images(file, width=1920, height=500,type_image="rectangle", output_folder='results/'):
	result_folder = path.join(path.dirname(__file__), output_folder)

	def comparaison_array(a,operateur,b):
		if len(a)!=len(b):
			return False
		for i in range(len(a)):
			if not eval(str(a[i])+operateur+str(b[i])):
				return False
		return True

	def lum (r,g,b):
		return math.sqrt( .241 * r + .691 * g + .068 * b )

	def step (rgb, repetitions=1):
		r,g,b=rgb
		lum = math.sqrt( .241 * r + .691 * g + .068 * b )
		h, s, v = colorsys.rgb_to_hsv(r,g,b)
		h2 = int(h * repetitions)
		lum2 = int(lum * repetitions)
		v2 = int(v * repetitions)
		if h2 % 2 == 1:
			v2 = repetitions - v2
			lum = repetitions - lum
		return (h2, lum, v2)


	with open(path.join(result_folder, file+'-data.json')) as json_file:
		couleurs_dominantes = json.load(json_file)
	couleurs_dominantes_step = [[255,255,255]]*len(couleurs_dominantes)
	couleurs_dominantes_lum = [[255,255,255]]*len(couleurs_dominantes)
	couleurs_dominantes_hue = [[255,255,255]]*len(couleurs_dominantes)

	for i in range(len(couleurs_dominantes)):
		if type(couleurs_dominantes[i]) is int:
			palette_d_image = [couleurs_dominantes[i],couleurs_dominantes[i],couleurs_dominantes[i]]
		else:
			palette_d_image = couleurs_dominantes[i].copy()
		palette_d_image.sort(key=lambda rgb: step(rgb,8))
		couleurs_dominantes_step[i]=palette_d_image.copy()
		palette_d_image.sort(key=lambda rgb: lum(*rgb))
		couleurs_dominantes_lum[i]=palette_d_image.copy()
		palette_d_image.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb) )
		couleurs_dominantes_hue[i]=palette_d_image.copy()
	print("tri des couleurs terminé")


	if type_image=="rectangle":
		code_barre = np.zeros((height, width, 3), dtype=np.uint8)
		code_barre_step = np.zeros((height, width, 3), dtype=np.uint8)
		code_barre_lum = np.zeros((height, width, 3), dtype=np.uint8)
		code_barre_hue = np.zeros((height, width, 3), dtype=np.uint8)
		code_barre_simple = np.zeros((height, width, 3), dtype=np.uint8)
		for i in range(width):
			nouvel_indice = int(i*(len(couleurs_dominantes)-1)/width)
			taille_palette_step = len(couleurs_dominantes_step[nouvel_indice])
			taille_palette_lum = len(couleurs_dominantes_lum[nouvel_indice])
			taille_palette_hue = len(couleurs_dominantes_hue[nouvel_indice])

			for ii in range(height):
				code_barre[ii][i]=couleurs_dominantes[nouvel_indice][int(ii*taille_palette_step/height)]
				code_barre_step[ii][i]=couleurs_dominantes_step[nouvel_indice][int(ii*taille_palette_step/height)]
				code_barre_lum[ii][i]=couleurs_dominantes_lum[nouvel_indice][int(ii*taille_palette_lum/height)]
				code_barre_hue[ii][i]=couleurs_dominantes_hue[nouvel_indice][int(ii*taille_palette_hue/height)]
				code_barre_simple[ii][i] = max(couleurs_dominantes[nouvel_indice][0:3])
		
		code_barre = cv2.cvtColor(code_barre, cv2.COLOR_BGR2RGB)
		code_barre_step = cv2.cvtColor(code_barre_step, cv2.COLOR_BGR2RGB)
		code_barre_lum = cv2.cvtColor(code_barre_lum, cv2.COLOR_BGR2RGB)
		code_barre_hue = cv2.cvtColor(code_barre_hue, cv2.COLOR_BGR2RGB)
		code_barre_simple = cv2.cvtColor(code_barre_simple, cv2.COLOR_BGR2RGB)
		# Save the image
		try:
			cv2.imwrite(path.join(result_folder, '%s_code_barre.png' % file), code_barre)
			cv2.imwrite(path.join(result_folder, '%s_code_barre_step.png' % file), code_barre_step)
			cv2.imwrite(path.join(result_folder, '%s_code_barre_lum.png' % file), code_barre_lum)
			cv2.imwrite(path.join(result_folder, '%s_code_barre_hue.png' % file), code_barre_hue)
			cv2.imwrite(path.join(result_folder, '%s_code_barre_simple.png' % file), code_barre_simple)
		except Exception as error:
			print('Image Write Error: %s' % error)
	
	if type_image=="cercle":
		code_barre_cercle = np.zeros((width*2, width*2, 3), dtype=np.uint8)
		precedent_indice = 0
		taille_palette = 1
		for distance_au_centre in range(width):
			resolution_angulaire = 90*1/(np.arctan2(1,max(1,distance_au_centre*1.4))*57.4)
			nouvel_indice=int((width-int(distance_au_centre))/width*(len(couleurs_dominantes_lum)-1))
			# couleur_pix_precedent = couleurs_dominantes_lum[precedent_indice]
			# taille_palette_precedent = taille_palette
			# precedent_indice = nouvel_indice
			couleur_pix = couleurs_dominantes_lum[nouvel_indice]
			taille_palette = len(couleurs_dominantes_lum[nouvel_indice])
			print(distance_au_centre)
			for angle_iter in range(int(resolution_angulaire)):
				angle = (angle_iter/resolution_angulaire*90.5)/57.4
				a=couleur_pix[int(angle*57.4/361*taille_palette)-1]
				# b=couleur_pix_precedent[int(angle*57.4/361*taille_palette_precedent)-1]
				for interval in range(-1,1):
					code_barre_cercle[width+int(math.cos(angle)*(distance_au_centre+interval/2))][width+int(math.sin(angle)*(distance_au_centre+interval/2))]=a
					code_barre_cercle[width+int(math.cos(angle)*(distance_au_centre+interval/2))][width+int(math.sin(angle+math.pi/2)*(distance_au_centre+interval/2))]=a
					code_barre_cercle[width+int(math.cos(angle)*(distance_au_centre+interval/2))][width+int(math.sin(angle+math.pi)*(distance_au_centre+interval/2))]=a
					code_barre_cercle[width+int(math.cos(angle)*(distance_au_centre+interval/2))][width+int(math.sin(angle+3*math.pi/2)*(distance_au_centre+interval/2))]=a
		code_barre_cercle = cv2.cvtColor(code_barre_cercle, cv2.COLOR_BGR2RGB)
		try:
			cv2.imwrite(path.join(result_folder, '%s_cercle.jpg' % file), code_barre_cercle)
		except Exception as error:
			print('Image Write Error: %s' % error)

ici = "."
root = Tk()
root.withdraw()
fichiers =  filedialog.askopenfilename(initialdir = ici ,title = "Select file",filetypes = (("movies files","*.mkv"),("all files","*.*")),multiple=True)

count=0
utilisation = []
commandes_generales = []
for fichier in fichiers:
	print("["+ str(count+1) +"/"+ str(len(fichiers)) +"] "+ fichier)
	extension_fichier = fichier.split(".")[-1]
	if extension_fichier==".json":
		nom_fichier=fichier.split("-data.json")[0]
		generer_images(nom_fichier,1920,512)
	else:
		nom_fichier = generer_couleurs_film(fichier,"results/")
		generer_images(nom_fichier,1920,512)