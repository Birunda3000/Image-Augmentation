import Aug

unive = 1
#_______________________________________________________Pipe exemplo 1_______________________________________________________
pipe = Aug.Pipe()

pipe.add( Aug.Skew(prob=unive, skew_type='RANDOM', magnitude=0.9) )
#1
pipe.add( Aug.Invert(prob=unive) )
#2
pipe.add(Aug.Brilho(prob=unive, min_factor= 0.1, max_factor= 10))
#3
pipe.add(Aug.Color(prob=unive, min_factor=-20, max_factor=20))
#4
pipe.add(Aug.Contrast(prob=unive, min_factor=-5, max_factor=5) )
#5
pipe.add(Aug.Flip(prob=unive))
#6
pipe.add(Aug.Shear(prob=unive, max_shear_left = 30, max_shear_right = 30 ))
#7
pipe.add( Aug.Distort(prob=unive, grid_width=4, grid_height=4, magnitude=8) )
#8
pipe.add( Aug.Zoom(prob=unive, min_factor=1, max_factor=9) )
#9
pipe.add(Aug.Zoom_Random(prob=unive, percentage_area=0.5, randomise=True))
#10
pipe.add(Aug.Random_Erasing(prob=unive, rectangle_area=0.3, repetitions=3))
#11
pipe.add( Aug.Shift(prob=unive, horizontal_max=0.8, vertical_max=0.8, randomise=True, fill='nearest') )
#12
pipe.add( Aug.Rotacao(prob=unive, max_left_rotation=90, max_right_rotation=90, fill='edge') )
#13
pipe.add( Aug.Gaussian(prob=unive) )
#14
pipe.add( Aug.Edge(prob=unive) )
#15
pipe.add( Aug.Grey_Erosion(prob=unive) )
#16
pipe.add( Aug.Random_Noise(prob=unive)  )
#17
#_______________________________________________________Pipe exemplo 2_______________________________________________________
pipe2 = Aug.Pipe()
unive_2 = 1
pipe2.add( Aug.Skew(prob=unive_2, skew_type='RANDOM', magnitude=0.5) )
#1
pipe2.add( Aug.Invert(prob=unive_2) )
#2
pipe2.add(Aug.Zoom_Random(prob=unive_2, percentage_area=0.6, randomise=True))
#3
pipe2.add(Aug.Random_Erasing(prob=1, rectangle_area=0.3, repetitions=3))#prob = 1
#4
pipe2.add( Aug.Shift(prob=unive_2, horizontal_max=0.3, vertical_max=0.3, randomise=True, fill='nearest') )
#5
pipe2.add( Aug.Rotacao(prob=unive_2, max_left_rotation=40, max_right_rotation=40, fill='edge') )
#6