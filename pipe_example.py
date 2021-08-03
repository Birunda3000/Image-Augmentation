import Aug

unive = 1

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