import Aug

pipe = Aug.Pipe()

pipe.add( Aug.Skew(prob=1, skew_type='RANDOM', magnitude=0.9) )

pipe.add( Aug.Invert(prob=1) )

pipe.add(Aug.Brilho(prob=1, min_factor= 0.1, max_factor= 10))

pipe.add(Aug.Color(prob=1, min_factor=-20, max_factor=20))

pipe.add(Aug.Contrast(prob=1, min_factor=-5, max_factor=5) )

pipe.add(Aug.Flip(prob=1))

pipe.add(Aug.Shear(prob=1, max_shear_left = 30, max_shear_right = 30 ))

pipe.add( Aug.Distort(prob=1, grid_width=4, grid_height=4, magnitude=8) )

pipe.add( Aug.Zoom(prob=1, min_factor=1, max_factor=9) )

pipe.add(Aug.Zoom_Random(prob=1, percentage_area=0.5, randomise=True))

pipe.add(Aug.Random_Erasing(prob=1, rectangle_area=0.3, repetitions=3))

pipe.add( Aug.Shift(prob=1, horizontal_max=0.8, vertical_max=0.8, randomise=True, fill='nearest') )


#pipe.print()