import glob
import imageio.v2 as imageio
gan_dir = 'cifar_32'

anim_file = f'Gan_Tut/plots/{gan_dir}/gif.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob(f'Gan_Tut/plots/{gan_dir}/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)