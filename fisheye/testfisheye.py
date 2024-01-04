from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 100
pfov = 90

# img = "images/zoomed_cropped_image.jpg"
# img_out = f"fisheye/result_crop_{dtype}_{format}_{pfov}_{fov}.jpg"

# obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

# # To save image locally 
# obj.convert(outfile=img_out)

# # To use the converted image in memory

# # new_image = obj.convert()


img_out = f"fisheye/result_crop.jpg"
obj = Defisheye(self.File, dtype=self.dtype, format=self.format, fov=self.fov, pfov=self.fov)
obj.convert(outfile=img_out)
new_image = obj.convert()
width, height = new_image.size
print(new_image)
print(width)
print(height)
# Menghitung ukuran baru untuk zoom in 20% dari sisi kiri dan kanan
left_crop = int(width * 0.1)  # 20% dari sisi kiri
right_crop = width - left_crop  # 20% dari sisi kanan
        # Melakukan zoom in dengan cropping pada sisi kiri dan kanan gambar
imageocr = new_image.crop((left_crop, 0, right_crop, height))
imageocr= new_image.rotate(90, expand=True)
imageocr.save(self.zoom)
print("get zoom!")