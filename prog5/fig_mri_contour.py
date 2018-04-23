import matplotlib.pyplot as plt
from skimage import measure
import re
import sys
from matplotlib import collections as mc

with open(sys.argv[1], "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read()[13:]
    data = data.decode('utf-8')

data = re.sub("[^0-9][ ]", "", data)
s1 = data.split('\n')
k = 0
x = []
for i in s1:
    s2 = i.split(' ')
    if(s2 != ['']):
        l = 0
        y = []
        for j in s2:
            if(j != ''):
                y.append(int(j))
        x.append(y)

fig, ax = plt.subplots()
im = ax.imshow(x, cmap=plt.cm.gray)
lines = []
threshold = int(sys.argv[2])

for i in range(len(x)):
        for j in range(len(x[i])):
                if(i + 1 < len(x) and j + 1 < len(x[i])):
                    counter = 0
                    if(x[i + 1][j] > threshold):
                        counter = counter + 1
                    if(x[i + 1][j + 1] > threshold):
                        counter = counter + 2
                    if(x[i][j + 1] > threshold):
                        counter = counter + 4
                    if(x[i][j] > threshold):
                        counter = counter + 8

                    if(counter == 1 or counter == 14):
                        lines.append([(1 + j, 0 + i), (2 + j, 1 + i)])
                    elif(counter == 2 or counter == 13):
                        lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])
                    elif(counter == 3 or counter == 12):
                        lines.append([(-1 + j, 0 + i), (1 + j, 0 + i)])
                    elif(counter == 4 or counter == 11):
                        lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])
                    elif(counter == 5):
                        lines.append([(-1 + j, 0 + i), (0 + j, 1 + i)])
                        lines.append([(0 + j, -1 + i), (1 + j, 0 + i)])
                    elif(counter == 6 or counter == 9):
                        lines.append([(0 + j, 1 + i), (0 + j, -1 + i)])
                    elif(counter == 7 or counter == 8):
                        lines.append([(-1 + j, 0 + i), (0 + j, 1 + i)])
                    elif(counter == 10):
                        lines.append([(-1 + j, 0 + i), (0 + j, -1 + i)])
                        lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])


lc = mc.LineCollection(lines, colors="red", linewidths=1)
ax.add_collection(lc)

#contours = measure.find_contours(x, 40)

#fig, ax = plt.subplots()
#ax.imshow(x, cmap=plt.cm.gray)

#for n, contour in enumerate(contours):
#    ax.plot(contour[:, 1], contour[:, 0], linewidth=1.8, color='red')

#ax.axis('image')

#im = ax.imshow(x, cmap=plt.cm.gray)
fig.colorbar(im,ax=ax)
ax.set_title('mri, threshold = ' + sys.argv[2])
plt.savefig(sys.argv[3])
plt.show()