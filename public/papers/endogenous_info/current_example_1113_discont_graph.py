from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

#This program creates figure 2 in the paper directly: The numbers were created with the program that is used to create table 1 and are copied directly into this program.

euplot=   [1.6950000000000003, 1.6950000007837454, 1.6950000341688998, 1.699487217251449, 1.7026377654647775, 1.7042236359607876, 1.7053179169566866, 1.7058917456497897, 1.7063607496910749, 1.7066187714905698, 1.7067807867005294, 1.7067946319775456, 1.7069249385558181, 1.7069408267095785, 1.7069330084552228, 1.7069080201304765, 1.7068706717180964,1.70686205642, 1.7079033526,1.7077758469307516, 1.707603158545258, 1.7074405234585552, 1.7072869411944369, 1.7071414796714346, 1.7070035115433386, 1.7068722756426848, 1.7067471974729864, 1.7066278105469679] 
piplot=    [0.140820610979981, 0.1423502605255389, 0.14309825349589622, 0.14423643220228888, 0.14610248060713915, 0.1480386011366015, 0.1499021922273973, 0.15161181019141345, 0.15317161870985485, 0.15461538981240763, 0.15592894769653087, 0.15713336985867432, 0.15824962224481087, 0.1592803734041579, 0.16023797980039323, 0.161130666458567, 0.16196550356152328,0.16212608033,0.16220660927, 0.162786459377535, 0.16356753465230622, 0.16429867474961313, 0.16498529017853603, 0.16563198540894675, 0.16624270893691473, 0.16682087027602255, 0.16736943222916367, 0.16789098439491013] 
gamplot=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,0.8, 0.85,0.86,0.865, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
gamplot1=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,0.8, 0.85,0.86]
gamplot2=[0.865, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
euplot1=   [1.6950000000000003, 1.6950000007837454, 1.6950000341688998, 1.699487217251449, 1.7026377654647775, 1.7042236359607876, 1.7053179169566866, 1.7058917456497897, 1.7063607496910749, 1.7066187714905698, 1.7067807867005294, 1.7067946319775456, 1.7069249385558181, 1.7069408267095785, 1.7069330084552228, 1.7069080201304765, 1.7068706717180964,1.70686205642]
euplot2=[1.7079033526,1.7077758469307516, 1.707603158545258, 1.7074405234585552, 1.7072869411944369, 1.7071414796714346, 1.7070035115433386, 1.7068722756426848, 1.7067471974729864, 1.7066278105469679] 

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.88)

par1 = host.twinx()

host.set_xlim(0.1, 1.3)
host.set_ylim(0.14, 0.17)

host.set_xlabel("$\\gamma$")
host.set_ylabel("Profits")
par1.set_ylabel("Utility")

p1, = host.plot(gamplot, piplot, 'b--',label="Profits")
p2, = par1.plot(gamplot1, euplot1,'r-', label="Utility")
p3, =par1.plot(gamplot2, euplot2,'r-')

par1.set_ylim(1.694, 1.71)


host.legend(loc=4)#this sets the legend into the lower right

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.draw()
plt.show()
