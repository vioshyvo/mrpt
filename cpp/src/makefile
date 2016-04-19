CXX= g++
CPPFLAGS= -Wall -O2 -std=c++11 -g -I/usr/local/include
LDLIBS= -lblas -llapack
OBJS= Contour.o Contours.o knn.o main.o Mrpt.o Result.o 

mrpt: $(OBJS)
	g++ -o ../bin/mrpt $(OBJS) $(LDLIBS)

Contour.o: Result.h Mrpt.h Contour.h knn.h
Contours.o: Result.h knn.h Contour.h Contours.h 
knn.o: knn.h 
main.o: Contours.h Result.h Mrpt.h knn.h
Mrpt.o: knn.h Mrpt.h
Result.o: Result.h 
 
